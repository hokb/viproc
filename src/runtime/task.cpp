#include "task.hpp"

#include <cassert>
#include <cfenv>
#include <exception>

namespace viproc {

Task::Task(std::shared_ptr<Kernel> kernel, const char* issue_site)
    : kernel_(std::move(kernel)), issue_site_(issue_site) {}

void Task::wait_completed() const {
    if (completed()) {
        return;
    }
    {
        // Under the lock: either execute() sees the flag and notifies, or it
        // completed before and the loop below does not block.
        std::lock_guard lock(mutex_);
        waiters_ = true;
    }
    TaskState s = state();
    while (s != TaskState::Completed) {
        state_.wait(s, std::memory_order_acquire);
        s = state();
    }
}

void Task::advance(TaskState next) {
    [[maybe_unused]] const TaskState prev = state_.load(std::memory_order_relaxed);
    assert(next > prev && "task states only move forward");
    state_.store(next, std::memory_order_release);
}

bool Task::add_consumer(const std::shared_ptr<Task>& consumer) {
    std::lock_guard lock(mutex_);
    if (state_.load(std::memory_order_relaxed) == TaskState::Completed) {
        return false;
    }
    consumers_.push_back(consumer);
    return true;
}

void Task::add_upstream_error(const TaskError& error) {
    std::lock_guard lock(mutex_);
    if (!upstream_error_) {
        upstream_error_ = error;
    }
}

bool Task::input_done(const Task* producer) {
    if (producer != nullptr && producer->error_) {
        std::lock_guard lock(mutex_);
        if (!upstream_error_) {
            upstream_error_ = producer->error_;
        }
    }
    return pending_.fetch_sub(1, std::memory_order_acq_rel) == 1;
}

std::vector<std::shared_ptr<Task>> Task::execute() {
    // No lock: every write of upstream_error_ happens before the release
    // decrement of pending_ by its writer, and this thread observed the final
    // decrement (or got the task from the thread that did).
    if (upstream_error_) {
        error_ = upstream_error_;
    }
    if (!error_) {
        const std::span<const Operand> inputs(operands_.data(), ninputs_);
        const std::span<Operand> outputs(operands_.data() + ninputs_, operands_.size() - ninputs_);
        for (Operand& out : outputs) {
            out.buffer->ensure_allocated();
        }
        advance(TaskState::Allocated);
        std::feclearexcept(FE_ALL_EXCEPT);
        KernelResult r;
        try {
            r = kernel_->run(inputs, outputs);
        } catch (const std::exception& e) {
            r = KernelResult{1, e.what(), 0};
        } catch (...) {
            r = KernelResult{1, "unknown exception in kernel", 0};
        }
        fp_flags_ = r.fp_flags | std::fetestexcept(FE_ALL_EXCEPT);
        if (r.status != 0) {
            error_ = TaskError{std::move(r.message), issue_site_};
        }
    }

    // Done reading: release the inputs (and the async_reads they hold).
    for (std::size_t i = 0; i < ninputs_; ++i) {
        Operand& in = operands_[i];
        in.buffer->async_reads.fetch_sub(1, std::memory_order_release);
    }
    operands_.clear();
    kernel_.reset();

    std::vector<std::shared_ptr<Task>> consumers;
    bool notify;
    {
        std::lock_guard lock(mutex_);
        state_.store(TaskState::Completed, std::memory_order_release);
        consumers.swap(consumers_);
        notify = waiters_;
    }
    if (notify) {
        state_.notify_all();
    }

    std::vector<std::shared_ptr<Task>> runnable;
    for (auto& c : consumers) {
        if (c->input_done(this)) {
            runnable.push_back(std::move(c));
        }
    }
    return runnable;
}

} // namespace viproc
