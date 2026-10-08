#include "task.hpp"

#include <cassert>
#include <cfenv>
#include <exception>

namespace viproc {

Task::Task(std::shared_ptr<Kernel> kernel, std::string issue_site)
    : kernel_(std::move(kernel)), issue_site_(std::move(issue_site)) {}

void Task::wait_completed() const {
    TaskState s = state();
    while (s != TaskState::Completed) {
        state_.wait(s, std::memory_order_acquire);
        s = state();
    }
}

void Task::set_operands(std::vector<Operand> inputs, std::vector<Operand> outputs) {
    inputs_ = std::move(inputs);
    outputs_ = std::move(outputs);
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
    {
        std::lock_guard lock(mutex_);
        if (upstream_error_) {
            error_ = upstream_error_;
        }
    }
    if (!error_) {
        for (Operand& out : outputs_) {
            out.buffer->ensure_allocated();
        }
        advance(TaskState::Allocated);
        std::feclearexcept(FE_ALL_EXCEPT);
        KernelResult r;
        try {
            r = kernel_->run(inputs_, outputs_);
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
    for (Operand& in : inputs_) {
        in.buffer->async_reads.fetch_sub(1, std::memory_order_release);
    }
    inputs_.clear();
    outputs_.clear();
    kernel_.reset();

    std::vector<std::shared_ptr<Task>> consumers;
    {
        std::lock_guard lock(mutex_);
        state_.store(TaskState::Completed, std::memory_order_release);
        consumers.swap(consumers_);
    }
    state_.notify_all();

    std::vector<std::shared_ptr<Task>> runnable;
    for (auto& c : consumers) {
        if (c->input_done(this)) {
            runnable.push_back(std::move(c));
        }
    }
    return runnable;
}

} // namespace viproc
