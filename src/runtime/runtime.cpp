#include "runtime.hpp"

#include <algorithm>

namespace viproc {

Runtime::Runtime(RuntimeOptions options) : options_(std::move(options)) {
    std::size_t n = options_.workers;
    if (n == 0) {
        n = std::max(1u, std::thread::hardware_concurrency());
    }
    if (options_.max_active_tasks == 0) {
        options_.max_active_tasks = 1;
    }
    workers_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

Runtime::~Runtime() {
    wait_all();
    {
        std::lock_guard lock(queue_mutex_);
        stopping_ = true;
    }
    queue_cv_.notify_all();
    for (std::thread& t : workers_) {
        t.join();
    }
}

ArrayPtr Runtime::wrap(Layout layout, std::shared_ptr<Buffer> buffer) {
    return std::make_shared<Array>(std::move(layout), std::move(buffer), nullptr);
}

std::vector<ArrayPtr> Runtime::issue(std::shared_ptr<Kernel> kernel,
                                     std::span<const ArrayPtr> inputs,
                                     std::span<const ArrayPtr> inouts,
                                     std::span<const Layout> new_outputs, std::string issue_site) {
    // Back-pressure: limit how far the main thread runs ahead. Only the main
    // thread increments active_, so check-then-increment is race free.
    for (std::size_t n = active_.load(std::memory_order_acquire); n >= options_.max_active_tasks;
         n = active_.load(std::memory_order_acquire)) {
        active_.wait(n, std::memory_order_acquire);
    }
    active_.fetch_add(1, std::memory_order_acq_rel);

    auto task = std::make_shared<Task>(std::move(kernel), std::move(issue_site));

    // Producers to wait for: those of all read arrays (inputs and in/outs),
    // captured before the in/outs get the new task as their producer.
    std::vector<std::shared_ptr<Task>> producers;
    std::optional<TaskError> failed_input;
    auto note_producer = [&](Array& a) {
        if (!a.ready()) {
            producers.push_back(a.producer_);
        } else if (a.last_error_ && !failed_input) {
            failed_input = a.last_error_;
        }
    };

    // Copy-on-write decision for in/outs, before this task adds its own reads.
    std::vector<std::shared_ptr<Buffer>> inout_targets;
    inout_targets.reserve(inouts.size());
    for (const ArrayPtr& a : inouts) {
        if (a->buffer_->async_reads.load(std::memory_order_acquire) > 0) {
            inout_targets.push_back(std::make_shared<Buffer>(a->layout_.nbytes()));
        } else {
            inout_targets.push_back(a->buffer_);
        }
    }

    std::vector<Operand> in_ops;
    in_ops.reserve(inputs.size() + inouts.size());
    for (const ArrayPtr& a : inputs) {
        note_producer(*a);
        in_ops.push_back({a->buffer_, a->layout_});
    }
    for (const ArrayPtr& a : inouts) {
        note_producer(*a);
        in_ops.push_back({a->buffer_, a->layout_});
    }
    for (Operand& op : in_ops) {
        op.buffer->async_reads.fetch_add(1, std::memory_order_relaxed);
    }

    std::vector<Operand> out_ops;
    std::vector<ArrayPtr> results;
    out_ops.reserve(inouts.size() + new_outputs.size());
    results.reserve(new_outputs.size());
    for (std::size_t i = 0; i < inouts.size(); ++i) {
        Array& a = *inouts[i];
        if (inout_targets[i] != a.buffer_) {
            // Copy-on-write: the new buffer gets a fresh contiguous layout.
            a.layout_ = Layout::contiguous(a.layout_.dtype, a.layout_.shape);
            a.buffer_ = inout_targets[i];
        }
        a.producer_ = task;
        out_ops.push_back({a.buffer_, a.layout_});
    }
    for (const Layout& l : new_outputs) {
        Layout contiguous = Layout::contiguous(l.dtype, l.shape);
        auto buffer = std::make_shared<Buffer>(contiguous.nbytes());
        out_ops.push_back({buffer, contiguous});
        results.push_back(std::make_shared<Array>(contiguous, buffer, task));
    }

    task->set_operands(std::move(in_ops), std::move(out_ops));
    if (failed_input) {
        task->add_upstream_error(*failed_input);
    }
    // Phase 1: shapes are known at issue time.
    task->advance(TaskState::SizeCompleted);

    for (const std::shared_ptr<Task>& p : producers) {
        task->add_pending_input();
        if (!p->add_consumer(task)) {
            task->producer_already_done(*p); // cannot reach 0: issue guard held
        }
    }
    if (task->release_issue_guard()) {
        trigger_from_main(task);
    }
    return results;
}

void Runtime::trigger_from_main(const std::shared_ptr<Task>& task) {
    const bool offload = !options_.should_offload || options_.should_offload(*task);
    if (offload) {
        enqueue(task);
        return;
    }
    // Run inline; consumers it unblocks go to the workers so the main thread
    // can keep issuing.
    for (std::shared_ptr<Task>& next : run(task)) {
        enqueue(std::move(next));
    }
}

std::optional<TaskError> Runtime::wait(Array& a) {
    if (a.producer_) {
        a.producer_->wait_completed();
    }
    a.ready();
    return a.last_error_;
}

void Runtime::wait_all() {
    for (std::size_t n = active_.load(std::memory_order_acquire); n != 0;
         n = active_.load(std::memory_order_acquire)) {
        active_.wait(n, std::memory_order_acquire);
    }
}

std::vector<std::shared_ptr<Task>> Runtime::run(const std::shared_ptr<Task>& task) {
    std::vector<std::shared_ptr<Task>> runnable = task->execute();
    active_.fetch_sub(1, std::memory_order_acq_rel);
    active_.notify_all();
    return runnable;
}

void Runtime::enqueue(std::shared_ptr<Task> task) {
    {
        std::lock_guard lock(queue_mutex_);
        queue_.push_back(std::move(task));
    }
    queue_cv_.notify_one();
}

void Runtime::worker_loop() {
    std::shared_ptr<Task> next;
    for (;;) {
        if (!next) {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) {
                return; // stopping
            }
            next = std::move(queue_.front());
            queue_.pop_front();
        }
        // Trampoline: run() returns instead of calling consumers, so the stack
        // is unwound before the continuation starts.
        std::vector<std::shared_ptr<Task>> runnable = run(next);
        next.reset();
        if (!runnable.empty()) {
            next = std::move(runnable.front());
            for (std::size_t i = 1; i < runnable.size(); ++i) {
                enqueue(std::move(runnable[i]));
            }
        }
    }
}

} // namespace viproc
