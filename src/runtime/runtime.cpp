#include "runtime.hpp"

#include <algorithm>
#include <chrono>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#elif defined(_MSC_VER) && defined(_M_ARM64)
#include <intrin.h>
#endif
#include <cstring>

namespace viproc {

namespace {

// Tells the CPU we are spin-waiting (cheaper than a yield syscall).
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
#if defined(_MSC_VER)
    __yield();
#else
    asm volatile("yield");
#endif
#else
    std::this_thread::yield();
#endif
}

} // namespace

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

namespace {

// Copies a whole buffer (copy-on-write with views).
class CopyKernel : public Kernel {
  public:
    KernelResult run(std::span<const Operand> in, std::span<const Operand> out) override {
        std::memcpy(out[0].first(), in[0].first(), in[0].layout.nbytes());
        return {};
    }
};

bool same_view(const Layout& x, const Layout& y) {
    return x.dtype == y.dtype && x.shape == y.shape && x.strides == y.strides &&
           x.offset == y.offset;
}

Layout bytes_of(const Buffer& b) {
    return Layout::contiguous(DType::Bool, {static_cast<std::int64_t>(b.nbytes())});
}

} // namespace

ArrayPtr Runtime::wrap(Layout layout, std::shared_ptr<Buffer> buffer) {
    return std::make_shared<Array>(std::move(layout), std::make_shared<Storage>(std::move(buffer)));
}

ArrayPtr Runtime::view(const ArrayPtr& base, Layout layout) {
    return std::make_shared<Array>(std::move(layout), base->storage_);
}

void Runtime::acquire_slot() {
    // Back-pressure: limit how far the main thread runs ahead. Only the main
    // thread increments active_, so check-then-increment is race free.
    for (std::size_t n = active_.load(std::memory_order_acquire); n >= options_.max_active_tasks;
         n = active_.load(std::memory_order_acquire)) {
        active_.wait(n, std::memory_order_acquire);
    }
    active_.fetch_add(1, std::memory_order_acq_rel);
}

void Runtime::launch(const std::shared_ptr<Task>& task,
                     const std::vector<std::shared_ptr<Task>>& producers) {
    for (const std::shared_ptr<Task>& p : producers) {
        task->add_pending_input();
        if (!p->add_consumer(task)) {
            task->producer_already_done(*p); // cannot reach 0: issue guard held
        }
    }
    if (task->release_issue_guard()) {
        trigger_from_main(task);
    }
}

void Runtime::issue_storage_copy(Storage& s) {
    std::shared_ptr<Buffer> old = s.buffer_;
    auto fresh = std::make_shared<Buffer>(old->nbytes());
    std::vector<std::shared_ptr<Task>> producers;
    std::optional<TaskError> failed;
    if (!s.ready()) {
        producers.push_back(s.producer_);
    } else {
        failed = s.last_error_;
    }

    acquire_slot();
    auto task = std::make_shared<Task>(std::make_shared<CopyKernel>(), "copy-on-write");
    old->async_reads.fetch_add(1, std::memory_order_relaxed);
    task->set_operands({{old, bytes_of(*old)}, {fresh, bytes_of(*fresh)}}, 1);
    if (failed) {
        task->add_upstream_error(*failed);
    }
    task->advance(TaskState::SizeCompleted);
    s.buffer_ = fresh;
    s.producer_ = task;
    launch(task, producers);
}

std::vector<ArrayPtr> Runtime::issue(std::shared_ptr<Kernel> kernel,
                                     std::span<const ArrayPtr> inputs,
                                     std::span<const ArrayPtr> inouts,
                                     std::span<const Layout> new_outputs, const char* issue_site) {
    // 1. Snapshot everything this op reads (inputs, then in/outs): buffer,
    //    layout and pending producer, before any copy-on-write switch. Reads
    //    always see the data as it was before this op.
    std::vector<Operand> in_ops;
    std::vector<std::shared_ptr<Task>> producers;
    std::optional<TaskError> failed_input;
    // One vector for all operands: inputs, then outputs (see Task).
    in_ops.reserve(inputs.size() + 2 * inouts.size() + new_outputs.size());
    auto snapshot = [&](Array& a) {
        Storage& s = *a.storage_;
        if (!s.ready()) {
            producers.push_back(s.producer_);
        } else if (s.last_error_ && !failed_input) {
            failed_input = s.last_error_;
        }
        in_ops.push_back({s.buffer_, a.layout_});
    };
    for (const ArrayPtr& a : inputs) {
        snapshot(*a);
    }
    for (const ArrayPtr& a : inouts) {
        snapshot(*a);
    }

    // 2. Copy-on-write decisions, before this op counts its own reads. All
    //    layout changes happen here, at issue time: what later tasks learn
    //    (layout, size) never changes afterwards.
    std::vector<Storage*> switched;
    for (const ArrayPtr& ap : inouts) {
        Array& a = *ap;
        Storage& s = *a.storage_;
        if (std::find(switched.begin(), switched.end(), &s) != switched.end()) {
            continue; // same storage written twice by this op: switched already
        }
        const bool pending_readers = s.buffer_->async_reads.load(std::memory_order_acquire) > 0;
        // An input reading the same storage through a different view would see
        // partly updated data if the op wrote in place; NumPy semantics are
        // "all inputs read before any output is written".
        bool overlapping_input = false;
        for (const ArrayPtr& in : inputs) {
            overlapping_input = overlapping_input ||
                                (in->storage_ == a.storage_ && !same_view(in->layout_, a.layout_));
        }
        if (!pending_readers && !overlapping_input) {
            continue; // write in place
        }
        switched.push_back(&s);
        const bool only_view = a.storage_.use_count() == 1;
        const bool covers_buffer = a.layout_.nbytes() == s.buffer_->nbytes();
        if (only_view && covers_buffer) {
            // Consolidated: no copy task. The op reads the old buffer (step 1)
            // and writes a new, contiguous one.
            a.layout_ = Layout::contiguous(a.layout_.dtype, a.layout_.shape);
            s.buffer_ = std::make_shared<Buffer>(a.layout_.nbytes());
        } else {
            // Views share the storage (or the op writes only part of it):
            // copy the whole buffer first, views keep their layouts.
            issue_storage_copy(s);
        }
    }

    // 3. The task itself.
    acquire_slot();
    auto task = std::make_shared<Task>(std::move(kernel), issue_site);
    for (Operand& op : in_ops) {
        op.buffer->async_reads.fetch_add(1, std::memory_order_relaxed);
    }

    const std::size_t ninputs = in_ops.size();
    std::vector<ArrayPtr> results;
    results.reserve(new_outputs.size());
    for (const ArrayPtr& ap : inouts) {
        Storage& s = *ap->storage_;
        // A storage copy issued in step 2 must finish before this op writes.
        if (s.producer_ && s.producer_ != task &&
            std::find(producers.begin(), producers.end(), s.producer_) == producers.end()) {
            producers.push_back(s.producer_);
        }
        s.producer_ = task;
        in_ops.push_back({s.buffer_, ap->layout_});
    }
    for (const Layout& l : new_outputs) {
        Layout contiguous = Layout::contiguous(l.dtype, l.shape);
        auto buffer = std::make_shared<Buffer>(contiguous.nbytes());
        in_ops.push_back({buffer, contiguous});
        results.push_back(std::make_shared<Array>(std::move(contiguous),
                                                  std::make_shared<Storage>(buffer, task)));
    }

    task->set_operands(std::move(in_ops), ninputs);
    if (failed_input) {
        task->add_upstream_error(*failed_input);
    }
    // Phase 1: shapes are known at issue time.
    task->advance(TaskState::SizeCompleted);
    launch(task, producers);
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
    Storage& s = *a.storage_;
    if (s.producer_) {
        s.producer_->wait_completed();
    }
    s.ready();
    return s.last_error_;
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
        queued_.fetch_add(1, std::memory_order_seq_cst);
    }
    // A spinning worker will pick the task up. The seq_cst order between this
    // load and the worker's decrement of spinning_ before it sleeps (followed
    // by its locked re-check of the queue) rules out a lost wake-up.
    if (spinning_.load(std::memory_order_seq_cst) == 0 &&
        sleeping_.load(std::memory_order_seq_cst) > 0) {
        queue_cv_.notify_one();
    }
}

std::shared_ptr<Task> Runtime::try_pop() {
    std::lock_guard lock(queue_mutex_);
    if (queue_.empty()) {
        return nullptr;
    }
    std::shared_ptr<Task> t = std::move(queue_.front());
    queue_.pop_front();
    queued_.fetch_sub(1, std::memory_order_relaxed);
    return t;
}

void Runtime::worker_loop() {
    using clock = std::chrono::steady_clock;
    constexpr auto kSpin = std::chrono::microseconds(50);
    std::shared_ptr<Task> next;
    for (;;) {
        if (!next) {
            next = try_pop();
        }
        if (!next) {
            // At most one worker spins (the others sleep), so spinning never
            // takes a core away from the main thread or a working worker.
            int expected = 0;
            if (spinning_.compare_exchange_strong(expected, 1, std::memory_order_seq_cst)) {
                const auto until = clock::now() + kSpin;
                while (!next && clock::now() < until) {
                    if (queued_.load(std::memory_order_relaxed) > 0) {
                        next = try_pop();
                    } else {
                        cpu_relax();
                    }
                }
                spinning_.store(0, std::memory_order_seq_cst);
                // Work found while others sleep: let the next worker spin.
                if (next && queued_.load(std::memory_order_seq_cst) > 0 &&
                    sleeping_.load(std::memory_order_seq_cst) > 0) {
                    queue_cv_.notify_one();
                }
            }
        }
        if (!next) {
            std::unique_lock lock(queue_mutex_);
            sleeping_.fetch_add(1, std::memory_order_seq_cst);
            queue_cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            sleeping_.fetch_sub(1, std::memory_order_seq_cst);
            if (queue_.empty()) {
                return; // stopping
            }
            next = std::move(queue_.front());
            queue_.pop_front();
            queued_.fetch_sub(1, std::memory_order_relaxed);
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
