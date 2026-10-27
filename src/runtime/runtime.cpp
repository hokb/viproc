#include "runtime.hpp"

#include "cpu.hpp"
#include "pool.hpp"

#include <algorithm>
#include <cfenv>
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

// How long the main thread spins at a sync point before it sleeps. Sleeping
// costs the completing thread a futex wake (a syscall) and the main thread a
// wake-up: 10-20 us together, several times a medium-sized op (issue #45).
// Only sync points spin, where the main thread waits for one result.
constexpr auto kSyncSpin = std::chrono::microseconds(50);

// Spins until done() holds or kSyncSpin has passed; returns done().
template <class Done> bool spin_until(Done done) {
    using clock = std::chrono::steady_clock;
    const auto until = clock::now() + kSyncSpin;
    do {
        for (int i = 0; i < 16; ++i) {
            if (done()) {
                return true;
            }
            cpu_relax();
        }
    } while (clock::now() < until);
    return done();
}

} // namespace

Runtime::Runtime(RuntimeOptions options) : options_(std::move(options)), policy_(options_.offload) {
    std::size_t n = options_.workers;
    if (n == 0) {
        n = physical_core_count();
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
    const char* name() const override { return "copy-on-write"; }
    KernelResult run(std::span<const Operand> in, std::span<const Operand> out) override {
        std::memcpy(out[0].first(), in[0].first(), in[0].layout().nbytes());
        return {};
    }
};

bool same_view(const Layout& x, const Layout& y) {
    return x.dtype == y.dtype && x.shape == y.shape && x.strides == y.strides &&
           x.offset == y.offset;
}

Layout bytes_of(const CountableArray& b) {
    return Layout::contiguous(DType::Bool, {static_cast<std::int64_t>(b.nbytes())});
}

} // namespace

namespace {

using BlockPool = FixedPool<sizeof(ArrayBlock), alignof(ArrayBlock)>;

ArrayBlock* new_block(int alive) {
    // Default-, not value-initialized: `ArrayBlock()` would zero the whole
    // block (inline shapes and strides included) before constructing it.
    auto* b = ::new (BlockPool::allocate()) ArrayBlock;
    b->alive = alive;
    return b;
}

void free_block(ArrayBlock* b) {
    b->~ArrayBlock();
    BlockPool::deallocate(b);
}

} // namespace

void AliasGroup::last_ref_dropped(AliasGroup* g) {
    ArrayBlock* b = g->block_;
    b->group.reset();
    if (--b->alive == 0) {
        free_block(b);
    }
}

void Array::last_ref_dropped(Array* a) {
    ArrayBlock* b = a->block_;
    b->array.reset(); // may drop the group of the same block first
    if (--b->alive == 0) {
        free_block(b);
    }
}

namespace {

// A base array with a new group, both in one block.
ArrayPtr new_array(MetadataPtr meta, std::shared_ptr<CountableArray> buffer,
                   std::shared_ptr<Task> last_writer) {
    ArrayBlock* block = new_block(2); // owned by the references from here on
    AliasGroup& g = block->group.emplace(std::move(buffer), std::move(last_writer), block);
    Array& a = block->array.emplace(std::move(meta), Ref<AliasGroup>(&g), block);
    return ArrayPtr(&a);
}

} // namespace

ArrayPtr Runtime::wrap(Layout layout, std::shared_ptr<CountableArray> buffer) {
    return new_array(make_metadata(std::move(layout)), std::move(buffer), nullptr);
}

ArrayPtr Runtime::view(const ArrayPtr& base, Layout&& layout) {
    ArrayBlock* block = new_block(1);
    Array& a = block->array.emplace(make_metadata(std::move(layout)), base->group_, block);
    return ArrayPtr(&a);
}

void Runtime::wait_active_below(std::size_t limit) {
    if (active_.load(std::memory_order_acquire) < limit) {
        return;
    }
    // seq_cst pairs with run(): it decrements active_, then reads the waiter
    // count. Either it sees this waiter and notifies, or the load below sees
    // its decrement.
    // Completions notify only once active_ is below `limit` (wake_below_):
    // waking the main thread on every completion would cost a futex wake
    // and a context switch per task.
    wake_below_.store(limit, std::memory_order_seq_cst);
    active_waiters_.fetch_add(1, std::memory_order_seq_cst);
    for (std::size_t n = active_.load(std::memory_order_seq_cst); n >= limit;
         n = active_.load(std::memory_order_seq_cst)) {
        active_.wait(n, std::memory_order_acquire);
    }
    active_waiters_.fetch_sub(1, std::memory_order_relaxed);
}

void Runtime::acquire_slot() {
    // Back-pressure: limit how far the main thread runs ahead. Only the main
    // thread increments active_, so check-then-increment is race free. At
    // the limit it waits until a quarter of the active tasks completed
    // (hysteresis): resuming after every single completion ping-pongs
    // between the threads, one wake-up per op.
    const std::size_t limit = options_.max_active_tasks;
    if (active_.load(std::memory_order_acquire) >= limit) {
        wait_active_below(limit - limit / 4);
    }
    active_.fetch_add(1, std::memory_order_acq_rel);
}

void Runtime::launch(const std::shared_ptr<Task>& task, const TaskList& producers,
                     std::int64_t elements) {
    ++seq_;
    task->set_id(seq_);
    task->set_fp_mode(fp_mode_);
    // While pending, the task's work is attributed to the decision that
    // started the work it waits for.
    for (const std::shared_ptr<Task>& p : producers) {
        if (p->blame() != nullptr) {
            task->set_blame(p->blame(), p->blame_seq());
            break;
        }
    }
    task->add_pending_inputs(producers.size());
    for (std::size_t i = 0; i < producers.size(); ++i) {
        if (!producers[i]->add_consumer(task, i)) {
            task->producer_already_done(*producers[i]); // cannot reach 0: issue guard held
        }
    }
    if (task->release_issue_guard()) {
        trigger_from_main(task, elements);
    }
}

void Runtime::issue_group_copy(AliasGroup& g) {
    std::shared_ptr<CountableArray> old = g.current_;
    auto fresh = make_pooled<CountableArray>(old->nbytes(),
                                             CountableArray::deferred); // the copy task allocates
    TaskList producers;
    ErrorRef failed;
    if (!g.ready()) {
        producers.emplace_back(g.last_writer_);
    } else {
        failed = g.last_error_;
    }

    // Counted, but not held back: the op that needs the copy holds a slot
    // already, and waiting here while holding it could deadlock.
    active_.fetch_add(1, std::memory_order_acq_rel);
    auto task = make_pooled<Task>(std::make_shared<CopyKernel>(), "copy-on-write");
    old->async_reads.fetch_add(1, std::memory_order_relaxed);
    MetadataPtr meta = make_metadata(bytes_of(*old)); // same bytes for both versions
    task->operands_for_issue().push_back({old, meta});
    task->operands_for_issue().push_back({fresh, std::move(meta)});
    task->set_ninputs(1);
    if (failed) {
        task->add_upstream_error(failed);
    }
    task->advance(TaskState::SizeCompleted);
    g.current_ = fresh;
    g.last_writer_ = task;
    launch(task, producers, static_cast<std::int64_t>(old->nbytes()));
}

std::vector<ArrayPtr> Runtime::issue(const std::shared_ptr<Kernel>& kernel,
                                     std::span<const ArrayPtr> inputs,
                                     std::span<const ArrayPtr> inouts,
                                     std::span<const Layout> new_outputs, const char* issue_site) {
    std::vector<ArrayPtr> results(new_outputs.size());
    issue(kernel, inputs, inouts, new_outputs, issue_site, results);
    return results;
}

void Runtime::issue(const std::shared_ptr<Kernel>& kernel, std::span<const ArrayPtr> inputs,
                    std::span<const ArrayPtr> inouts, std::span<const Layout> new_outputs,
                    const char* issue_site, std::span<ArrayPtr> results, std::int64_t work_elements,
                    std::span<const std::shared_ptr<CountableArray>> output_buffers) {
    const std::int64_t elements = work_elements >= 0     ? work_elements
                                  : !new_outputs.empty() ? new_outputs[0].size()
                                  : !inouts.empty()      ? inouts[0]->layout().size()
                                                         : 0;
    if (issue_direct(kernel, inputs, inouts, new_outputs, issue_site, results, elements,
                     output_buffers)) {
        return;
    }
    // The task first (back-pressure: wait for a slot before creating it), so
    // the operands are built in place.
    acquire_slot();
    auto task = make_pooled<Task>(kernel, issue_site);
    Operands& in_ops = task->operands_for_issue(); // inputs, then outputs

    // 1. Snapshot everything this op reads (inputs, then in/outs): buffer,
    //    layout and pending producer, before any copy-on-write switch. Reads
    //    always see the data as it was before this op.
    TaskList producers;
    ErrorRef failed_input;
    // An error on an input or in/out passes to this op: data computed from a
    // failed op, or written into its result, is failed too (also when the op
    // overwrites every element: a failed program does not continue).
    auto snapshot = [&](Array& a) {
        AliasGroup& g = *a.group_;
        if (!g.ready()) {
            producers.emplace_back(g.last_writer_);
        } else if (g.last_error_ && !failed_input) {
            failed_input = g.last_error_;
        }
        in_ops.push_back({g.current_, a.meta_});
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
    std::vector<AliasGroup*> switched;
    for (const ArrayPtr& ap : inouts) {
        Array& a = *ap;
        AliasGroup& g = *a.group_;
        if (std::find(switched.begin(), switched.end(), &g) != switched.end()) {
            continue; // same group written twice by this op: switched already
        }
        const bool pending_readers = g.current_->has_readers();
        // An input reading the same group through a different view would see
        // partly updated data if the op wrote in place; NumPy semantics are
        // "all inputs read before any output is written".
        bool overlapping_input = false;
        for (const ArrayPtr& in : inputs) {
            overlapping_input = overlapping_input ||
                                (in->group_ == a.group_ && !same_view(in->layout(), a.layout()));
        }
        if (!pending_readers && !overlapping_input) {
            continue; // write in place
        }
        switched.push_back(&g);
        const bool only_view = a.group_.use_count() == 1;
        const bool covers_buffer = a.layout().nbytes() == g.current_->nbytes();
        if (only_view && covers_buffer && kernel->writes_all()) {
            // Consolidated: no copy task. The op reads the old buffer (step 1)
            // and writes a new one with the same layout (as NumPy, the array
            // keeps its strides; later reductions see NumPy's element order).
            g.current_ =
                make_pooled<CountableArray>(g.current_->nbytes(), CountableArray::deferred);
        } else {
            // Views share the group (or the op writes only part of it, or
            // only some of its elements): the whole old data goes into the new
            // buffer first, views keep their layouts. Not a task of its own:
            // this op does it right before its kernel, when its inputs (the
            // readers that feed it, as in x[i] = f(x[i-1])) are done; a buffer
            // nobody reads any more is then taken over, not copied. Its own
            // read of the in/outs on `g` (step 1) moves to the new buffer.
            std::shared_ptr<CountableArray> old = g.current_;
            g.current_ = make_pooled<CountableArray>(old->nbytes(), CountableArray::deferred);
            old->async_reads.fetch_add(1, std::memory_order_relaxed);
            task->add_copy_on_write(old, g.current_);
            for (std::size_t j = 0; j < inouts.size(); ++j) {
                if (inouts[j]->group_ == a.group_) {
                    in_ops[inputs.size() + j].buffer = g.current_;
                }
            }
        }
    }

    // 3. Outputs. Count the reads; reads of a buffer this op writes in place
    //    are not counted (see Operand::counted_read).
    for (Operand& op : in_ops) {
        for (const ArrayPtr& ap : inouts) {
            op.counted_read = op.counted_read && op.buffer != ap->group_->current_;
        }
        if (op.counted_read) {
            op.buffer->async_reads.fetch_add(1, std::memory_order_relaxed);
        }
    }

    const std::size_t ninputs = in_ops.size();
    for (const ArrayPtr& ap : inouts) {
        AliasGroup& g = *ap->group_;
        // A group copy issued in step 2 must finish before this op writes.
        if (g.last_writer_ && g.last_writer_ != task &&
            std::find(producers.begin(), producers.end(), g.last_writer_) == producers.end()) {
            producers.emplace_back(g.last_writer_);
        }
        g.last_writer_ = task;
        in_ops.push_back({g.current_, ap->meta_});
    }
    for (std::size_t i = 0; i < new_outputs.size(); ++i) {
        const Layout& l = new_outputs[i];
        if (i < output_buffers.size() && output_buffers[i]) {
            MetadataPtr st = make_metadata(l);
            in_ops.push_back({output_buffers[i], st});
            results[i] = new_array(std::move(st), output_buffers[i], task);
            continue;
        }
        // Dense layouts (C, F, any axis order) are kept: ufunc results follow
        // NumPy's order (Layout::keep_order).
        Layout contiguous = l.dense() ? l : Layout::contiguous(l.dtype, l.shape);
        // The task takes the memory when it runs (D11).
        auto buffer = make_pooled<CountableArray>(contiguous.nbytes(), CountableArray::deferred);
        MetadataPtr st = make_metadata(std::move(contiguous));
        in_ops.push_back({buffer, st});
        results[i] = new_array(std::move(st), std::move(buffer), task);
    }

    task->set_ninputs(ninputs);
    if (failed_input) {
        task->add_upstream_error(failed_input);
    }
    // Phase 1: shapes are known at issue time. The output memory is taken by
    // the task itself, right before its kernel runs (memory-management.md,
    // D11): from a dying input or from the pool, while it is hot in cache.
    task->advance(TaskState::SizeCompleted);
    launch(task, producers, elements);
}

bool Runtime::issue_direct(const std::shared_ptr<Kernel>& kernel, std::span<const ArrayPtr> inputs,
                           std::span<const ArrayPtr> inouts, std::span<const Layout> new_outputs,
                           const char* issue_site, std::span<ArrayPtr> results,
                           std::int64_t elements,
                           std::span<const std::shared_ptr<CountableArray>> output_buffers) {
    if (options_.should_offload || elements >= policy_.options().sync_below) {
        return false;
    }
    for (const ArrayPtr& a : inputs) {
        AliasGroup& g = *a->group_;
        if (!g.ready() || g.last_error_) {
            return false;
        }
    }
    for (const ArrayPtr& a : inouts) {
        AliasGroup& g = *a->group_;
        if (!g.ready() || g.last_error_ || g.current_->has_readers()) {
            return false; // copy-on-write or upstream errors: the task path
        }
        for (const ArrayPtr& in : inputs) {
            if (in->group_ == a->group_ && !same_view(in->layout(), a->layout())) {
                return false;
            }
        }
    }
    Operands ops;
    for (const ArrayPtr& a : inputs) {
        ops.push_back({a->group_->current_, a->layout(), false});
    }
    for (const ArrayPtr& a : inouts) {
        ops.push_back({a->group_->current_, a->layout(), false});
    }
    const std::size_t ninputs = ops.size();
    for (const ArrayPtr& a : inouts) {
        ops.push_back({a->group_->current_, a->layout(), false});
    }
    for (std::size_t i = 0; i < new_outputs.size(); ++i) {
        const Layout& l = new_outputs[i];
        if (i < output_buffers.size() && output_buffers[i]) {
            MetadataPtr st = make_metadata(l);
            ops.push_back({output_buffers[i], st, false});
            results[i] = new_array(std::move(st), output_buffers[i], nullptr);
            continue;
        }
        Layout contiguous = l.dense() ? l : Layout::contiguous(l.dtype, l.shape);
        auto buffer = make_pooled<CountableArray>(contiguous.nbytes());
        MetadataPtr st = make_metadata(std::move(contiguous));
        ops.push_back({buffer, st, false});
        results[i] = new_array(std::move(st), std::move(buffer), nullptr);
    }
    ++seq_;
    ++direct_;
    std::feclearexcept(FE_ALL_EXCEPT);
    KernelResult r;
    try {
        r = kernel->run(std::span<const Operand>(ops.data(), ninputs),
                        std::span<const Operand>(ops.data() + ninputs, ops.size() - ninputs));
    } catch (const std::exception& e) {
        r = KernelResult{1, e.what(), 0};
    } catch (...) {
        r = KernelResult{1, "unknown exception in kernel", 0};
    }
    ErrorRef e;
    if (r.status != 0) {
        e = errors_->record(TaskError{std::move(r.message), issue_site, kernel->name(), seq_});
    } else if (fp_mode_.mode != 0) {
        const int fe = r.fp_flags | std::fetestexcept(FE_ALL_EXCEPT);
        if (fe != 0) {
            e = report_fp_errors(*errors_, *kernel, fe, fp_mode_, issue_site, seq_);
        }
    }
    if (e) {
        for (const ArrayPtr& a : inouts) {
            a->group_->last_error_ = e;
        }
        for (std::size_t i = 0; i < new_outputs.size(); ++i) {
            results[i]->group_->last_error_ = e;
        }
    }
    return true;
}

void Runtime::trigger_from_main(const std::shared_ptr<Task>& task, std::int64_t elements) {
    bool offload;
    if (options_.should_offload) {
        offload = options_.should_offload(*task);
    } else {
        const OffloadPolicy::Decision d = policy_.decide(task->issue_site(), elements);
        offload = d.offload;
        task->set_blame(d.stats, seq_);
    }
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

ErrorRef Runtime::wait(Array& a) {
    AliasGroup& g = *a.group_;
    if (g.last_writer_) {
        Task& p = *g.last_writer_;
        if (!p.completed()) {
            ++syncs_; // the main thread blocks
        }
        // Any access right after issuing counts, finished or not: async had
        // nothing to overlap with in between.
        if (p.mark_blocked()) {
            policy_.on_blocked(p.blame(), seq_ - p.blame_seq());
        }
        // Spin first: a waiter that registers costs the completing thread a
        // notify and itself a wake-up.
        if (!spin_until([&] { return p.completed(); })) {
            p.wait_completed();
        }
    }
    g.ready();
    return g.last_error_;
}

ErrorRef Runtime::prepare_host_write(Array& a) {
    ErrorRef e = wait(a);
    AliasGroup& g = *a.group_;
    if (g.current_->host_views > 0) {
        issue_group_copy(g);
        e = wait(a);
    }
    g.current_->wait_task_readers();
    return e;
}

void Runtime::wait_all() {
    // A sync point: spin first, as wait(). Back-pressure does not spin: it
    // waits for many tasks while the workers use every core.
    if (!spin_until([&] { return active_.load(std::memory_order_acquire) == 0; })) {
        wait_active_below(1);
    }
}

TaskList Runtime::run(const std::shared_ptr<Task>& task) {
    TaskList runnable = task->execute(errors_.get());
    const std::size_t left = active_.fetch_sub(1, std::memory_order_seq_cst) - 1;
    if (active_waiters_.load(std::memory_order_seq_cst) > 0 &&
        left < wake_below_.load(std::memory_order_seq_cst)) {
        active_.notify_all();
    }
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
        TaskList runnable = run(next);
        next.reset();
        if (!runnable.empty()) {
            next = std::move(runnable[0]);
            for (std::size_t i = 1; i < runnable.size(); ++i) {
                enqueue(std::move(runnable[i]));
            }
        }
    }
}

} // namespace viproc
