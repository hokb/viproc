#include "task.hpp"

#include <thread>

#include "pool.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <cfenv>
#include <chrono>
#include <cstring>
#include <exception>

namespace viproc {

namespace {
using MetadataPool = FixedPool<sizeof(Metadata), alignof(Metadata)>;
}

MetadataPtr make_metadata(const Layout& sd) {
    return MetadataPtr(::new (MetadataPool::allocate()) Metadata(sd));
}

MetadataPtr make_metadata(Layout&& sd) {
    return MetadataPtr(::new (MetadataPool::allocate()) Metadata(std::move(sd)));
}

void Metadata::destroy(const Metadata* s) {
    s->~Metadata();
    MetadataPool::deallocate(const_cast<Metadata*>(s));
}

Task::Task(std::shared_ptr<Kernel> kernel, const char* issue_site)
    : kernel_(std::move(kernel)), issue_site_(issue_site) {}

Task::~Task() {
    // A task destroyed without completing (never launched) may still list
    // consumers; drop their references.
    std::uintptr_t h = consumers_.load(std::memory_order_acquire);
    if (!closed(h)) {
        for (auto* n = reinterpret_cast<ConsumerNode*>(h & ~kWaiter); n != nullptr;) {
            ConsumerNode* next = n->next;
            n->task.reset();
            n = next;
        }
    }
}

void Task::wait_completed() const {
    if (completed()) {
        return;
    }
    // Either execute()'s exchange sees the bit and notifies, or it closed the
    // list before and this one sees kClosed (and, acquired, the state).
    if (closed(consumers_.fetch_or(kWaiter, std::memory_order_acq_rel))) {
        return;
    }
    TaskState s = state();
    while (s != TaskState::Completed) {
        state_.wait(s, std::memory_order_acquire);
        s = state();
    }
}

void Task::wait_state(TaskState s) const {
    if (state() >= s) {
        return;
    }
    // seq_cst with advance(): it stores the state, then loads the word with
    // the waiter bit; this sets the bit, then loads the state.
    if (closed(consumers_.fetch_or(kWaiter, std::memory_order_seq_cst))) {
        return; // completed
    }
    for (TaskState c = state_.load(std::memory_order_seq_cst); c < s; c = state()) {
        state_.wait(c, std::memory_order_acquire);
    }
}

namespace {

// Steps (size, layout) of configuring tasks that became due on this thread,
// run (worker) or handed to the pool (main thread) by the runtime after the
// signal delivery that made them due: no nesting along chains of sizes.
thread_local std::vector<std::shared_ptr<Task>> t_due_steps;

} // namespace

VIPROC_TLS_INITIAL_EXEC thread_local bool detail::t_any_due = false;
using detail::t_any_due;

std::vector<std::shared_ptr<Task>> Task::take_due_steps() {
    std::vector<std::shared_ptr<Task>> v;
    if (t_any_due) {
        v.swap(t_due_steps);
        t_any_due = false;
    }
    return v;
}

void Task::step_due(const std::shared_ptr<Task>& self) {
    if (kernel_->sizes_from_data()) {
        return; // its steps run in execute(), when the data is there (S4)
    }
    t_due_steps.push_back(self);
    t_any_due = true;
}

void Task::advance(TaskState next) {
    [[maybe_unused]] const TaskState prev = state_.load(std::memory_order_relaxed);
    assert(static_cast<int>(next) == static_cast<int>(prev) + 1 &&
           "task states move to the next one, none skipped (pipelining.md, S1)");
    assert(next != TaskState::Completed && "execute() completes");
    const Signal s = signal_of(next);
    if (s == signal_of(prev)) { // DeviceSelected: no signal of its own
        state_.store(next, std::memory_order_release);
        return;
    }
    // seq_cst, with signal_consumers()'s load of the list: a registration
    // either is in the list this thread walks or sees the new state in
    // add_consumer() (it pushes, then loads the state, both seq_cst).
    state_.store(next, std::memory_order_seq_cst);
    signal_consumers(s);
}

void Task::signal_consumers(Signal s) {
    const std::uintptr_t h = consumers_.load(std::memory_order_seq_cst);
    if (closed(h)) {
        return;
    }
    if ((h & kWaiter) != 0) {
        state_.notify_all(); // the main thread waits for this stage (wait_state)
    }
    // Nodes stay in the list (and alive: each holds its consumer) until
    // execute() closes it; pushes after this load see the new state.
    for (auto* n = reinterpret_cast<ConsumerNode*>(h & ~kWaiter); n != nullptr; n = n->next) {
        if (s == Signal::Allocated) {
            // The common case: the size signal delivered, nobody delivering.
            // Only this thread completes this task, so while it delivers,
            // nobody requests more than Allocated: no raise to wait for.
            std::uint8_t idle = 1 << 4 | 1;
            if (n->signals.compare_exchange_strong(idle, 2 << 4 | 1, std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
                n->task->receive(Signal::Allocated, *this, n->task);
                n->signals.store(2 << 4 | 2, std::memory_order_release);
                continue;
            }
            if ((idle >> 4) >= 2) {
                continue; // the registering thread delivers (or delivered) it
            }
        }
        [[maybe_unused]] const bool runnable = signal(*n, n->task, s);
        assert(!runnable && "only Completed makes a consumer runnable");
    }
}

void Task::add_pending_inputs(std::size_t n) {
    if (n > kInlineNodes) {
        more_nodes_ = std::make_unique<ConsumerNode[]>(n - kInlineNodes);
    }
    const int k = static_cast<int>(n);
    pending_.fetch_add(k, std::memory_order_relaxed);
    sized_pending_.fetch_add(k, std::memory_order_relaxed);
    allocated_pending_.fetch_add(k, std::memory_order_relaxed);
    layout_pending_.fetch_add(k, std::memory_order_relaxed);
}

void Task::add_consumer(const std::shared_ptr<Task>& consumer, std::size_t i) {
    ConsumerNode& n = consumer->node(i);
    // A size known already (phase 1: always) is signalled before the node is
    // published, so no other thread can deliver it: one count for the issue
    // guard to drop (presized_), no atomic per registration.
    if (signal_of(state_.load(std::memory_order_acquire)) >= Signal::SizeCompleted) {
        n.signals.store(1 << 4 | 1, std::memory_order_relaxed);
        ++consumer->presized_;
        if (size_error_) {
            consumer->claim_size_error(size_error_);
        }
    }
    if (!configures_) {
        ++consumer->prelaid_; // its layout is known (and unused unless the consumer configures)
    }
    std::uintptr_t h = consumers_.load(std::memory_order_acquire);
    if (!closed(h)) {
        n.task = consumer; // the list's reference, taken over by execute()
        do {
            if (closed(h)) {
                n.task.reset();
                break;
            }
            n.next = reinterpret_cast<ConsumerNode*>(h & ~kWaiter);
        } while (!consumers_.compare_exchange_weak(
            h, reinterpret_cast<std::uintptr_t>(&n) | (h & kWaiter), std::memory_order_seq_cst,
            std::memory_order_acquire));
    }
    // The signals this task gave before it could see the node (all three if
    // the list was closed: execute() stores Completed before it closes it).
    // seq_cst: see advance(). Acquire: Completed makes error_ visible.
    const Signal upto = signal_of(state_.load(std::memory_order_seq_cst));
    if (static_cast<int>(upto) > (n.signals.load(std::memory_order_relaxed) >> 4)) {
        [[maybe_unused]] const bool runnable = signal(n, consumer, upto);
        assert(!runnable && "the issue guard is held");
    }
}

bool Task::signal(ConsumerNode& n, const std::shared_ptr<Task>& consumer, Signal upto) const {
    const int want = static_cast<int>(upto);
    std::uint8_t w = n.signals.load(std::memory_order_acquire);
    for (;;) {
        if (want <= (w >> 4)) {
            return false; // requested already: delivered or being delivered
        }
        // acq_rel: the delivering thread sees what the requester saw (this
        // task's error_ for Completed).
        if (n.signals.compare_exchange_weak(w, static_cast<std::uint8_t>(want << 4 | (w & 15)),
                                            std::memory_order_acq_rel, std::memory_order_acquire)) {
            break;
        }
    }
    const int done0 = w & 15;
    if (done0 < (w >> 4)) {
        return false; // another thread is delivering; it delivers ours too
    }
    bool runnable = false;
    int done = done0, target = want;
    for (;;) {
        while (done < target) {
            ++done;
            runnable = consumer->receive(static_cast<Signal>(done), *this, consumer) || runnable;
        }
        std::uint8_t expected = static_cast<std::uint8_t>(target << 4 | done0);
        if (n.signals.compare_exchange_strong(
                expected, static_cast<std::uint8_t>(target << 4 | target),
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            return runnable;
        }
        target = expected >> 4; // raised meanwhile: deliver those too
    }
}

bool Task::receive(Signal s, const Task& producer, const std::shared_ptr<Task>& self) {
    switch (s) {
    case Signal::SizeCompleted:
        if (producer.size_error_) {
            claim_size_error(producer.size_error_);
        }
        if (sized_pending_.fetch_sub(1, std::memory_order_acq_rel) == 1 && configures_) {
            step_due(self);
        }
        return false;
    case Signal::Allocated:
        allocated_pending_.fetch_sub(1, std::memory_order_acq_rel);
        if (configures_ && producer.configures_ &&
            layout_pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            step_due(self);
        }
        return false;
    case Signal::Completed:
        return input_done(&producer);
    case Signal::None:
        break;
    }
    return false;
}

bool Task::release_issue_guard(const std::shared_ptr<Task>& self) {
    const int presized = std::exchange(presized_, 0);
    if (sized_pending_.fetch_sub(1 + presized, std::memory_order_acq_rel) == 1 + presized &&
        configures_) {
        step_due(self); // the runtime hands it to the pool
    }
    allocated_pending_.fetch_sub(1, std::memory_order_acq_rel);
    if (configures_) {
        const int prelaid = std::exchange(prelaid_, 0);
        if (layout_pending_.fetch_sub(1 + prelaid, std::memory_order_acq_rel) == 1 + prelaid) {
            step_due(self); // inputs laid out (the size may still be due: then nothing)
        }
    }
    return input_done(nullptr);
}

Task::Job Task::take_job() {
    std::uint8_t j = jobs_.load(std::memory_order_acquire);
    for (;;) {
        // A run entry wins: execute() runs the steps first anyway.
        const std::uint8_t take = (j & static_cast<std::uint8_t>(Job::Run)) != 0
                                      ? static_cast<std::uint8_t>(Job::Run)
                                      : static_cast<std::uint8_t>(Job::Steps);
        if ((j & take) == 0) {
            return Job::Run; // no bit: a run that was not queued as a job
        }
        if (jobs_.compare_exchange_weak(j, static_cast<std::uint8_t>(j & ~take),
                                        std::memory_order_acq_rel, std::memory_order_acquire)) {
            return static_cast<Job>(take);
        }
    }
}

void Task::run_due_steps() {
    const int due = sized_pending_.load(std::memory_order_acquire) != 0    ? 0
                    : layout_pending_.load(std::memory_order_acquire) != 0 ? 1
                                                                           : 2;
    if (due > 0) {
        run_steps(due);
    }
}

void Task::finish_steps() {
    run_steps(2);
    while ((steps_.load(std::memory_order_acquire) & 15) != 2) {
        std::this_thread::yield();
    }
}

void Task::run_steps(int upto) {
    // As signal(): whoever raises the request while no step runs runs the
    // steps, up to whatever is requested by the time it is done.
    std::uint8_t w = steps_.load(std::memory_order_acquire);
    for (;;) {
        if (upto <= (w >> 4)) {
            return;
        }
        if (steps_.compare_exchange_weak(w, static_cast<std::uint8_t>(upto << 4 | (w & 15)),
                                         std::memory_order_acq_rel, std::memory_order_acquire)) {
            break;
        }
    }
    const int done0 = w & 15;
    if (done0 < (w >> 4)) {
        return;
    }
    int done = done0, target = upto;
    for (;;) {
        while (done < target) {
            ++done;
            if (done == 1) {
                size_step();
            } else {
                layout_step();
            }
        }
        std::uint8_t expected = static_cast<std::uint8_t>(target << 4 | done0);
        if (steps_.compare_exchange_strong(expected,
                                           static_cast<std::uint8_t>(target << 4 | target),
                                           std::memory_order_acq_rel, std::memory_order_acquire)) {
            return;
        }
        target = expected >> 4;
    }
}

void Task::size_step() {
    InlineVector<const Layout*, 4> in;
    for (std::size_t i = 0; i < ninputs_; ++i) {
        in.push_back(&operands_[i].layout());
    }
    InlineVector<Layout, 2> out;
    for (std::size_t o = ninputs_; o < operands_.size(); ++o) {
        Layout l;
        l.dtype = operands_[o].layout().dtype;
        out.push_back(std::move(l));
    }
    const bool from_data = kernel_->sizes_from_data(); // in execute(): error_ is this thread's
    if (size_upstream_claimed_.load(std::memory_order_acquire)) {
        size_error_ = size_upstream_; // an input could not be sized: neither can this op
    } else if (from_data && error_) {
        size_error_ = error_; // an input's data failed: no size from it
    } else {
        KernelResult r;
        try {
            r = from_data
                    ? kernel_->size_from_data(std::span<const Operand>(operands_.data(), ninputs_),
                                              std::span<Layout>(out.data(), out.size()))
                    : kernel_->size(std::span<const Layout* const>(in.data(), in.size()),
                                    std::span<Layout>(out.data(), out.size()));
        } catch (const std::exception& e) {
            r = KernelResult{1, e.what(), 0};
        }
        if (r.status != 0 && errors_ != nullptr) {
            size_error_ =
                errors_->record(TaskError{std::move(r.message), issue_site_, kernel_->name(), id_});
        }
    }
    for (std::size_t k = 0; k < out.size(); ++k) {
        Layout& sd = operands_[ninputs_ + k].meta->sd_for_task();
        sd.shape = size_error_ ? Shape{} : std::move(out.data()[k].shape);
        sd.offset = 0;
    }
    advance(TaskState::SizeCompleted);
}

void Task::layout_step() {
    InlineVector<const Layout*, 4> in;
    for (std::size_t i = 0; i < ninputs_; ++i) {
        in.push_back(&operands_[i].layout());
    }
    InlineVector<Layout, 2> out;
    for (std::size_t o = ninputs_; o < operands_.size(); ++o) {
        out.emplace_back(operands_[o].layout());
    }
    if (!size_error_) {
        kernel_->layout(std::span<const Layout* const>(in.data(), in.size()),
                        std::span<Layout>(out.data(), out.size()));
    } else {
        for (Layout& l : std::span<Layout>(out.data(), out.size())) {
            l = Layout::contiguous(l.dtype, Shape{});
        }
    }
    for (std::size_t k = 0; k < out.size(); ++k) {
        Operand& op = operands_[ninputs_ + k];
        op.buffer->set_size(out.data()[k].nbytes());
        // Strides and offset only: the shape is published already (readers
        // may read it since SizeCompleted, S5).
        Layout& sd = op.meta->sd_for_task();
        assert(out.data()[k].shape == sd.shape && "Kernel::layout keeps the shape");
        sd.strides = std::move(out.data()[k].strides);
        sd.offset = out.data()[k].offset;
    }
    advance(TaskState::DeviceSelected);
    advance(TaskState::Allocated);
}

int Task::inputs_pending(Signal stage) const {
    switch (stage) {
    case Signal::SizeCompleted:
        return sized_pending_.load(std::memory_order_relaxed);
    case Signal::Allocated:
        return allocated_pending_.load(std::memory_order_relaxed);
    case Signal::Completed:
        return pending_.load(std::memory_order_relaxed);
    case Signal::None:
        break;
    }
    return 0;
}

bool Task::input_done(const Task* producer) {
    if (producer != nullptr && producer->error_) {
        set_upstream_error(producer->error_);
    }
    if (pending_.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        return false;
    }
    if (configures_) {
        (void)add_job(Job::Run); // see take_job(); runs once
    }
    return true;
}

namespace {

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define VIPROC_TSAN 1
#endif
#endif
#if defined(__SANITIZE_THREAD__)
#define VIPROC_TSAN 1
#endif

void acquire_fence() {
#if !defined(VIPROC_TSAN) // TSan does not model fences (GCC: -Werror=tsan)
    std::atomic_thread_fence(std::memory_order_acquire);
#endif
}

bool same_layout(const Layout& a, const Layout& b) {
    return a.dtype == b.dtype && a.offset == b.offset && a.shape == b.shape &&
           a.strides == b.strides;
}

} // namespace

void Task::copy_on_write() {
    for (Copy& c : copies_) {
        // The only reference to the old buffer is ours and so is the only
        // read (same check as in allocate_outputs): every other reader is
        // done, nobody can read it again, so no copy is needed.
        bool taken =
            c.from.use_count() == 1 && c.from->async_reads.load(std::memory_order_acquire) == 1;
        if (taken) {
            acquire_fence();
            taken = c.to->take_block(*c.from);
        }
        if (!taken) {
            std::memcpy(c.to->ensure_allocated(), c.from->data(), c.from->nbytes());
        }
        c.from->release_read();
    }
    copies_.clear();
}

void Task::allocate_outputs() {
    const bool reuse = !error_ && kernel_->in_place_ok();
    for (std::size_t o = ninputs_; o < operands_.size(); ++o) {
        Operand& out = operands_[o];
        if (out.buffer->allocated()) {
            continue;
        }
        if (reuse) {
            for (std::size_t i = 0; i < ninputs_; ++i) {
                Operand& in = operands_[i];
                // The only reference to the input's buffer is this operand: no
                // array, export or other task can read it any more, and none
                // can get it again. The acquire load of async_reads (only our
                // own read left) pairs with the other tasks' release_read(),
                // the fence with the release of the other references (e.g. an
                // export the main thread read): their reads happen before our
                // writes.
                if (!in.counted_read || in.buffer.use_count() != 1 ||
                    !same_layout(in.layout(), out.layout()) ||
                    in.buffer->async_reads.load(std::memory_order_acquire) != 1) {
                    continue;
                }
                acquire_fence();
                if (!out.buffer->take_block(*in.buffer)) {
                    continue;
                }
                // The input now reads the output's memory (same layout, same
                // data): its own buffer is gone, and so is its read.
                in.buffer->release_read();
                in.counted_read = false;
                in.buffer = out.buffer;
                break;
            }
        }
        out.buffer->ensure_allocated();
    }
}

ErrorRef::ErrorRef(const ErrorRef& o) : table_(o.table_), id_(o.id_) {
    if (table_) {
        table_->acquire(id_);
    }
}

ErrorRef& ErrorRef::operator=(const ErrorRef& o) {
    if (this != &o) {
        ErrorRef copy(o);
        *this = std::move(copy);
    }
    return *this;
}

ErrorRef& ErrorRef::operator=(ErrorRef&& o) noexcept {
    if (this != &o) {
        reset();
        table_ = std::move(o.table_);
        id_ = std::exchange(o.id_, 0);
    }
    return *this;
}

void ErrorRef::reset() {
    if (table_) {
        table_->release(id_);
        table_.reset();
        id_ = 0;
    }
}

ErrorTable::Record* ErrorTable::find(std::uint64_t id) {
    for (Record& r : records_) {
        if (r.error.id == id) {
            return &r;
        }
    }
    return nullptr;
}

void ErrorTable::update_pending() {
    bool any = dropped_ > 0 || !fp_events_.empty();
    for (const Record& r : records_) {
        any = any || !r.reported;
    }
    pending_.store(any, std::memory_order_release);
}

ErrorRef ErrorTable::record(TaskError e) {
    const std::uint64_t id = e.id;
    bool was_pending;
    {
        std::lock_guard lock(mutex_);
        records_.push_back({std::move(e), 1, false});
        was_pending = pending_.exchange(true, std::memory_order_acq_rel);
    }
    if (!was_pending) {
        if (auto fn = notify_.load(std::memory_order_acquire)) {
            fn(notify_arg_);
        }
    }
    return ErrorRef(shared_from_this(), id);
}

void ErrorTable::acquire(std::uint64_t id) {
    std::lock_guard lock(mutex_);
    Record* r = find(id);
    if (r->refs++ == 0 && !r->reported) {
        --orphans_;
    }
}

void ErrorTable::release(std::uint64_t id) {
    std::lock_guard lock(mutex_);
    Record* r = find(id);
    if (--r->refs > 0) {
        return;
    }
    if (!r->reported && orphans_ < kMaxOrphans) {
        ++orphans_; // kept until take() reports it
        return;
    }
    if (!r->reported) {
        ++dropped_; // counted only
    }
    *r = std::move(records_.back());
    records_.pop_back();
}

std::optional<TaskError> ErrorTable::take(std::size_t* more) {
    std::lock_guard lock(mutex_);
    std::optional<TaskError> first;
    std::size_t unreported = 0;
    for (const Record& r : records_) {
        if (!r.reported) {
            ++unreported;
            if (!first || r.error.id < first->id) {
                first = r.error;
            }
        }
    }
    if (more != nullptr) {
        *more = first ? unreported - 1 + dropped_ : 0;
    }
    for (Record& r : records_) {
        r.reported = true;
    }
    std::erase_if(records_, [](const Record& r) { return r.refs == 0; });
    orphans_ = 0;
    dropped_ = 0;
    pending_.store(!fp_events_.empty(), std::memory_order_release);
    return first;
}

void ErrorTable::record_fp(FpEvent e) {
    bool was_pending;
    {
        std::lock_guard lock(mutex_);
        if (fp_events_.size() >= kMaxFpEvents) {
            return; // the main thread is far behind: NumPy's dedup would hide them anyway
        }
        fp_events_.push_back(std::move(e));
        was_pending = pending_.exchange(true, std::memory_order_acq_rel);
    }
    if (!was_pending) {
        if (auto fn = notify_.load(std::memory_order_acquire)) {
            fn(notify_arg_);
        }
    }
}

std::vector<FpEvent> ErrorTable::take_fp() {
    std::lock_guard lock(mutex_);
    std::vector<FpEvent> events;
    events.swap(fp_events_);
    std::stable_sort(events.begin(), events.end(),
                     [](const FpEvent& a, const FpEvent& b) { return a.id < b.id; });
    update_pending();
    return events;
}

ErrorRef report_fp_errors(ErrorTable& errors, const Kernel& kernel, int fe_flags, FpMode mode,
                          const char* issue_site, std::uint64_t id) {
    const char* name = kernel.fp_name();
    if (mode.mode == 0 || name == nullptr) {
        return {};
    }
    const int status =
        ((fe_flags & FE_DIVBYZERO) != 0 ? 1 : 0) | ((fe_flags & FE_OVERFLOW) != 0 ? 2 : 0) |
        ((fe_flags & FE_UNDERFLOW) != 0 ? 4 : 0) | ((fe_flags & FE_INVALID) != 0 ? 8 : 0);
    static const char* const kinds[4] = {"divide by zero", "overflow", "underflow",
                                         "invalid value"};
    int report = 0;
    for (int c = 0; c < 4; ++c) {
        const int m = (mode.mode >> (3 * c)) & 7;
        if ((status & (1 << c)) == 0 || m == 0) {
            continue;
        }
        if (m == kFpRaise) {
            if (report != 0) {
                errors.record_fp(FpEvent{report, status, mode, name, issue_site, id});
            }
            return errors.record(TaskError{std::string("FloatingPointError: ") + kinds[c] +
                                               " encountered in " + name,
                                           issue_site, kernel.name(), id});
        }
        report |= 1 << c;
    }
    if (report != 0) {
        errors.record_fp(FpEvent{report, status, mode, name, issue_site, id});
    }
    return {};
}

TaskError ErrorTable::raise_at_sync_point(const ErrorRef& ref) {
    std::lock_guard lock(mutex_);
    Record* r = find(ref.id());
    r->reported = true;
    update_pending();
    return r->error;
}

std::size_t ErrorTable::size() const {
    std::lock_guard lock(mutex_);
    return records_.size();
}

TaskList Task::execute(ErrorTable* errors) {
    // No lock: every write of upstream_error_ happens before the release
    // decrement of pending_ by its writer, and this thread observed the final
    // decrement (or got the task from the thread that did).
    // Every input signalled its size and memory before its completion
    // (ConsumerNode::signals delivers in order). Equal to pending_, not 0:
    // tests run tasks without releasing the issue guard.
    assert(sized_pending_.load(std::memory_order_relaxed) ==
               pending_.load(std::memory_order_relaxed) &&
           allocated_pending_.load(std::memory_order_relaxed) ==
               pending_.load(std::memory_order_relaxed));
    if (upstream_error_) {
        error_ = std::move(upstream_error_);
    }
    // Output memory as late as possible (design doc D11): here, right before
    // the kernel, on the thread that runs it. Also on errors: readers of an
    // output never see a buffer without memory.
    // Device selection: always the CPU in phase 1, a stage of its own all the
    // same (pipelining.md, S1: none skipped).
    if (configures_) {
        // Its size and layout steps are due (every input completed); run
        // them here if nobody did, or wait for the thread running them.
        finish_steps(); // usually done: Runtime::run calls it first
        if (!error_ && size_error_) {
            error_ = size_error_;
        }
        copy_on_write();
        allocate_outputs();
    } else {
        assert(state_.load(std::memory_order_relaxed) == TaskState::SizeCompleted);
        state_.store(TaskState::DeviceSelected, std::memory_order_release);
        copy_on_write();
        allocate_outputs();
        // As advance(Allocated), without its checks (seq_cst: see advance()).
        state_.store(TaskState::Allocated, std::memory_order_seq_cst);
        signal_consumers(Signal::Allocated);
    }
    if (!error_) {
        const std::span<const Operand> inputs(operands_.data(), ninputs_);
        const std::span<Operand> outputs(operands_.data() + ninputs_, operands_.size() - ninputs_);
        std::feclearexcept(FE_ALL_EXCEPT);
        const auto t0 = timed_elements_ > 0 ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
        KernelResult r;
        try {
            r = kernel_->run(inputs, outputs);
        } catch (const std::exception& e) {
            r = KernelResult{1, e.what(), 0};
        } catch (...) {
            r = KernelResult{1, "unknown exception in kernel", 0};
        }
        if (timed_elements_ > 0 && r.status == 0) {
            const std::chrono::duration<double, std::nano> ns =
                std::chrono::steady_clock::now() - t0;
            if (DirectCost* c = kernel_->direct_cost()) {
                const float per =
                    static_cast<float>(ns.count() / static_cast<double>(timed_elements_));
                c->task_sample.store(static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(per))
                                             << 32 |
                                         static_cast<std::uint64_t>(timed_elements_),
                                     std::memory_order_relaxed);
            }
        }
        fp_flags_ = r.fp_flags | std::fetestexcept(FE_ALL_EXCEPT);
        if (r.status != 0) {
            if (errors != nullptr) {
                error_ = errors->record(
                    TaskError{std::move(r.message), issue_site_, kernel_->name(), id_});
            }
        } else if (fp_flags_ != 0 && fp_mode_.mode != 0 && errors != nullptr) {
            error_ = report_fp_errors(*errors, *kernel_, fp_flags_, fp_mode_, issue_site_, id_);
        }
    }

    // Done reading: release the inputs (and the async_reads they hold).
    for (std::size_t i = 0; i < ninputs_; ++i) {
        Operand& in = operands_[i];
        if (in.counted_read) {
            in.buffer->release_read();
        }
    }
    operands_.clear();
    kernel_.reset();

    assert(state_.load(std::memory_order_relaxed) == TaskState::Allocated);
    state_.store(TaskState::Completed, std::memory_order_release);
    // Close the list: registrations from now on fail and account for this
    // task themselves. Release: they see error_ and the state; acquire: we see
    // the nodes pushed.
    const std::uintptr_t h = consumers_.exchange(kClosed, std::memory_order_acq_rel);
    if ((h & kWaiter) != 0) {
        state_.notify_all();
    }

    // The stack holds the consumers newest first; reverse it, so they are
    // notified in registration order (the first one becomes the
    // continuation, as before).
    ConsumerNode* list = nullptr;
    for (auto* n = reinterpret_cast<ConsumerNode*>(h & ~kWaiter); n != nullptr;) {
        ConsumerNode* next = n->next;
        n->next = list;
        list = n;
        n = next;
    }
    TaskList runnable;
    while (list != nullptr) {
        // Read the link and take the reference before the decrement: after
        // it, another producer may run the consumer (whose node this is).
        ConsumerNode* next = list->next;
        std::shared_ptr<Task> c = std::move(list->task);
        // The common case, all but Completed delivered and nobody delivering:
        // mark it delivered first (no signal follows it, so no other thread
        // can deliver anything after), then deliver.
        std::uint8_t idle = 2 << 4 | 2;
        const bool ready =
            list->signals.compare_exchange_strong(idle, 3 << 4 | 3, std::memory_order_acq_rel,
                                                  std::memory_order_acquire)
                ? c->input_done(this)
                : signal(*list, c, Signal::Completed);
        if (ready) {
            runnable.push_back(std::move(c));
        }
        list = next;
    }
    return runnable;
}

} // namespace viproc
