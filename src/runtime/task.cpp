#include "task.hpp"

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

void Task::advance(TaskState next) {
    [[maybe_unused]] const TaskState prev = state_.load(std::memory_order_relaxed);
    assert(next > prev && "task states only move forward");
    state_.store(next, std::memory_order_release);
}

void Task::add_pending_inputs(std::size_t n) {
    if (n > kInlineNodes) {
        more_nodes_ = std::make_unique<ConsumerNode[]>(n - kInlineNodes);
    }
    pending_.fetch_add(static_cast<int>(n), std::memory_order_relaxed);
}

bool Task::add_consumer(const std::shared_ptr<Task>& consumer, std::size_t i) {
    // Acquire: on kClosed, the caller reads this task's error_ (written before
    // the release exchange in execute()).
    std::uintptr_t h = consumers_.load(std::memory_order_acquire);
    if (closed(h)) {
        return false;
    }
    ConsumerNode& n = consumer->node(i);
    n.task = consumer; // the list's reference, taken over by execute()
    do {
        if (closed(h)) {
            n.task.reset();
            return false;
        }
        n.next = reinterpret_cast<ConsumerNode*>(h & ~kWaiter);
    } while (
        !consumers_.compare_exchange_weak(h, reinterpret_cast<std::uintptr_t>(&n) | (h & kWaiter),
                                          std::memory_order_release, std::memory_order_acquire));
    return true;
}

bool Task::input_done(const Task* producer) {
    if (producer != nullptr && producer->error_) {
        set_upstream_error(producer->error_);
    }
    return pending_.fetch_sub(1, std::memory_order_acq_rel) == 1;
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
    if (upstream_error_) {
        error_ = std::move(upstream_error_);
    }
    // Output memory as late as possible (design doc D11): here, right before
    // the kernel, on the thread that runs it. Also on errors: readers of an
    // output never see a buffer without memory.
    copy_on_write();
    allocate_outputs();
    advance(TaskState::Allocated);
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
                c->task_sample.store(
                    static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(per)) << 32 |
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
        if (c->input_done(this)) {
            runnable.push_back(std::move(c));
        }
        list = next;
    }
    return runnable;
}

} // namespace viproc
