// Tasks: one per array op, carrying the op's progress through its stages.
#pragma once

#include "countable_array.hpp"
#include "layout.hpp"
#include "offload.hpp"
#include "pool.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace viproc {

namespace detail {
// The thread has steps due (Task::step_due): a trivial thread-local, checked
// after every task.
VIPROC_TLS_INITIAL_EXEC extern thread_local bool t_any_due;
} // namespace detail

// Stages of a task, in order. The design (docs/design/pipelining.md, S1)
// requires each stage to be reached exactly once, in increasing order:
// SizeCompleted at issue (phase 1), DeviceSelected (the CPU) and Allocated in
// execute() right before the kernel, Completed after it. Only the direct
// path (Runtime::issue_direct) has no task and no stages.
enum class TaskState : std::uint8_t {
    Created,
    SizeCompleted,
    DeviceSelected,
    Allocated,
    Completed,
};

class Task;

// Tasks a task hands on: its consumers, the consumers it unblocks. Inline for
// the common one or two (no heap allocation per task), else on the heap.
using TaskList = InlineVector<std::shared_ptr<Task>, 2>;

// The stage signals a consumer receives from each producer it registered
// with (Task::add_consumer): exactly these three, each exactly once, in this
// order, whatever the producer's state at registration
// (docs/design/pipelining.md, S3). DeviceSelected is a state of its own (S1)
// but not signalled while the CPU is the only device.
enum class Signal : std::uint8_t {
    None = 0,
    SizeCompleted = 1,
    Allocated = 2,
    Completed = 3,
};

// The last signal a task in state `s` has given.
constexpr Signal signal_of(TaskState s) {
    switch (s) {
    case TaskState::Created:
        return Signal::None;
    case TaskState::SizeCompleted:
    case TaskState::DeviceSelected:
        return Signal::SizeCompleted;
    case TaskState::Allocated:
        return Signal::Allocated;
    case TaskState::Completed:
        return Signal::Completed;
    }
    return Signal::None;
}

// A task's registration with one of its producers: an entry in the producer's
// lock-free consumer list (Task::add_consumer). The node lives in the consumer
// task (inline, one per producer it waits on) and holds a reference to it
// while it is in the list; the producer takes that reference over when it
// completes.
struct ConsumerNode {
    ConsumerNode* next = nullptr;
    std::shared_ptr<Task> task;
    // The signals of this registration (Task::signal): high nibble the last
    // one requested, low nibble the last one delivered. A thread that raises
    // the request while the two are equal delivers (up to whatever is
    // requested by then); a raise while they differ leaves its signals to the
    // thread delivering. So each is delivered once, in order, by one thread
    // at a time, without a lock: the producer advancing and the main thread
    // registering may both find a signal due.
    std::atomic<std::uint8_t> signals{0};
};

// The layout of one array or view (docs/design/array-data-model.md, A1, A2): its
// size descriptor (SD: dtype, shape, byte strides, offset) and the task of the
// op that configures it while the SD is incomplete. One per array and per
// view, created by the main thread with the array, from a pool. The array and
// the operands of tasks reading or writing it refer to it (atomic count), so
// it is released on any thread; tasks reference the SD instead of copying it.
// The SD never changes once complete. Phase 1 knows every SD at issue: the
// main thread fills it and `task` stays null.
class MetadataPtr;
class ViewRule;

class Metadata {
  public:
    explicit Metadata(const Layout& sd) : sd_(sd) {}
    explicit Metadata(Layout&& sd) : sd_(std::move(sd)) {}
    Metadata(const Metadata&) = delete;
    Metadata& operator=(const Metadata&) = delete;

    const Layout& sd() const { return sd_; }
    // The op configuring the SD, while it is incomplete; null otherwise.
    const std::shared_ptr<Task>& task() const { return task_; }
    // Main thread: sets / drops the configuring task (Runtime::issue_deferred,
    // Runtime::wait_stage).
    void set_task(std::shared_ptr<Task> t) const { task_ = std::move(t); }
    // The configuring task only, in its size and layout steps (before it
    // signals SizeCompleted / Allocated): the SD it fills in.
    Layout& sd_for_task() const { return const_cast<Layout&>(sd_); }
    // The number of dimensions of an SD its task completes, known at issue
    // (Runtime::issue_deferred): set once by the main thread before the task
    // is launched, never changed; -1 for SDs complete at issue.
    int rank() const { return rank_; }
    void set_rank(int r) const { rank_ = r; }
    // O2: the rule a view's SD follows from its base's, while the base's
    // task still completes it (ViewRule); null otherwise. Set once by the
    // main thread before the view is used.
    const ViewRule* rule() const { return rule_.get(); }
    void set_rule(std::shared_ptr<const ViewRule> r) const { rule_ = std::move(r); }

  private:
    friend class MetadataPtr;
    friend MetadataPtr make_metadata(const Layout& sd);
    friend MetadataPtr make_metadata(Layout&& sd);
    // Destroys `s` and returns its memory to the pool (any thread).
    static void destroy(const Metadata* s);

    Layout sd_;
    // Strong, but no leak: the task's output operand refers back to this
    // Metadata only until the task completes (Task::execute clears all
    // operands). Main thread only, like AliasGroup::last_writer: it registers
    // on the task's stages and drops the task once it sees it completed
    // (docs/design/array-data-model.md, "Cycles"). Workers never read it.
    mutable std::shared_ptr<Task> task_;
    mutable int rank_ = -1;
    mutable std::shared_ptr<const ViewRule> rule_;
    mutable std::atomic<int> refs_{1}; // the creating reference
};

// Intrusive, atomically counted reference to a Metadata: no control block, and
// the first reference is the one make_metadata() returns.
class MetadataPtr {
  public:
    MetadataPtr() = default;
    MetadataPtr(std::nullptr_t) {}
    MetadataPtr(const MetadataPtr& o) : p_(o.p_) {
        if (p_ != nullptr) {
            p_->refs_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    MetadataPtr(MetadataPtr&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    MetadataPtr& operator=(MetadataPtr o) noexcept {
        std::swap(p_, o.p_);
        return *this;
    }
    ~MetadataPtr() { reset(); }

    void reset() {
        const Metadata* p = std::exchange(p_, nullptr);
        if (p != nullptr && p->refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            Metadata::destroy(p);
        }
    }
    const Metadata* get() const { return p_; }
    const Metadata& operator*() const { return *p_; }
    const Metadata* operator->() const { return p_; }
    explicit operator bool() const { return p_ != nullptr; }

  private:
    friend MetadataPtr make_metadata(const Layout& sd);
    friend MetadataPtr make_metadata(Layout&& sd);
    explicit MetadataPtr(const Metadata* adopted) : p_(adopted) {}
    const Metadata* p_ = nullptr;
};

// A Metadata from the per-thread pool (pool.hpp), with one reference.
MetadataPtr make_metadata(const Layout& sd);
MetadataPtr make_metadata(Layout&& sd);

// One argument of a kernel call: the data version (CountableArray) captured
// at issue time and the SD it is read or written with. Capturing the version
// (not the array or its group) keeps readers on the data they were issued
// against, even if the group switches to a new version later
// (copy-on-write). `meta` keeps the SD alive; ops that run directly on the
// main thread (Runtime::issue_direct) leave it null and point `sd` at the
// array's SD, which outlives the call.
struct Operand {
    std::shared_ptr<CountableArray> buffer;
    MetadataPtr meta;
    const Layout* sd = nullptr;
    // Input only: counted in buffer->async_reads. An op does not count reads
    // of the buffer it writes in place: every later writer waits for it as
    // the group's last writer anyway.
    bool counted_read = true;

    Operand() = default;
    Operand(std::shared_ptr<CountableArray> b, MetadataPtr s, bool counted = true)
        : buffer(std::move(b)), meta(std::move(s)), sd(&meta->sd()), counted_read(counted) {}
    Operand(std::shared_ptr<CountableArray> b, const Layout& l, bool counted)
        : buffer(std::move(b)), sd(&l), counted_read(counted) {}

    const Layout& layout() const { return *sd; }
    std::byte* first() const { return buffer->data() + sd->offset; }
};

// Operands of one task, inline up to a ternary op with one output.
using Operands = InlineVector<Operand, 4>;

struct KernelResult {
    int status = 0; // 0 = success
    std::string message;
    int fp_flags = 0; // FE_* flags raised while running (see <cfenv>)
};

// The computation of an op. Runs on a worker or the main thread, without the
// GIL. Kernels must not own Python objects: a kernel may be destroyed on any
// thread.
class Kernel {
  public:
    virtual ~Kernel() = default;
    virtual KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) = 0;
    // True if the kernel gives the same result when an output shares its
    // memory with an input of the same layout (element-wise loops, as NumPy's
    // `out=` and temporary elision). The task may then hand a dying input's
    // block to the output (Task::execute, design doc D11).
    virtual bool in_place_ok() const { return false; }
    // True if the kernel writes every element of its in/out operands. Only
    // then may copy-on-write give a consolidated in/out array a fresh buffer
    // without copying the old data into it (Runtime::issue, step 2); kernels
    // that write a subset (scatter, masked assignment) keep the others.
    virtual bool writes_all() const { return true; }
    // The NumPy function the op runs ("numpy.power", "numpy.add.reduce"),
    // reported with its errors. A string with static storage.
    virtual const char* name() const { return "array op"; }
    // The name NumPy's floating-point error reports give the op ("divide",
    // "reduce", "cast", "matmul": "overflow encountered in reduce"), or null
    // if NumPy does not check the FP flags for it (copies, indexing, sort,
    // loops flagged no_floatingpoint_errors). Error handling, E8.
    virtual const char* fp_name() const { return nullptr; }
    // The cost estimate of a kernel shared by many ops, or null (kernels made
    // per op learn nothing across ops). Main thread only.
    virtual DirectCost* direct_cost() { return nullptr; }

    // --- Ops that configure their outputs in the task (pipelining.md, S2;
    // Runtime::issue_deferred). Called by the task, never on the main thread.
    // The output sizes, once every input reached SizeCompleted: inputs give
    // dtype, shape and offset only (S5: not their strides); outputs come with
    // their dtype, the kernel sets their shape. A non-zero status fails the
    // op (its message as in run(): "ValueError: ...").
    virtual KernelResult size(std::span<const Layout* const> inputs, std::span<Layout> outputs) {
        (void)inputs;
        (void)outputs;
        return KernelResult{1, "RuntimeError: the op cannot size its outputs", 0};
    }
    // S4 (pipelining.md): the output sizes depend on the inputs' data
    // (np.zeros(n) with a pending n). Such a task sizes its outputs when it
    // runs, every input completed (size_from_data() instead of size()); its
    // three signals follow each other.
    virtual bool sizes_from_data() const { return false; }
    virtual KernelResult size_from_data(std::span<const Operand> inputs,
                                        std::span<Layout> outputs) {
        (void)inputs;
        (void)outputs;
        return KernelResult{1, "RuntimeError: the op cannot size its outputs", 0};
    }
    // The output strides (and offset), once every input reached Allocated
    // (their strides are known now); the shapes stay as size() set them.
    // Default: C order.
    virtual void layout(std::span<const Layout* const> inputs, std::span<Layout> outputs) {
        (void)inputs;
        for (Layout& o : outputs) {
            o = Layout::contiguous(o.dtype, std::move(o.shape));
        }
    }
};

// NumPy's floating-point error modes (np.errstate), captured at issue: three
// bits per category, NumPy's UFUNC_ERR_* values (0 ignore, 1 warn, 2 raise,
// 3 call, 4 print, 5 log), categories in NumPy's order of reporting:
// divide (bits 0-2), over (3-5), under (6-8), invalid (9-11). `ctx` names
// the bridge's call/log object for this mode (0 if none).
struct FpMode {
    std::uint16_t mode = 0;
    std::uint16_t ctx = 0;
};
inline constexpr int kFpRaise = 2;

// FP errors of an op to report in the main thread (modes other than raise).
struct FpEvent {
    int report = 0; // NumPy's FPE bits (1 divide, 2 over, 4 under, 8 invalid) to report
    int status = 0; // all FPE bits the op raised (passed to an errstate call)
    FpMode mode;
    const char* name = nullptr; // Kernel::fp_name()
    std::string issue_site;
    std::uint64_t id = 0;
};

// The error of a failed op (docs/design/error-handling.md).
struct TaskError {
    // "ValueError: ..." when the kernel knows the Python exception type
    // (bridges raise that type), else a plain message.
    std::string message;
    std::string issue_site; // where the failing op was issued (user code)
    std::string op;         // Kernel::name() of the failing op
    std::uint64_t id = 0;   // the failing op's issue number (Runtime: ops issued before it + 1)
};

class ErrorTable;

// A reference to a failed op's record in the runtime's ErrorTable; empty for
// ops that did not fail. Tasks and alias groups hold only this, never the error
// itself: the record exists once, in the runtime, and every thread reads it
// from there. Copies and destruction lock the table, so they cost something
// only for failed ops (an empty reference is a null check).
class ErrorRef {
  public:
    ErrorRef() = default;
    ErrorRef(const ErrorRef& o);
    ErrorRef(ErrorRef&& o) noexcept : table_(std::move(o.table_)), id_(std::exchange(o.id_, 0)) {}
    ErrorRef& operator=(const ErrorRef& o);
    ErrorRef& operator=(ErrorRef&& o) noexcept;
    ~ErrorRef() {
        if (table_) {
            reset();
        }
    }

    explicit operator bool() const { return table_ != nullptr; }
    // The failed op's issue number (0 if empty).
    std::uint64_t id() const { return id_; }
    ErrorTable* table() const { return table_.get(); }
    void reset();
    bool operator==(const ErrorRef& o) const { return id_ == o.id_ && table_ == o.table_; }

  private:
    friend class ErrorTable;
    ErrorRef(std::shared_ptr<ErrorTable> table, std::uint64_t id)
        : table_(std::move(table)), id_(id) {}

    std::shared_ptr<ErrorTable> table_;
    std::uint64_t id_ = 0;
};

// The one place where errors of failed ops live (asynchronous error
// reporting, docs/design/error-handling.md). Owned by the runtime; its records
// are reference counted by ErrorRefs (the failed task, its outputs, consumers
// that inherit it) and outlive the runtime only while an array still refers to
// one. The thread that runs a failing op records it before the op completes;
// the main thread checks pending() at every call into viproc (one atomic load)
// and takes the earliest unreported one. Only an op's own error is a record;
// consumers refer to the same record (same id). All methods lock the table
// (failed ops only), except pending().
class ErrorTable : public std::enable_shared_from_this<ErrorTable> {
  public:
    // Any thread: records the failure of an op, not yet reported.
    ErrorRef record(TaskError e);
    bool pending() const { return pending_.load(std::memory_order_acquire); }
    // Main thread: the error of the earliest issued failed op not reported
    // yet, if any; all records are reported afterwards. `more` receives how
    // many other unreported ones there were.
    std::optional<TaskError> take(std::size_t* more = nullptr);
    // Main thread: the error a sync point raises; reported from then on.
    TaskError raise_at_sync_point(const ErrorRef& r);
    // Records held (tests).
    std::size_t size() const;
    // Any thread: an op's FP errors to report (warn, call, print, log).
    void record_fp(FpEvent e);
    // Main thread: the FP events not reported yet, in issue order.
    std::vector<FpEvent> take_fp();
    // Called (from the recording thread, without locks) when the table goes
    // from nothing to report to something to report. The bridge uses it to
    // raise the error in its main thread right away. Set before ops are issued.
    void set_notify(void (*fn)(void*), void* arg) {
        notify_arg_ = arg;
        notify_.store(fn, std::memory_order_release);
    }

  private:
    friend class ErrorRef;
    void acquire(std::uint64_t id);
    void release(std::uint64_t id);

    struct Record {
        TaskError error;
        std::size_t refs;
        bool reported;
    };
    Record* find(std::uint64_t id);
    void update_pending();

    // Unreported records nothing refers to any more are kept for take() up
    // to this number; further ones are only counted.
    static constexpr std::size_t kMaxOrphans = 64;
    std::atomic<bool> pending_{false};
    std::atomic<void (*)(void*)> notify_{nullptr};
    void* notify_arg_ = nullptr;
    mutable std::mutex mutex_;
    std::vector<Record> records_;    // few: one per failed op still referenced
    std::vector<FpEvent> fp_events_; // bounded: kMaxFpEvents
    static constexpr std::size_t kMaxFpEvents = 1024;
    std::size_t orphans_ = 0;
    std::size_t dropped_ = 0;
};

// One dimension of a basic-indexing view: the base dimension kept, sliced
// (start/stop/step as PySlice_Unpack gives them, adjusted to the extent
// when the base's shape is known), indexed (dropped), or a new axis.
struct ViewStep {
    enum Kind : std::uint8_t { Keep, Slice, Index, NewAxis };
    Kind kind = Keep;
    std::int64_t start = 0, stop = 0, step = 1; // Slice; Index: start
};

// O2: a view (basic indexing, transpose) of an array whose SD a task still
// completes. Its SD follows from the base's by NumPy's rules
// (get_view_from_index, PyArray_Transpose); the first thread that needs it
// computes it, once per stage: the shape when the base is size completed,
// strides and offset when it is allocated (a consumer task's size and
// layout steps, or the main thread after waiting for the base,
// Runtime::wait_stage). An index out of bounds is the view's error, recorded
// once (NumPy's IndexError, raised at the call in NumPy).
class ViewRule {
  public:
    // `steps` for a basic key (one per view dimension or consumed base
    // dimension), else `perm` (a transpose: view axis i is base axis perm[i]).
    ViewRule(MetadataPtr base, std::vector<ViewStep> steps, std::vector<int> perm, const char* site,
             std::uint64_t id)
        : base_(std::move(base)), steps_(std::move(steps)), perm_(std::move(perm)), site_(site),
          id_(id) {}
    // Any thread, once the base reached the stage: fills `view`'s shape
    // (layout false) or its whole SD (layout true; the shape first).
    ErrorRef derive(const Metadata& view, bool layout, ErrorTable* errors) const;

  private:
    ErrorRef compute(const Metadata& view, bool layout, ErrorTable* errors) const;

    MetadataPtr base_;
    std::vector<ViewStep> steps_;
    std::vector<int> perm_;
    const char* site_;
    std::uint64_t id_;
    mutable std::once_flag once_[2];
    mutable ErrorRef error_; // written in the first stage's call_once only
};

// After a kernel ran with FP flags `fe_flags` (FE_*): NumPy's reporting of
// FP errors under `mode` (_check_ufunc_fperr): the categories set, in
// NumPy's order, are recorded for the main thread (warn, call, print, log)
// until one whose mode is raise: that one fails the op (the returned
// record, "FloatingPointError: divide by zero encountered in divide").
// Any thread. Empty if the op does not fail.
ErrorRef report_fp_errors(ErrorTable& errors, const Kernel& kernel, int fe_flags, FpMode mode,
                          const char* issue_site, std::uint64_t id);

class Task {
  public:
    // `issue_site` must stay valid for the task's lifetime (a string literal
    // or an interned string, see Runtime::issue()).
    Task(std::shared_ptr<Kernel> kernel, const char* issue_site);

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    TaskState state() const { return state_.load(std::memory_order_acquire); }
    bool completed() const { return state() == TaskState::Completed; }
    // Blocks until the task is completed.
    void wait_completed() const;
    // Blocks until the task reached `s` (or completed).
    void wait_state(TaskState s) const;

    // Configures its outputs itself (Runtime::issue_deferred): sizes on its
    // inputs' SizeCompleted signals, strides on their Allocated signals, on
    // the thread delivering the last one (or a pool job, never the main
    // thread). Set at issue, before launch; `errors` records a size error.
    void set_configures(ErrorTable* errors) {
        configures_ = true;
        errors_ = errors;
    }
    bool configures() const { return configures_; }
    // A size step failed (or an input's did): the op's error, valid once the
    // task reached SizeCompleted (acquire through state()).
    const ErrorRef& size_error() const { return size_error_; }
    // Runs the steps (size, then layout) whose inputs' signals all arrived
    // and that nobody ran yet; a step another thread runs is left to it.
    // Any thread but the main thread.
    void run_due_steps();
    // Runs the size and layout steps if nobody did, or waits for the thread
    // running them (every input completed: both are due).
    void finish_steps();
    // The pool's queue may hold a configuring task twice: as a step job and
    // to run. take_job() tells which entry this is (one bit per entry).
    enum class Job : std::uint8_t { Run = 1, Steps = 2 };
    // True if the bit was not set: queue an entry (else one is queued and
    // will do this job too: steps run whatever is due when it runs).
    bool add_job(Job j) {
        const auto b = static_cast<std::uint8_t>(j);
        return (jobs_.fetch_or(b, std::memory_order_acq_rel) & b) == 0;
    }
    Job take_job();

    const char* issue_site() const { return issue_site_; }
    // The op's issue number (set by the runtime when it launches the task).
    void set_id(std::uint64_t id) { id_ = id; }
    // The op's size: the most elements of any operand (a reduction's work
    // is its input, not its result). Decides where an unblocked task runs
    // (Runtime::worker_loop). Any thread once the task is runnable (its
    // inputs completed; a configuring task's outputs may still be sized
    // concurrently, so only its inputs count).
    std::int64_t work() const {
        std::int64_t n = 0;
        const std::size_t count = configures_ ? ninputs_ : operands_.size();
        for (std::size_t i = 0; i < count; ++i) {
            n = std::max(n, operands_[i].layout().size());
        }
        return n;
    }
    // Valid once completed (acquire through state()).
    const ErrorRef& error() const { return error_; }
    int fp_flags() const { return fp_flags_; }
    // NumPy's errstate at issue (set by the runtime when it launches the task).
    void set_fp_mode(FpMode m) { fp_mode_ = m; }

    // --- Main thread, while issuing -------------------------------------
    // The operands, filled in place by the issuing code: the inputs followed
    // by the outputs, the first `ninputs` being inputs.
    Operands& operands_for_issue() { return operands_; }
    void set_ninputs(std::size_t ninputs) { ninputs_ = ninputs; }
    // Times the kernel when the task runs and leaves the time per element of
    // `elements` in the kernel's DirectCost::task_sample (a re-test of an
    // expensive loop, OffloadPolicy::promote).
    void time_kernel(std::int64_t elements) { timed_elements_ = elements; }
    void advance(TaskState next);
    // Announces `n` pending producers before registering on them, and makes
    // room for their consumer nodes (inline up to kInlineNodes).
    void add_pending_inputs(std::size_t n);
    // Registers `consumer` for this task's stage signals, with the consumer's
    // node `i` (one per producer, i < the n of add_pending_inputs): it
    // receives SizeCompleted, Allocated and Completed exactly once each, in
    // this order. Signals of stages this task has passed already are
    // delivered here, by the calling thread; the others by the thread that
    // advances this task. Cannot make `consumer` runnable (its issue guard
    // is held). Lock-free; the main thread only.
    void add_consumer(const std::shared_ptr<Task>& consumer, std::size_t i);
    // Drops the issue guard (see pending_). Returns true if the task is
    // runnable now, i.e. all producers it waits on have completed. A
    // configuring task's size step may become due (take_due_steps()).
    bool release_issue_guard(const std::shared_ptr<Task>& self);
    // The configuring tasks whose steps became due on this thread since the
    // last call (signals delivered here): a worker runs them
    // (run_due_steps()), the main thread hands them to the pool.
    static std::vector<std::shared_ptr<Task>> take_due_steps();
    static bool due_steps_empty() { return !detail::t_any_due; }
    // Producers whose `stage` signal has not arrived yet, plus the issue
    // guard while it is held (tests; any thread, relaxed).
    int inputs_pending(Signal stage) const;
    // Marks an input as failed (its producer completed with an error before
    // this task was issued). The task then skips its kernel and fails too.
    void add_upstream_error(const ErrorRef& error) { set_upstream_error(error); }
    // Copy-on-write inside this task: before its kernel runs, `to` (one of
    // its deferred output buffers) gets the old data of `from`, whose read
    // the caller counted. Decided as late as possible, right before the
    // kernel: if nothing else reads `from` any more, `to` takes its block
    // over instead of copying (design doc D8).
    void add_copy_on_write(std::shared_ptr<CountableArray> from,
                           std::shared_ptr<CountableArray> to) {
        copies_.push_back({std::move(from), std::move(to)});
    }

    // Dispatch statistics this task's work is attributed to: the site whose
    // main-thread decision started it (inherited by consumers issued while it
    // was pending), and the op count at that decision. Main thread only.
    void set_blame(SiteStats* stats, std::uint64_t seq) {
        blame_ = stats;
        blame_seq_ = seq;
    }
    SiteStats* blame() const { return blame_; }
    std::uint64_t blame_seq() const { return blame_seq_; }
    // True the first time only (count one block per task).
    bool mark_blocked() { return !std::exchange(blocked_, true); }

    // --- Any thread ---------------------------------------------------------
    // Runs the kernel (or skips it if an input failed), completes the task and
    // returns the consumers that became runnable. Never runs consumers itself:
    // the calling thread schedules them as continuations. A kernel error is
    // recorded in `errors` (if given) before the task completes, so a sync
    // point that sees the failed task finds its record too.
    TaskList execute(ErrorTable* errors = nullptr);

    ~Task();

    // Consumer nodes inline in the task; more producers take a heap array.
    static constexpr std::size_t kInlineNodes = 3;

  private:
    // Gives every output without memory its block: a dying input's (same
    // layout, in_place_ok kernels) or one from the element pool.
    void allocate_outputs();
    // Runs the copies of add_copy_on_write (take the block or copy).
    void copy_on_write();

    // Called once per producer when it completes (and once for the issue
    // guard with nullptr). Returns true when the last one arrives.
    bool input_done(const Task* producer);
    // Raises the signals requested for node `n` of `consumer` to `upto` and
    // delivers the due ones unless another thread is delivering (see
    // ConsumerNode::signals). Returns true if a Completed signal delivered
    // here made `consumer` runnable.
    bool signal(ConsumerNode& n, const std::shared_ptr<Task>& consumer, Signal upto) const;
    // One signal of `producer` arriving (exactly once per producer and
    // stage) at `self` (this task). Returns true if it made this task
    // runnable.
    bool receive(Signal s, const Task& producer, const std::shared_ptr<Task>& self);
    // A step (1 size, 2 layout) became due: run it on this thread, after the
    // current signal delivery (a worklist per thread, no nesting), or hand
    // it to the runtime (main thread: step_due_).
    void step_due(const std::shared_ptr<Task>& self);

  public:
    // An input that cannot be sized (failed): this op is not sized either.
    void claim_size_error(const ErrorRef& e) {
        if (!size_upstream_claimed_.exchange(true, std::memory_order_relaxed)) {
            size_upstream_ = e;
        }
    }

  private:
    // The size and layout steps (S2), each once, in order (steps_).
    void size_step();
    void layout_step();
    void run_steps(int upto);
    // Delivers `s` to every registered consumer (SizeCompleted, Allocated;
    // Completed closes the list in execute()). The task's state is `s`'s.
    void signal_consumers(Signal s);
    // The first upstream error wins (any thread, before its pending_
    // decrement; execute() reads it after the last one).
    void set_upstream_error(const ErrorRef& error) {
        if (!upstream_claimed_.exchange(true, std::memory_order_relaxed)) {
            upstream_error_ = error;
        }
    }
    ConsumerNode& node(std::size_t i) {
        return i < kInlineNodes ? nodes_[i] : more_nodes_[i - kInlineNodes];
    }

    std::atomic<TaskState> state_{TaskState::Created};
    // Number of producers still to complete, plus one guard held by the main
    // thread while it registers the task, so it cannot start early.
    std::atomic<int> pending_{1};
    // The same for the SizeCompleted and Allocated signals. When the last one
    // arrives, all inputs' sizes (memory) are known. Phase 1 knows every
    // size at issue and allocates right before the kernel, so nothing reacts
    // yet; tasks with sizes from their inputs (pipelining.md, S2-S4) will.
    std::atomic<int> sized_pending_{1};
    std::atomic<int> allocated_pending_{1};
    // SizeCompleted signals add_consumer() delivered before publishing the
    // node (main thread, while issuing): dropped with the issue guard.
    int presized_ = 0;
    // Configuring tasks: inputs whose layout is still unknown, plus the issue
    // guard. Their layout step needs the inputs' strides, not their memory:
    // an input sized at issue (its producer does not configure) has them
    // already and counts as laid out at registration (prelaid_, dropped with
    // the guard); a configuring producer's Allocated signal brings them.
    std::atomic<int> layout_pending_{1};
    int prelaid_ = 0;
    // Configuring tasks (set_configures): their steps, combined as the
    // signals of a ConsumerNode (high nibble requested, low nibble done;
    // 1 size, 2 layout), the size error, an input's size error (first one,
    // written before the sized_pending_ decrement), and the jobs queued for
    // them (take_job).
    bool configures_ = false;
    ErrorTable* errors_ = nullptr;
    std::atomic<std::uint8_t> steps_{0};
    std::atomic<std::uint8_t> jobs_{0};
    std::atomic<bool> size_upstream_claimed_{false};
    ErrorRef size_upstream_;
    ErrorRef size_error_;

    std::shared_ptr<Kernel> kernel_;
    Operands operands_; // inputs, then outputs
    std::size_t ninputs_ = 0;
    std::int64_t timed_elements_ = 0; // > 0: time the kernel (time_kernel)
    const char* issue_site_;
    std::uint64_t id_ = 0;
    struct Copy {
        std::shared_ptr<CountableArray> from, to;
    };
    std::vector<Copy> copies_; // copy-on-write, empty for most tasks

    // The consumers registered for this task's completion: a lock-free stack
    // of ConsumerNodes (the main thread pushes with a CAS), closed for good
    // when the task completes (exchange with kClosed: a later push fails, so
    // no consumer is lost or notified twice). Bit kWaiter: someone waits (or
    // is about to) in wait_completed(); only then does completion notify.
    // std::atomic::notify_all is not free even without waiters (libstdc++
    // bumps a process-wide counter for 1-byte atomics). Waiter and completer
    // meet in this one atomic, so no lock and no seq_cst are needed.
    static constexpr std::uintptr_t kWaiter = 1;
    static constexpr std::uintptr_t kClosed = 2; // no node has this address
    static bool closed(std::uintptr_t h) { return (h & ~kWaiter) == kClosed; }
    mutable std::atomic<std::uintptr_t> consumers_{0};
    ConsumerNode nodes_[kInlineNodes];
    std::unique_ptr<ConsumerNode[]> more_nodes_;
    std::atomic<bool> upstream_claimed_{false};
    ErrorRef upstream_error_;

    ErrorRef error_;
    int fp_flags_ = 0;
    FpMode fp_mode_;

    // Main thread only (see set_blame).
    SiteStats* blame_ = nullptr;
    std::uint64_t blame_seq_ = 0;
    bool blocked_ = false;
};

} // namespace viproc
