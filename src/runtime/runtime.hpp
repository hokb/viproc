// The runtime: issues array ops as tasks and runs them on worker threads.
#pragma once

#include "countable_array.hpp"
#include "layout.hpp"
#include "offload.hpp"
#include "ref.hpp"
#include "task.hpp"
#include "task_queue.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace viproc {

// Look-ahead limit: maximum number of issued, not yet completed tasks.
inline constexpr std::size_t kDefaultMaxActiveTasks = 1000;

struct ArrayBlock;

// The node a base array and all of its views share (docs/design/array-data-model.md,
// A1). Main-thread only.
//
// The group, not the individual array, carries the current data version
// (`current`, a CountableArray) and the last task writing it (`last_writer`):
// a write through one view makes the base and all other views pending, and
// copy-on-write switches all of them to the new version together, without
// viproc knowing the views.
//
// State: `pending` while `last_writer` is set and not completed, `ready`
// otherwise. Only the main thread sets `last_writer` (ready -> pending); the
// writer completing makes it ready. So a ready group stays ready until the
// main thread issues a new write to it.
class AliasGroup : public MainThreadCounted {
  public:
    AliasGroup(std::shared_ptr<CountableArray> current, std::shared_ptr<Task> last_writer,
               ArrayBlock* block)
        : current_(std::move(current)), last_writer_(std::move(last_writer)), block_(block) {}

    const std::shared_ptr<CountableArray>& current() const { return current_; }
    // Last task that writes this group; null once known to be completed.
    const std::shared_ptr<Task>& last_writer() const { return last_writer_; }
    // Error of the last completed writer, if it failed (a reference into
    // the runtime's ErrorTable).
    const ErrorRef& last_error() const { return last_error_; }

    bool ready() {
        if (last_writer_ && last_writer_->completed()) {
            if (last_writer_->error()) {
                last_error_ = last_writer_->error();
            }
            last_writer_.reset(); // break the group -> task reference early
        }
        return !last_writer_;
    }

    static void last_ref_dropped(AliasGroup* g);

  private:
    friend class Runtime;

    std::shared_ptr<CountableArray> current_;
    std::shared_ptr<Task> last_writer_;
    ErrorRef last_error_;
    ArrayBlock* block_;
    // The issue site whose op's data flow this data is in, and the op count
    // at that issue (OffloadPolicy::credit, ChainScore); null: none.
    const char* chain_ = nullptr;
    std::uint64_t chain_seq_ = 0;
};

// An array as seen by the main thread (docs/design/array-data-model.md, A1): its
// Metadata (size descriptor) and its AliasGroup (data). Main-thread only:
// workers never touch Array or AliasGroup objects, only the Metadata
// objects, CountableArrays and Tasks they refer to. The array, not its Metadata, holds
// the group (A5): a task's operand keeps the Metadata alive, and must not keep
// the group's current version alive with it (run-time elision, D11).
class Array : public MainThreadCounted {
  public:
    Array(MetadataPtr meta, Ref<AliasGroup> group, ArrayBlock* block)
        : meta_(std::move(meta)), group_(std::move(group)), block_(block) {}

    // The array's size descriptor. An array sized by its task (S2,
    // Runtime::issue_deferred) is waited for until its layout is known (a
    // metadata query, S5): the main thread never reads an incomplete one.
    const Layout& layout() const {
        if (meta_->task()) [[unlikely]] {
            wait_configured();
        }
        return meta_->sd();
    }
    const MetadataPtr& meta() const { return meta_; }
    const Ref<AliasGroup>& group() const { return group_; }
    // The current data version.
    const std::shared_ptr<CountableArray>& buffer() const { return group_->current(); }
    const std::shared_ptr<Task>& producer() const { return group_->last_writer(); }
    bool ready() { return group_->ready(); }

    static void last_ref_dropped(Array* a);

  private:
    friend class Runtime;
    [[gnu::noinline]] void wait_configured() const;

    MetadataPtr meta_;
    Ref<AliasGroup> group_;
    ArrayBlock* block_;
};

// One allocation for an array and, for a base array, its group. Each object
// is destroyed when its last reference goes; the block is freed when both are
// gone. (The base array's reference to its own group is an ordinary one: the
// group outlives it only while views refer to it.)
struct ArrayBlock {
    int alive = 0; // objects not yet destroyed
    std::optional<AliasGroup> group;
    std::optional<Array> array;
};

using ArrayPtr = Ref<Array>;

using ShouldOffload = std::function<bool(const Task&)>;

struct RuntimeOptions {
    std::size_t workers = 0; // 0: default_worker_count(), physical cores - 1
    std::size_t max_active_tasks = kDefaultMaxActiveTasks;
    // Dispatch policy for tasks whose inputs are all ready (see offload.hpp).
    OffloadOptions offload;
    // Overrides the policy if set: true -> worker, false -> run immediately on
    // the main thread.
    ShouldOffload should_offload;
    // Called on the main thread right before and after each blocking wait
    // inside issue() (back-pressure, live exports): the Python bridge
    // releases the GIL there, since a running task may need it
    // (UfuncLoop::call_watched). Null: nothing to do.
    void (*block_enter)() = nullptr;
    void (*block_leave)() = nullptr;
    // When a worker's completion makes several tasks runnable, those with
    // fewer elements than this stay with that worker (its local
    // queue) and run there next (Task::work: the largest operand): handing one to another worker
    // costs a wake-up and moves its inputs to another core, more than it computes. Larger ones go
    // to the shared queue. A spinning (idle) worker takes local tasks over from busy ones. 0: all
    // go to the shared queue.
    std::int64_t local_below = 65536;
    // Optimization level (docs/design/goals.md, "Optimization levels"): 1 (O1,
    // default) keeps results bit-identical to NumPy; 2 (O2) allows stage
    // pipelining and single-threaded BLAS calls, results correct but possibly
    // deviating from NumPy's. 3-5 are accepted and act as 2 until they are
    // implemented.
    int opt_level = 1;
};

class Runtime {
  public:
    explicit Runtime(RuntimeOptions options = {});
    ~Runtime(); // waits for all tasks, then stops the workers

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // --- Main thread only ---------------------------------------------------

    // A ready array over existing memory.
    ArrayPtr wrap(Layout layout, std::shared_ptr<CountableArray> buffer);
    // A view of `base`: same alias group, different layout (shape, strides,
    // offset into its current data version). Metadata only, no task.
    ArrayPtr view(const ArrayPtr& base, Layout&& layout);
    // O2: a view of `base` while its task still completes base's SD (S2):
    // the view's SD follows from base's by `steps` (basic indexing) or
    // `perm` (a transpose) when it is known (ViewRule). Its rank is known
    // now; metadata queries wait as for base. Main thread.
    ArrayPtr view_deferred(const ArrayPtr& base, int rank, std::vector<ViewStep> steps,
                           std::vector<int> perm, const char* issue_site);
    // O2: `base` (1-d, sized by its task) reshaped into `dims` (order 'C' or
    // 'F'): a view by a rule as view_deferred. Main thread.
    ArrayPtr reshape_deferred(const ArrayPtr& base, Shape dims, char order, const char* issue_site);

    // Issues one array op and returns immediately.
    //  - `inputs` are read.
    //  - `inouts` are read and written in place (in/out semantics). If the
    //    old data is still needed (pending readers: async_reads > 0, or an
    //    input of this op overlaps the in/out with a different layout), the
    //    op writes a new buffer instead (copy-on-write, see CLAUDE.md).
    //  - `new_outputs` are layouts of arrays the op creates.
    //  - `issue_site` must outlive the task: a string literal or an interned
    //    string (the C ABI interns the sites it receives).
    // Kernel operands: inputs = [inputs..., inouts...],
    //                  outputs = [inouts..., new outputs...].
    // `results` receives one array per new output.
    // `work_elements` sizes the op for the dispatch policy; by default the
    // elements of the first output (a reduction passes its input's size).
    // `output_buffers` (optional, one per new output) supplies the memory of
    // a new output, laid out as given (e.g. a result NumPy allocated).
    //
    // Direct path: an op below the dispatch policy's `sync_below` whose
    // inputs and in/outs are all ready (no error, in/outs without readers
    // and not read through another view) runs right here, without a task:
    // its kernel is called on the main thread and its outputs are created
    // ready. The task path would run it inline too (rule 1 of the policy),
    // but pays for the task, its operand capture, read counting, the
    // look-ahead slot and the completion protocol. A kernel error is
    // recorded in errors() and the outputs refer to it, as with a task.
    void issue(const std::shared_ptr<Kernel>& kernel, std::span<const ArrayPtr> inputs,
               std::span<const ArrayPtr> inouts, std::span<const Layout> new_outputs,
               const char* issue_site, std::span<ArrayPtr> results, std::int64_t work_elements = -1,
               std::span<const std::shared_ptr<CountableArray>> output_buffers = {});
    std::vector<ArrayPtr> issue(const std::shared_ptr<Kernel>& kernel,
                                std::span<const ArrayPtr> inputs, std::span<const ArrayPtr> inouts,
                                std::span<const Layout> new_outputs, const char* issue_site);

    // Issues an op whose output sizes and layouts its task computes
    // (pipelining.md, S2; Kernel::size, Kernel::layout): the outputs are
    // created with their dtype only, no size, no memory. The task sizes them
    // when every input reached SizeCompleted and lays them out when every
    // input reached Allocated, on the thread delivering the last signal or
    // in a pool job, never on the main thread. Inputs only (no in/outs).
    // `ranks`: the outputs' numbers of dimensions, which the caller knows at
    // issue (the kernel's size step gives shapes of exactly that rank), so
    // later ops can be issued against them without a wait (rank()).
    void issue_deferred(const std::shared_ptr<Kernel>& kernel, std::span<const ArrayPtr> inputs,
                        std::span<const DType> outputs, std::span<const int> ranks,
                        const char* issue_site, std::span<ArrayPtr> results);
    // Metadata query: waits until `a`'s size (SizeCompleted) or layout
    // (Allocated: strides, offset) is known. No wait for arrays sized at
    // issue. Returns the error of an op that could not size it.
    ErrorRef wait_stage(Array& a, TaskState s);
    // True if `a`'s size or layout may still be unknown (cheap; main thread).
    static bool unsized(const Array& a) { return static_cast<bool>(a.meta()->task()); }
    // `a`'s number of dimensions, without a wait (main thread).
    static int rank(const Array& a) {
        const Metadata& m = *a.meta();
        return m.task() ? m.rank() : static_cast<int>(m.sd().shape.size());
    }

    // Sync point: waits until `a` is ready. Returns the error of the task that
    // produced it, if any.
    ErrorRef wait(Array& a);
    // Sync point before NumPy code on the main thread writes `a`'s buffer
    // directly (design doc D7): waits for `a`'s producer; if NumPy views of
    // the buffer are alive (host_views), switches the group to a copy first
    // (copy-on-write, so those views keep their data); then waits until no
    // issued task reads the buffer. Afterwards nothing but the caller touches
    // the buffer until the main thread issues the next op.
    ErrorRef prepare_host_write(Array& a);
    // Waits until no task is active.
    void wait_all();

    // --- Any thread ---------------------------------------------------------
    std::size_t active_tasks() const { return active_.load(std::memory_order_acquire); }
    // Main thread: the dispatch policy and its counters.
    const OffloadPolicy& offload_policy() const { return policy_; }
    // Ops that took the direct path (see issue()).
    std::uint64_t direct_count() const { return direct_; }
    // Tasks issued so far (ops that did not take the direct path, plus the
    // copy-on-write copies issued for them).
    std::uint64_t task_count() const { return seq_ - direct_ - views_; }
    // Sync points so far that found their array pending (wait() blocked; main
    // thread; wait_all() not counted).
    std::uint64_t sync_count() const { return syncs_; }
    // Metadata queries (wait_stage) that found the size or layout unknown, so
    // the main thread blocked (main thread only).
    std::uint64_t size_wait_count() const { return size_waits_; }
    // Sleeping workers enqueue() woke (any thread; relaxed).
    std::uint64_t wake_count() const { return wakes_.load(std::memory_order_relaxed); }
    // Main thread, before issuing ops: replaces the dispatch policy.
    void set_offload_options(const OffloadOptions& o) {
        policy_ = OffloadPolicy(o);
        policy_.set_workers(workers_.size());
    }
    std::size_t worker_count() const { return workers_.size(); }
    // Main thread: the optimization level (RuntimeOptions::opt_level), set
    // before the first op is issued.
    int opt_level() const { return options_.opt_level; }
    void set_opt_level(int level) { options_.opt_level = level; }
    // The one store of failed ops' errors (see ErrorTable). Any thread.
    ErrorTable& errors() { return *errors_; }
    // Main thread: NumPy's errstate for the ops issued from now on (the
    // bridge sets it when the issuing code's errstate changes).
    void set_fp_mode(FpMode m) { fp_mode_ = m; }

  private:
    // Executes `task` on the current thread and returns its runnable consumers.
    TaskList run(const std::shared_ptr<Task>& task);
    // The configuring tasks whose steps became due on this thread: run them
    // (a worker) or queue them as step jobs (the main thread).
    [[gnu::noinline]] void due_steps(bool here);
    [[gnu::noinline]] TaskList step_job(Task& task);
    [[gnu::noinline]] void configure_now(Task& task);
    // Main thread, waiting for a stage: runs the steps it queued as step jobs
    // (main_steps_) itself, and the steps they make due.
    [[gnu::noinline]] void help_steps();
    void enqueue(std::shared_ptr<Task> task);
    void trigger_from_main(const std::shared_ptr<Task>& task, std::int64_t elements);
    // The direct path of issue(): see there.
    bool issue_direct(const std::shared_ptr<Kernel>& kernel, std::span<const ArrayPtr> inputs,
                      std::span<const ArrayPtr> inouts, std::span<const Layout> new_outputs,
                      const char* issue_site, std::span<ArrayPtr> results, std::int64_t elements,
                      std::span<const std::shared_ptr<CountableArray>> output_buffers);
    void acquire_slot();
    // Registers `task` on its pending producers and triggers it if none.
    void launch(const std::shared_ptr<Task>& task, const TaskList& producers,
                std::int64_t elements);
    // Copy-on-write with views: switches `g` to a new version holding a copy of
    // the old data (issued as its own task); views keep their layouts.
    void issue_group_copy(AliasGroup& g);

    std::thread::id main_thread_ = std::this_thread::get_id(); // created there
    RuntimeOptions options_;
    std::shared_ptr<ErrorTable> errors_ = std::make_shared<ErrorTable>();
    OffloadPolicy policy_;     // main thread only
    std::uint64_t seq_ = 0;    // ops issued so far (main thread only)
    std::uint64_t views_ = 0;  // deferred views among them (an id for their errors, no task)
    FpMode fp_mode_;           // errstate of the ops issued now (main thread only)
    std::uint64_t direct_ = 0; // ... of which ran directly, without a task
    std::uint64_t syncs_ = 0;
    std::uint64_t size_waits_ = 0; // waits that found the array pending (main thread only)
    // Tasks whose steps the main thread queued as step jobs, not yet seen
    // laid out (main thread only). A step job waits in the pool behind
    // running kernels; a metadata query waiting for one runs it (help_steps).
    std::vector<std::shared_ptr<Task>> main_steps_;
    std::atomic<std::size_t> active_{0};
    // Threads waiting in wait_active_below(): run() notifies only if > 0.
    std::atomic<int> active_waiters_{0};
    std::atomic<std::size_t> wake_below_{0}; // a waiter's threshold (wait_active_below)
    // Blocks until fewer than `limit` tasks are active.
    void wait_active_below(std::size_t limit);

    // Pool queue (`queue_`, lock-free, task_queue.hpp). Idle workers spin
    // for a short while (checking `queued_`) before they sleep; enqueue()
    // wakes a sleeper only if no worker is spinning, so a stream of small
    // tasks does not cost a wake-up syscall per task. Once an enqueue() found
    // no spinner and had to wake a sleeper (`stream_`: tasks arrive faster
    // than one spinner takes them), a spinner that finds work hands the role
    // to a sleeper, so the next enqueue() again finds a spinner: the wake-up
    // is paid by a worker, not by the issuing main thread. A spin that finds
    // nothing ends the stream; without one, a hand-off would only delay the
    // task the spinner found.
    // No more wake-ups are outstanding than tasks are queued
    // (`wakes_pending_`; .NET's ThreadPool keeps one thread request):
    // enqueue() wakes only then, a woken worker takes one off when it runs,
    // and a worker that took a task and leaves more queued wakes the next
    // one. Without it the main thread notified on every enqueue() in the
    // window between a notify and the woken worker's start (1,500^2 column
    // loop, 8 workers: ~2 of 3 tasks, a futex call each on the thread that
    // limits the loop).
    // Sleepers are woken last in, first out (`sleepers_`, one semaphore
    // per worker), as .NET's LowLevelLifoSemaphore: the worker that slept
    // last has its caches warm, and workers the load does not need stay
    // asleep instead of taking turns. Only sleeping and waking take
    // `sleep_mutex_`; enqueue() takes no lock unless it wakes someone.
    TaskQueue queue_;
    std::atomic<std::size_t> queued_{0}; // incremented before the push
    std::atomic<int> spinning_{0};
    std::atomic<int> sleeping_{0}; // == sleepers_.size(), changed under sleep_mutex_
    std::atomic<bool> stream_{false};
    std::atomic<std::size_t> wakes_pending_{0}; // woken, not yet running
    std::atomic<std::uint64_t> wakes_{0};       // wake-ups by enqueue()
    std::atomic<bool> stopping_{false};
    std::mutex sleep_mutex_;
    std::vector<std::size_t> sleepers_; // guarded by sleep_mutex_, LIFO
    struct WorkerSlot {
        std::binary_semaphore wake{0};
    };
    std::vector<std::unique_ptr<WorkerSlot>> slots_;
    // Wakes the last sleeper if no worker spins and fewer wake-ups are
    // pending than `wanted` (the tasks a woken worker would find). Any
    // thread; takes sleep_mutex_ only when it wakes. True if it woke one.
    bool wake_one(std::size_t wanted);
    // Sleeps until woken (or until work or stopping shows up while it
    // registers). Worker threads only.
    void sleep(std::size_t id);
    std::shared_ptr<Task> try_pop();
    // Per worker: unblocked small tasks (RuntimeOptions::local_below). The
    // owner pushes and pops at the back (most recent: its inputs are hot),
    // a spinning worker steals from the front. `local_` counts all of them.
    struct LocalQueue {
        std::mutex mutex;
        std::deque<std::shared_ptr<Task>> tasks;
    };
    std::vector<std::unique_ptr<LocalQueue>> locals_;
    std::atomic<std::size_t> local_{0};
    std::shared_ptr<Task> pop_local(std::size_t worker);
    std::shared_ptr<Task> steal(std::size_t thief);
    void worker_loop(std::size_t id);
    std::vector<std::thread> workers_;
};

} // namespace viproc
