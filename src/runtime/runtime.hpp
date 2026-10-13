// The runtime: issues array ops as tasks and runs them on worker threads.
#pragma once

#include "buffer.hpp"
#include "layout.hpp"
#include "offload.hpp"
#include "ref.hpp"
#include "task.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace viproc {

// Look-ahead limit: maximum number of issued, not yet completed tasks.
inline constexpr std::size_t kDefaultMaxActiveTasks = 1000;

struct ArrayBlock;

// Shared storage of a base array and all of its views. Main-thread only.
//
// The storage, not the individual array, carries the data version (buffer)
// and the last task writing it (producer): a write through one view makes the
// base and all other views pending, and copy-on-write switches all of them to
// the new buffer together.
//
// State: `pending` while `producer` is set and not completed, `ready`
// otherwise. Only the main thread sets `producer` (ready -> pending); the
// producer completing makes it ready. So a ready storage stays ready until the
// main thread issues a new write to it.
class Storage : public MainThreadCounted {
  public:
    Storage(std::shared_ptr<Buffer> buffer, std::shared_ptr<Task> producer, ArrayBlock* block)
        : buffer_(std::move(buffer)), producer_(std::move(producer)), block_(block) {}

    const std::shared_ptr<Buffer>& buffer() const { return buffer_; }
    // Last task that writes this storage; null once known to be completed.
    const std::shared_ptr<Task>& producer() const { return producer_; }
    // Error of the last completed producer, if it failed.
    const std::optional<TaskError>& last_error() const { return last_error_; }

    bool ready() {
        if (producer_ && producer_->completed()) {
            last_error_ = producer_->error();
            producer_.reset(); // break the storage -> task reference early
        }
        return !producer_;
    }

    static void last_ref_dropped(Storage* s);

  private:
    friend class Runtime;

    std::shared_ptr<Buffer> buffer_;
    std::shared_ptr<Task> producer_;
    std::optional<TaskError> last_error_;
    ArrayBlock* block_;
};

// An array as seen by the main thread: a layout (view) onto a shared storage.
// Main-thread only: workers never touch Array or Storage objects, only the
// Buffers and Tasks they refer to.
class Array : public MainThreadCounted {
  public:
    Array(Layout layout, Ref<Storage> storage, ArrayBlock* block)
        : layout_(std::move(layout)), storage_(std::move(storage)), block_(block) {}

    const Layout& layout() const { return layout_; }
    const Ref<Storage>& storage() const { return storage_; }
    const std::shared_ptr<Buffer>& buffer() const { return storage_->buffer(); }
    const std::shared_ptr<Task>& producer() const { return storage_->producer(); }
    bool ready() { return storage_->ready(); }

    static void last_ref_dropped(Array* a);

  private:
    friend class Runtime;

    Layout layout_;
    Ref<Storage> storage_;
    ArrayBlock* block_;
};

// One allocation for an array and, for a base array, its storage. Each
// object is destroyed when its last reference goes; the block is freed when
// both are gone. (The base array's reference to its own storage is an
// ordinary one: the storage outlives it only while views refer to it.)
struct ArrayBlock {
    int alive = 0; // objects not yet destroyed
    std::optional<Storage> storage;
    std::optional<Array> array;
};

using ArrayPtr = Ref<Array>;

using ShouldOffload = std::function<bool(const Task&)>;

struct RuntimeOptions {
    std::size_t workers = 0; // 0: std::thread::hardware_concurrency()
    std::size_t max_active_tasks = kDefaultMaxActiveTasks;
    // Dispatch policy for tasks whose inputs are all ready (see offload.hpp).
    OffloadOptions offload;
    // Overrides the policy if set: true -> worker, false -> run immediately on
    // the main thread.
    ShouldOffload should_offload;
};

class Runtime {
  public:
    explicit Runtime(RuntimeOptions options = {});
    ~Runtime(); // waits for all tasks, then stops the workers

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // --- Main thread only ---------------------------------------------------

    // A ready array over existing memory.
    ArrayPtr wrap(Layout layout, std::shared_ptr<Buffer> buffer);
    // A view of `base`: same storage, different layout (shape, strides,
    // offset into the storage's buffer). Metadata only, no task.
    ArrayPtr view(const ArrayPtr& base, Layout layout);

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
    // look-ahead slot and the completion protocol. A kernel error is stored
    // on the outputs (raised at their next sync point, as with a task).
    void issue(const std::shared_ptr<Kernel>& kernel, std::span<const ArrayPtr> inputs,
               std::span<const ArrayPtr> inouts, std::span<const Layout> new_outputs,
               const char* issue_site, std::span<ArrayPtr> results, std::int64_t work_elements = -1,
               std::span<const std::shared_ptr<Buffer>> output_buffers = {});
    std::vector<ArrayPtr> issue(const std::shared_ptr<Kernel>& kernel,
                                std::span<const ArrayPtr> inputs, std::span<const ArrayPtr> inouts,
                                std::span<const Layout> new_outputs, const char* issue_site);

    // Sync point: waits until `a` is ready. Returns the error of the task that
    // produced it, if any.
    std::optional<TaskError> wait(Array& a);
    // Sync point before NumPy code on the main thread writes `a`'s buffer
    // directly (design doc D7): waits for `a`'s producer; if NumPy views of
    // the buffer are alive (host_views), switches the storage to a copy first
    // (copy-on-write, so those views keep their data); then waits until no
    // issued task reads the buffer. Afterwards nothing but the caller touches
    // the buffer until the main thread issues the next op.
    std::optional<TaskError> prepare_host_write(Array& a);
    // Waits until no task is active.
    void wait_all();

    // --- Any thread ---------------------------------------------------------
    std::size_t active_tasks() const { return active_.load(std::memory_order_acquire); }
    // Main thread: the dispatch policy and its counters.
    const OffloadPolicy& offload_policy() const { return policy_; }
    // Ops that took the direct path (see issue()).
    std::uint64_t direct_count() const { return direct_; }
    // Main thread, before issuing ops: replaces the dispatch policy.
    void set_offload_options(const OffloadOptions& o) { policy_ = OffloadPolicy(o); }
    std::size_t worker_count() const { return workers_.size(); }

  private:
    void worker_loop();
    // Executes `task` on the current thread and returns its runnable consumers.
    std::vector<std::shared_ptr<Task>> run(const std::shared_ptr<Task>& task);
    void enqueue(std::shared_ptr<Task> task);
    void trigger_from_main(const std::shared_ptr<Task>& task, std::int64_t elements);
    // The direct path of issue(): see there.
    bool issue_direct(const std::shared_ptr<Kernel>& kernel, std::span<const ArrayPtr> inputs,
                      std::span<const ArrayPtr> inouts, std::span<const Layout> new_outputs,
                      const char* issue_site, std::span<ArrayPtr> results, std::int64_t elements,
                      std::span<const std::shared_ptr<Buffer>> output_buffers);
    void acquire_slot();
    // Registers `task` on its pending producers and triggers it if none.
    void launch(const std::shared_ptr<Task>& task,
                const std::vector<std::shared_ptr<Task>>& producers, std::int64_t elements);
    // Copy-on-write with views: switches `s` to a new buffer holding a copy of
    // the old data (issued as its own task); views keep their layouts.
    void issue_storage_copy(Storage& s);

    RuntimeOptions options_;
    OffloadPolicy policy_;     // main thread only
    std::uint64_t seq_ = 0;    // ops issued so far (main thread only)
    std::uint64_t direct_ = 0; // ... of which ran directly, without a task
    std::atomic<std::size_t> active_{0};
    // Threads waiting in wait_active_below(): run() notifies only if > 0.
    std::atomic<int> active_waiters_{0};
    // Blocks until fewer than `limit` tasks are active.
    void wait_active_below(std::size_t limit);

    // Pool queue. Idle workers spin for a short while (checking `queued_`
    // without the lock) before they sleep on `queue_cv_`; enqueue() wakes a
    // sleeper only if no worker is spinning, so a stream of small tasks does
    // not cost a wake-up syscall per task.
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Task>> queue_;
    std::atomic<std::size_t> queued_{0};
    std::atomic<int> spinning_{0};
    std::atomic<int> sleeping_{0};
    bool stopping_ = false;
    std::shared_ptr<Task> try_pop();
    std::vector<std::thread> workers_;
};

} // namespace viproc
