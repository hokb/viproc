// The runtime: issues array ops as tasks and runs them on worker threads.
#pragma once

#include "buffer.hpp"
#include "layout.hpp"
#include "offload.hpp"
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
class Storage {
  public:
    explicit Storage(std::shared_ptr<Buffer> buffer, std::shared_ptr<Task> producer = nullptr)
        : buffer_(std::move(buffer)), producer_(std::move(producer)) {}

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

  private:
    friend class Runtime;

    std::shared_ptr<Buffer> buffer_;
    std::shared_ptr<Task> producer_;
    std::optional<TaskError> last_error_;
};

// An array as seen by the main thread: a layout (view) onto a shared storage.
// Main-thread only: workers never touch Array or Storage objects, only the
// Buffers and Tasks they refer to.
class Array {
  public:
    Array(Layout layout, std::shared_ptr<Storage> storage)
        : layout_(std::move(layout)), storage_(std::move(storage)) {}

    const Layout& layout() const { return layout_; }
    const std::shared_ptr<Storage>& storage() const { return storage_; }
    const std::shared_ptr<Buffer>& buffer() const { return storage_->buffer(); }
    const std::shared_ptr<Task>& producer() const { return storage_->producer(); }
    bool ready() { return storage_->ready(); }

  private:
    friend class Runtime;

    Layout layout_;
    std::shared_ptr<Storage> storage_;
};

using ArrayPtr = std::shared_ptr<Array>;

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
    std::vector<ArrayPtr> issue(std::shared_ptr<Kernel> kernel, std::span<const ArrayPtr> inputs,
                                std::span<const ArrayPtr> inouts,
                                std::span<const Layout> new_outputs, const char* issue_site);

    // Sync point: waits until `a` is ready. Returns the error of the task that
    // produced it, if any.
    std::optional<TaskError> wait(Array& a);
    // Waits until no task is active.
    void wait_all();

    // --- Any thread ---------------------------------------------------------
    std::size_t active_tasks() const { return active_.load(std::memory_order_acquire); }
    // Main thread: the dispatch policy and its counters.
    const OffloadPolicy& offload_policy() const { return policy_; }
    // Main thread, before issuing ops: replaces the dispatch policy.
    void set_offload_options(const OffloadOptions& o) { policy_ = OffloadPolicy(o); }
    std::size_t worker_count() const { return workers_.size(); }

  private:
    void worker_loop();
    // Executes `task` on the current thread and returns its runnable consumers.
    std::vector<std::shared_ptr<Task>> run(const std::shared_ptr<Task>& task);
    void enqueue(std::shared_ptr<Task> task);
    void trigger_from_main(const std::shared_ptr<Task>& task, std::int64_t elements);
    void acquire_slot();
    // Registers `task` on its pending producers and triggers it if none.
    void launch(const std::shared_ptr<Task>& task,
                const std::vector<std::shared_ptr<Task>>& producers, std::int64_t elements);
    // Copy-on-write with views: switches `s` to a new buffer holding a copy of
    // the old data (issued as its own task); views keep their layouts.
    void issue_storage_copy(Storage& s);

    RuntimeOptions options_;
    OffloadPolicy policy_;  // main thread only
    std::uint64_t seq_ = 0; // ops issued so far (main thread only)
    std::atomic<std::size_t> active_{0};

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
