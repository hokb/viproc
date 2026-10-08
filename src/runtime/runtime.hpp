// The runtime: issues array ops as tasks and runs them on worker threads.
#pragma once

#include "buffer.hpp"
#include "layout.hpp"
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

// An array as seen by the main thread. All fields are main-thread only:
// workers never touch Array objects, only the Buffers and Tasks it refers to.
//
// State: `pending` while `producer` is set and not completed, `ready`
// otherwise. Only the main thread sets `producer` (ready -> pending); the
// producer completing makes it ready. So a ready array stays ready until the
// main thread issues a new write to it.
class Array {
  public:
    Array(Layout layout, std::shared_ptr<Buffer> buffer, std::shared_ptr<Task> producer)
        : layout_(std::move(layout)), buffer_(std::move(buffer)), producer_(std::move(producer)) {}

    const Layout& layout() const { return layout_; }
    const std::shared_ptr<Buffer>& buffer() const { return buffer_; }
    // Last task that writes this array; null once known to be completed.
    const std::shared_ptr<Task>& producer() const { return producer_; }

    bool ready() {
        if (producer_ && producer_->completed()) {
            last_error_ = producer_->error();
            producer_.reset(); // break the array -> task reference early
        }
        return !producer_;
    }

  private:
    friend class Runtime;

    Layout layout_;
    std::shared_ptr<Buffer> buffer_;
    std::shared_ptr<Task> producer_;
    std::optional<TaskError> last_error_; // error of the completed producer
};

using ArrayPtr = std::shared_ptr<Array>;

using ShouldOffload = std::function<bool(const Task&)>;

struct RuntimeOptions {
    std::size_t workers = 0; // 0: std::thread::hardware_concurrency()
    std::size_t max_active_tasks = kDefaultMaxActiveTasks;
    // Decides for a task whose inputs are all ready: true -> worker,
    // false -> run immediately on the main thread. Default: always offload.
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

    // Issues one array op and returns immediately.
    //  - `inputs` are read.
    //  - `inouts` are read and written in place (in/out semantics). If pending
    //    tasks still read an in/out array's buffer (async_reads > 0), the op
    //    writes a new buffer instead (copy-on-write).
    //  - `new_outputs` are layouts of arrays the op creates.
    // Kernel operands: inputs = [inputs..., inouts...],
    //                  outputs = [inouts..., new outputs...].
    std::vector<ArrayPtr> issue(std::shared_ptr<Kernel> kernel, std::span<const ArrayPtr> inputs,
                                std::span<const ArrayPtr> inouts,
                                std::span<const Layout> new_outputs, std::string issue_site);

    // Sync point: waits until `a` is ready. Returns the error of the task that
    // produced it, if any.
    std::optional<TaskError> wait(Array& a);
    // Waits until no task is active.
    void wait_all();

    // --- Any thread ---------------------------------------------------------
    std::size_t active_tasks() const { return active_.load(std::memory_order_acquire); }
    std::size_t worker_count() const { return workers_.size(); }

  private:
    void worker_loop();
    // Executes `task` on the current thread and returns its runnable consumers.
    std::vector<std::shared_ptr<Task>> run(const std::shared_ptr<Task>& task);
    void enqueue(std::shared_ptr<Task> task);
    void trigger_from_main(const std::shared_ptr<Task>& task);

    RuntimeOptions options_;
    std::atomic<std::size_t> active_{0};

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Task>> queue_;
    bool stopping_ = false;
    std::vector<std::thread> workers_;
};

} // namespace viproc
