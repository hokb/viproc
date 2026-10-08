// Tasks: one per array op, carrying the op's progress through its stages.
#pragma once

#include "buffer.hpp"
#include "layout.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace viproc {

// Stages of a task, in order. A task passes each stage at most once and only
// forward; it may skip stages (DeviceSelected is unused in phase 1).
enum class TaskState : std::uint8_t {
    Created,
    SizeCompleted,
    DeviceSelected,
    Allocated,
    Completed,
};

// One argument of a kernel call: the buffer captured at issue time and the
// view onto it. Capturing the buffer (not the array) keeps readers on the data
// version they were issued against, even if the array switches buffers later
// (copy-on-write).
struct Operand {
    std::shared_ptr<Buffer> buffer;
    Layout layout;

    std::byte* first() const { return buffer->data() + layout.offset; }
};

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
};

struct TaskError {
    std::string message;
    std::string issue_site; // where the failing op was issued
};

class Task {
  public:
    Task(std::shared_ptr<Kernel> kernel, std::string issue_site);

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    TaskState state() const { return state_.load(std::memory_order_acquire); }
    bool completed() const { return state() == TaskState::Completed; }
    // Blocks until the task is completed.
    void wait_completed() const;

    const std::string& issue_site() const { return issue_site_; }
    // Valid once completed (acquire through state()).
    const std::optional<TaskError>& error() const { return error_; }
    int fp_flags() const { return fp_flags_; }

    // --- Main thread, while issuing -------------------------------------
    void set_operands(std::vector<Operand> inputs, std::vector<Operand> outputs);
    void advance(TaskState next);
    // Announces one more pending producer before registering on it.
    void add_pending_input() { pending_.fetch_add(1, std::memory_order_relaxed); }
    // Registers `consumer` for this task's completion. Returns false if this
    // task has already completed; the caller then accounts for it itself.
    bool add_consumer(const std::shared_ptr<Task>& consumer);
    // Drops the issue guard (see pending_). Returns true if the task is
    // runnable now, i.e. all producers it waits on have completed.
    bool release_issue_guard() { return input_done(nullptr); }
    // Accounts for a producer that turned out to be completed already.
    bool producer_already_done(const Task& producer) { return input_done(&producer); }
    // Marks an input as failed (its producer completed with an error before
    // this task was issued). The task then skips its kernel and fails too.
    void add_upstream_error(const TaskError& error);

    // --- Any thread ---------------------------------------------------------
    // Runs the kernel (or skips it if an input failed), completes the task and
    // returns the consumers that became runnable. Never runs consumers itself:
    // the calling thread schedules them as continuations.
    std::vector<std::shared_ptr<Task>> execute();

  private:
    // Called once per producer when it completes (and once for the issue
    // guard with nullptr). Returns true when the last one arrives.
    bool input_done(const Task* producer);

    std::atomic<TaskState> state_{TaskState::Created};
    // Number of producers still to complete, plus one guard held by the main
    // thread while it registers the task, so it cannot start early.
    std::atomic<int> pending_{1};

    std::shared_ptr<Kernel> kernel_;
    std::vector<Operand> inputs_;
    std::vector<Operand> outputs_;
    std::string issue_site_;

    std::mutex mutex_; // guards consumers_, upstream_error_ and the switch to Completed
    std::vector<std::shared_ptr<Task>> consumers_;
    std::optional<TaskError> upstream_error_;

    std::optional<TaskError> error_;
    int fp_flags_ = 0;
};

} // namespace viproc
