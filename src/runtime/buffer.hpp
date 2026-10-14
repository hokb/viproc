// Element memory of arrays.
#pragma once

#include <atomic>
#include <cstddef>
#include <memory>

namespace viproc {

// A block of element memory, taken when the buffer is created: up to
// kInlineBytes inside the Buffer object, larger ones from the element pool
// (docs/design/memory-management.md, D3/D10). Buffers are created by the main
// thread only (at issue); the last reference may go on any thread, which
// returns the block to the pool.
//
// Thread safety: the producing task writes `data()` after all tasks it depends
// on completed; consumers read it only after the producer completed (acquire
// on the producer's state).
class Buffer {
  public:
    // A buffer of `nbytes` with uninitialized element memory.
    explicit Buffer(std::size_t nbytes);
    // Wraps memory owned elsewhere (e.g. by a NumPy array); `owner` keeps it alive.
    Buffer(void* data, std::size_t nbytes, std::shared_ptr<void> owner);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    std::byte* data() const { return data_; }
    // Logical size: the bytes the arrays on this buffer cover.
    std::size_t nbytes() const { return nbytes_; }
    // Size of the pool block behind the buffer (0: inline or foreign memory).
    std::size_t capacity() const { return capacity_; }

    // Number of issued, not yet completed tasks that read this buffer.
    // Incremented only by the main thread (at issue), decremented by the
    // reading task when it completes. See "Reference counting" in CLAUDE.md.
    std::atomic<int> async_reads{0};

    // Read-only NumPy views of this buffer handed out by an export (design
    // doc D7) that are still alive. Main thread only (exports are created and
    // released with the GIL). Like async_reads, a later in-place write must
    // not touch the buffer while it is > 0 (copy-on-write).
    int host_views = 0;

    // Any reader a writer must not disturb: issued tasks or NumPy views.
    bool has_readers() const {
        return host_views > 0 || async_reads.load(std::memory_order_acquire) > 0;
    }

    // A reading task is done with the buffer (any thread).
    void release_read() {
        if (async_reads.fetch_sub(1, std::memory_order_seq_cst) == 1 &&
            read_waiters_.load(std::memory_order_seq_cst) > 0) {
            async_reads.notify_all();
        }
    }
    // Main thread: blocks until no issued task reads the buffer any more.
    void wait_task_readers() {
        if (async_reads.load(std::memory_order_acquire) == 0) {
            return;
        }
        // seq_cst pairs with release_read(): either it sees the waiter and
        // notifies, or the load below sees its decrement.
        read_waiters_.fetch_add(1, std::memory_order_seq_cst);
        for (int n = async_reads.load(std::memory_order_seq_cst); n != 0;
             n = async_reads.load(std::memory_order_seq_cst)) {
            async_reads.wait(n, std::memory_order_acquire);
        }
        read_waiters_.fetch_sub(1, std::memory_order_relaxed);
    }

    // Up to this many bytes live inside the Buffer object itself: one
    // allocation for small arrays instead of two.
    static constexpr std::size_t kInlineBytes = 128;

  private:
    std::atomic<int> read_waiters_{0};
    alignas(64) std::byte inline_[kInlineBytes];
    std::byte* data_ = nullptr;
    std::size_t nbytes_;
    std::size_t capacity_ = 0; // > 0: data_ is a pool block
    std::shared_ptr<void> owner_;
};

} // namespace viproc
