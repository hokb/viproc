// Element memory of arrays: one data version, one handle per device.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace viproc {

// Devices a CountableArray can hold a copy on. Phase 1: the host only.
enum Device : std::uint8_t { kHost = 0, kDeviceCount };

// One version of an array's data (docs/design/array-data-model.md, A1): one handle
// per device, each empty or a valid copy of the same values. Host memory is up
// to kInlineBytes inside the object, larger blocks come from the element pool
// (docs/design/memory-management.md, D3, D10, D11). CountableArrays are
// created by the main thread only (at issue); the last reference may go on
// any thread, which returns the blocks to their pools. Readers capture the
// CountableArray they were issued against; copy-on-write points the
// AliasGroup to a new one, so readers keep their version.
//
// A deferred buffer (`CountableArray(nbytes, CountableArray::deferred)`)
// takes its block as late as possible: the task producing it calls
// `ensure_allocated()` or `take_block()` right before its kernel runs, on
// whatever thread runs it.
//
// Thread safety: the producing task writes `data()` after all tasks it depends
// on completed; consumers read it only after the producer completed (acquire
// on the producer's state). `ensure_allocated()` may race with itself (the
// main thread pinning the memory of a pending buffer, D11): one block wins.
class CountableArray {
  public:
    // A buffer of `nbytes` with uninitialized element memory, taken now.
    explicit CountableArray(std::size_t nbytes);
    // A buffer of `nbytes` whose memory is taken later (ensure_allocated(),
    // take_block()). Small buffers (inline) have their memory right away.
    struct Deferred {};
    static constexpr Deferred deferred{};
    CountableArray(std::size_t nbytes, Deferred);
    // Wraps memory owned elsewhere (e.g. by a NumPy array); `owner` keeps it alive.
    CountableArray(void* data, std::size_t nbytes, std::shared_ptr<void> owner);
    ~CountableArray();

    CountableArray(const CountableArray&) = delete;
    CountableArray& operator=(const CountableArray&) = delete;

    // Null for a deferred buffer whose memory was not taken yet.
    std::byte* data() const { return handles_[kHost].data.load(std::memory_order_acquire); }
    bool allocated() const { return data() != nullptr; }
    // Element memory from the element pool (now or, deferred, later): a block
    // of capacity_for(nbytes) bytes, reusable by another buffer of the bin.
    bool pooled() const { return handles_[kHost].capacity != 0; }

    // Takes the memory of a deferred buffer from the pool if it has none yet
    // (any thread; concurrent calls agree on one block). Returns data().
    std::byte* ensure_allocated();
    // Moves `src`'s pool block into this deferred, unallocated buffer, if the
    // block has this buffer's capacity. The caller must hold the only
    // reference to `src` (nobody else can read it); `src` is left without
    // memory. Any thread. False if nothing was moved.
    bool take_block(CountableArray& src);
    // Logical size: the bytes the arrays on this buffer cover.
    std::size_t nbytes() const { return nbytes_; }
    // Size of the pool block behind the buffer (0: inline or foreign memory).
    std::size_t capacity() const { return handles_[kHost].capacity; }

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

    // Up to this many bytes live inside the CountableArray object itself: one
    // allocation for small arrays instead of two.
    static constexpr std::size_t kInlineBytes = 128;

  private:
    std::atomic<int> read_waiters_{0};
    alignas(64) std::byte inline_[kInlineBytes];
    // Memory on one device; capacity > 0: data is (or will be) a pool block.
    struct DeviceHandle {
        std::atomic<std::byte*> data{nullptr};
        std::size_t capacity = 0;
    };
    DeviceHandle handles_[kDeviceCount];
    std::size_t nbytes_;
    std::shared_ptr<void> owner_;
};

} // namespace viproc
