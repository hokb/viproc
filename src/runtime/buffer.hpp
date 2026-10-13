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

    // Up to this many bytes live inside the Buffer object itself: one
    // allocation for small arrays instead of two.
    static constexpr std::size_t kInlineBytes = 128;

  private:
    alignas(64) std::byte inline_[kInlineBytes];
    std::byte* data_ = nullptr;
    std::size_t nbytes_;
    std::size_t capacity_ = 0; // > 0: data_ is a pool block
    std::shared_ptr<void> owner_;
};

} // namespace viproc
