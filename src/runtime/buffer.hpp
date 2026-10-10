// Element memory of arrays.
#pragma once

#include <atomic>
#include <cstddef>
#include <memory>

namespace viproc {

// A block of element memory. Created by the main thread (possibly without
// memory, for pending arrays); memory is allocated by the task that produces
// the data, in its `allocated` stage.
//
// Thread safety: `ensure_allocated()` is called only by the producing task,
// after all tasks it depends on have completed. Consumers read `data()` only
// after the producer completed (acquire on the producer's state).
class Buffer {
  public:
    // Unallocated buffer of `nbytes`; memory comes with ensure_allocated().
    explicit Buffer(std::size_t nbytes);
    // Wraps memory owned elsewhere (e.g. by a NumPy array); `owner` keeps it alive.
    Buffer(void* data, std::size_t nbytes, std::shared_ptr<void> owner);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void ensure_allocated();
    bool allocated() const { return data_ != nullptr; }
    std::byte* data() const { return data_; }
    std::size_t nbytes() const { return nbytes_; }

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
    bool owns_ = false;
    std::shared_ptr<void> owner_;
};

} // namespace viproc
