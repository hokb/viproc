// Pool for element memory (docs/design/memory-management.md, D3 and D10).
//
// Blocks are quantized into bins: a request of n bytes is rounded up to the
// next bin length, with 8 bins per power of two, so a block is at most 12.5 %
// larger than requested. All blocks of a bin have the same capacity: any
// cached block of the bin serves any request of the bin, found in O(1).
// Each bin is a LIFO free list (the most recently released block is the most
// likely to be in cache and mapped).
//
// Access model (D11): blocks are taken by the thread that runs the task
// producing them (a worker, or the main thread for ops it runs itself) and
// released on any thread. A thread that allocates keeps a few released blocks
// per bin in private free lists (no atomics, LIFO: hot in its cache); the
// rest goes onto a per-bin lock-free stack that an allocating thread takes
// over as a whole when its own list is empty (exchange: no ABA).
//
// Released blocks are not zeroed. A release that would grow the cached bytes
// beyond the bound goes straight back to the system.
//
// Blocks from 2 MiB on come page-aligned from the system and use huge pages
// where the OS allows it: transparent huge pages on Linux (madvise), large
// pages on Windows when the process holds SeLockMemoryPrivilege.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace viproc {

class ElementPool {
  public:
    // The process-wide pool.
    static ElementPool& instance();

    // A block of at least `nbytes` bytes (> 0); `capacity` receives its size.
    // Any thread. On failure: empties the pool, retries, then throws
    // std::bad_alloc.
    void* allocate(std::size_t nbytes, std::size_t& capacity);
    // Returns a block obtained from allocate(). Any thread.
    void release(void* block, std::size_t capacity);

    // Hands the calling thread's private blocks to the shared stacks and
    // unregisters it (runs at the exit of every allocating thread).
    void flush_thread();

    // Returns every cached block to the system: every thread's lists and the
    // shared stacks. Any thread.
    void trim();

    // Upper bound for cached bytes (default: half the physical memory, or
    // VIPROC_POOL_LIMIT; 0 disables caching).
    std::size_t limit() const { return limit_.load(std::memory_order_relaxed); }
    void set_limit(std::size_t bytes);

    struct Stats {
        std::uint64_t hits;    // requests served from the pool
        std::uint64_t misses;  // requests that allocated from the system
        std::uint64_t cached;  // bytes held by the pool now
        std::uint64_t dropped; // releases returned to the system (bound)
    };
    Stats stats() const;

    // Bin arithmetic (exposed for tests).
    static std::size_t capacity_for(std::size_t nbytes);

  private:
    ElementPool();

    static constexpr int kBins = 64 * 8;
    struct Node {
        Node* next;
    };
    struct Local; // per-thread free lists
    static Local& local();
    static int bin_of(std::size_t capacity);
    static void* system_allocate(std::size_t capacity);
    static void system_release(void* block, std::size_t capacity);
    void push_remote(int b, Node* first, Node* last);

    std::atomic<Node*> remote_[kBins] = {};
    // Threads with private free lists (they allocate), for trim().
    std::mutex registry_mutex_;
    std::vector<Local*> registry_;
    std::atomic<std::size_t> limit_;
    std::atomic<std::int64_t> cached_{0};
    std::atomic<std::uint64_t> hits_{0};
    std::atomic<std::uint64_t> misses_{0};
    std::atomic<std::uint64_t> dropped_{0};
};

} // namespace viproc
