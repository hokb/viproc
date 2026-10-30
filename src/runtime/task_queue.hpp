// TaskQueue: the worker pool's shared queue of runnable tasks, lock-free.
#pragma once

#include "task.hpp"

#include <atomic>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace viproc {

// Multi-producer, multi-consumer FIFO of tasks (Dmitry Vyukov's bounded
// MPMC queue): a push or pop is one CAS on its index plus a release store
// of the cell's sequence number, no lock. The main thread pushes every ready
// op it hands to the pool (`Runtime::enqueue`); a mutex there was taken once
// per op and contended with the workers polling the queue. A full ring
// (more runnable tasks than kCapacity: the look-ahead limit keeps it below
// 1,000 plus copy-on-write copies) spills into a locked overflow list, which
// costs one relaxed load per pop while it is empty. The order across ring
// and overflow is not strictly FIFO; tasks in the queue are independent.
// Any thread may push and pop.
class TaskQueue {
  public:
    static constexpr std::size_t kCapacity = 4096; // a power of two

    TaskQueue() : cells_(kCapacity) {
        for (std::size_t i = 0; i < kCapacity; ++i) {
            cells_[i].seq.store(i, std::memory_order_relaxed);
        }
    }
    TaskQueue(const TaskQueue&) = delete;
    TaskQueue& operator=(const TaskQueue&) = delete;

    void push(std::shared_ptr<Task> task) {
        std::size_t pos = tail_.load(std::memory_order_relaxed);
        for (;;) {
            Cell& c = cells_[pos & kMask];
            const std::size_t seq = c.seq.load(std::memory_order_acquire);
            const auto dif = static_cast<std::ptrdiff_t>(seq - pos);
            if (dif == 0) {
                if (tail_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    c.task = std::move(task);
                    c.seq.store(pos + 1, std::memory_order_release);
                    return;
                }
            } else if (dif < 0) {
                std::lock_guard lock(overflow_mutex_);
                overflow_.push_back(std::move(task));
                overflowed_.fetch_add(1, std::memory_order_release);
                return;
            } else {
                pos = tail_.load(std::memory_order_relaxed);
            }
        }
    }

    // nullptr if empty.
    std::shared_ptr<Task> pop() {
        std::size_t pos = head_.load(std::memory_order_relaxed);
        for (;;) {
            Cell& c = cells_[pos & kMask];
            const std::size_t seq = c.seq.load(std::memory_order_acquire);
            const auto dif = static_cast<std::ptrdiff_t>(seq - (pos + 1));
            if (dif == 0) {
                if (head_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    std::shared_ptr<Task> t = std::move(c.task);
                    c.seq.store(pos + kCapacity, std::memory_order_release);
                    return t;
                }
            } else if (dif < 0) {
                break; // empty ring
            } else {
                pos = head_.load(std::memory_order_relaxed);
            }
        }
        if (overflowed_.load(std::memory_order_acquire) == 0) {
            return nullptr;
        }
        std::lock_guard lock(overflow_mutex_);
        if (overflow_.empty()) {
            return nullptr;
        }
        std::shared_ptr<Task> t = std::move(overflow_.front());
        overflow_.pop_front();
        overflowed_.fetch_sub(1, std::memory_order_relaxed);
        return t;
    }

  private:
    static constexpr std::size_t kMask = kCapacity - 1;
    struct Cell {
        std::atomic<std::size_t> seq;
        std::shared_ptr<Task> task;
    };
    std::vector<Cell> cells_;
    alignas(64) std::atomic<std::size_t> tail_{0};
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> overflowed_{0};
    std::mutex overflow_mutex_;
    std::deque<std::shared_ptr<Task>> overflow_;
};

} // namespace viproc
