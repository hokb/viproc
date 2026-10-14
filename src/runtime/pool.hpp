// Memory pools for the fixed-size objects created once per op (Task, Buffer,
// ArrayBlock): no malloc/free per op once the pools are warm.
//
// Each thread keeps a private free list. A thread that allocates from a pool
// returns blocks to its own list (no atomics: the main thread both creates and,
// for inline ops, releases most objects). Other threads (workers dropping the
// last reference to a task or buffer) push blocks onto a shared lock-free
// stack, which an allocating thread takes over as a whole when its own list
// is empty. Taking the whole stack (exchange), never single nodes, rules out
// ABA. Pools only grow, up to the peak number of live objects; their memory
// is kept for the process lifetime. Any thread may allocate and release.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <new>

namespace viproc {

template <std::size_t Size, std::size_t Align> class FixedPool {
    struct Node {
        Node* next;
    };
    static constexpr std::size_t kSize = std::max(Size, sizeof(Node));
    static constexpr std::align_val_t kAlign{std::max(Align, alignof(Node))};

  public:
    static void* allocate() {
        Local& l = local();
        l.allocator = true;
        if (l.free == nullptr) {
            l.free = shared().exchange(nullptr, std::memory_order_acquire);
        }
        if (Node* n = l.free) {
            l.free = n->next;
            return n;
        }
        return ::operator new(kSize, kAlign);
    }

    static void deallocate(void* p) {
        Node* n = static_cast<Node*>(p);
        Local& l = local();
        if (l.allocator) {
            n->next = l.free;
            l.free = n;
            return;
        }
        Node* head = shared().load(std::memory_order_relaxed);
        do {
            n->next = head;
        } while (!shared().compare_exchange_weak(head, n, std::memory_order_release,
                                                 std::memory_order_relaxed));
    }

  private:
    // Trivially destructible on purpose: no thread_local destructor that
    // could run after the last pooled object of that thread is released. A
    // thread that allocated and exits leaves its free blocks behind.
    struct Local {
        Node* free = nullptr;
        bool allocator = false; // this thread allocates from the pool
    };
    static Local& local() {
        thread_local Local l;
        return l;
    }
    static std::atomic<Node*>& shared() {
        static std::atomic<Node*> s{nullptr};
        return s;
    }
};

// std allocator over FixedPool, for std::allocate_shared (control block and
// object in one pooled block) and similar single-object allocations.
template <class T> struct PoolAllocator {
    using value_type = T;
    PoolAllocator() = default;
    template <class U> PoolAllocator(const PoolAllocator<U>&) {}

    T* allocate(std::size_t n) {
        if (n != 1) {
            return std::allocator<T>().allocate(n);
        }
        return static_cast<T*>(FixedPool<sizeof(T), alignof(T)>::allocate());
    }
    void deallocate(T* p, std::size_t n) {
        if (n != 1) {
            std::allocator<T>().deallocate(p, n);
            return;
        }
        FixedPool<sizeof(T), alignof(T)>::deallocate(p);
    }
    template <class U> bool operator==(const PoolAllocator<U>&) const { return true; }
};

// std::make_shared with pooled memory.
template <class T, class... Args> std::shared_ptr<T> make_pooled(Args&&... args) {
    return std::allocate_shared<T>(PoolAllocator<T>(), std::forward<Args>(args)...);
}

} // namespace viproc
