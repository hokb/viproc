#include "element_pool.hpp"

#include <bit>
#include <cctype>
#include <cstdlib>
#include <new>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace viproc {

namespace {

constexpr std::size_t kPageAlignFrom = std::size_t{64} << 10; // 64 KiB
constexpr std::size_t kHugeFrom = std::size_t{2} << 20;       // 2 MiB

std::align_val_t alignment_for(std::size_t capacity) {
    if (capacity >= kHugeFrom) {
        return std::align_val_t{kHugeFrom}; // lets huge pages back all of it
    }
    return std::align_val_t{capacity >= kPageAlignFrom ? std::size_t{4096} : std::size_t{64}};
}

[[maybe_unused]] std::size_t physical_memory() {
#if defined(_WIN32)
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof(m);
    if (GlobalMemoryStatusEx(&m)) {
        return static_cast<std::size_t>(m.ullTotalPhys);
    }
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0) {
        return static_cast<std::size_t>(pages) * static_cast<std::size_t>(page);
    }
#endif
    return std::size_t{2} << 30;
}

#if defined(_WIN32)
// Windows counterpart of transparent huge pages: large pages need the
// "Lock pages in memory" privilege (SeLockMemoryPrivilege). Enabled once if
// the account holds it; 0 otherwise.
std::size_t large_page_size() {
    static const std::size_t size = [] {
        HANDLE token = nullptr;
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) {
            return std::size_t{0};
        }
        TOKEN_PRIVILEGES tp{};
        tp.PrivilegeCount = 1;
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        // AdjustTokenPrivileges succeeds without the privilege, too; only
        // GetLastError() tells.
        const bool held =
            LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid) &&
            AdjustTokenPrivileges(token, FALSE, &tp, 0, nullptr, nullptr) &&
            GetLastError() == ERROR_SUCCESS;
        CloseHandle(token);
        return held ? static_cast<std::size_t>(GetLargePageMinimum()) : std::size_t{0};
    }();
    return size;
}
#endif

// VIPROC_POOL_LIMIT: bytes, optionally with a K/M/G suffix; 0 disables caching.
std::size_t default_limit() {
#if defined(VIPROC_NO_ELEMENT_POOL)
    return 0; // sanitizer builds: every release goes back to the system
#else
    if (const char* env = std::getenv("VIPROC_POOL_LIMIT")) {
        char* end = nullptr;
        unsigned long long v = std::strtoull(env, &end, 10);
        switch (end != nullptr ? std::toupper(static_cast<unsigned char>(*end)) : 0) {
        case 'G':
            v <<= 10;
            [[fallthrough]];
        case 'M':
            v <<= 10;
            [[fallthrough]];
        case 'K':
            v <<= 10;
            break;
        default:
            break;
        }
        return static_cast<std::size_t>(v);
    }
    return physical_memory() / 2;
#endif
}

} // namespace

struct ElementPool::Local {
    Node* free[kBins];
    bool allocator; // this thread allocates (keeps released blocks itself)
};

ElementPool::Local& ElementPool::local() {
    // Trivially destructible on purpose, as in FixedPool: no destructor that
    // could run after the thread's last release. Only the issuing thread
    // allocates, and it lives as long as the process.
    thread_local Local l{};
    return l;
}

ElementPool& ElementPool::instance() {
    static ElementPool* pool = new ElementPool(); // never destroyed: blocks may
    return *pool;                                 // be released at exit
}

ElementPool::ElementPool() : limit_(default_limit()) {}

std::size_t ElementPool::capacity_for(std::size_t nbytes) {
    if (nbytes <= 128) {
        return nbytes <= 16 ? 16 : (nbytes + 15) & ~std::size_t{15};
    }
    // 8 bins per power of two: round up to a multiple of 2^(g-3), where
    // 2^g <= nbytes - 1 < 2^(g+1).
    const int g = std::bit_width(nbytes - 1) - 1;
    const std::size_t step = std::size_t{1} << (g - 3);
    return ((nbytes - 1) / step + 1) * step;
}

int ElementPool::bin_of(std::size_t capacity) {
    if (capacity <= 128) {
        return static_cast<int>(capacity / 16) - 1; // 16, 32, ..., 128
    }
    const int g = std::bit_width(capacity - 1) - 1;
    const int j = static_cast<int>((capacity - 1) >> (g - 3)) & 7;
    return 8 + (g - 7) * 8 + j;
}

void* ElementPool::system_allocate(std::size_t capacity) {
#if defined(_WIN32)
    if (capacity >= kHugeFrom) {
        // Page-aligned from the system, as on Linux; large pages when allowed
        // and the block is a whole number of them (always from 16 MiB on).
        const std::size_t large = large_page_size();
        void* p = nullptr;
        if (large != 0 && capacity % large == 0) {
            p = VirtualAlloc(nullptr, capacity, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                             PAGE_READWRITE);
        }
        if (p == nullptr) {
            p = VirtualAlloc(nullptr, capacity, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        }
        if (p == nullptr) {
            throw std::bad_alloc();
        }
        return p;
    }
#endif
    void* p = ::operator new(capacity, alignment_for(capacity));
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (capacity >= kHugeFrom) {
        madvise(p, capacity, MADV_HUGEPAGE); // a hint; failure is harmless
    }
#endif
    return p;
}

void ElementPool::system_release(void* block, std::size_t capacity) {
#if defined(_WIN32)
    if (capacity >= kHugeFrom) {
        VirtualFree(block, 0, MEM_RELEASE);
        return;
    }
#endif
    ::operator delete(block, alignment_for(capacity));
}

void* ElementPool::allocate(std::size_t nbytes, std::size_t& capacity) {
    capacity = capacity_for(nbytes);
    const int b = bin_of(capacity);
    Local& l = local();
    l.allocator = true;
    if (l.free[b] == nullptr) {
        l.free[b] = remote_[b].exchange(nullptr, std::memory_order_acquire);
    }
    if (Node* n = l.free[b]) {
        l.free[b] = n->next;
        cached_.fetch_sub(static_cast<std::int64_t>(capacity), std::memory_order_relaxed);
        hits_.fetch_add(1, std::memory_order_relaxed);
        return n;
    }
    misses_.fetch_add(1, std::memory_order_relaxed);
    try {
        return system_allocate(capacity);
    } catch (const std::bad_alloc&) {
        trim(); // cached blocks of other sizes may make room
        return system_allocate(capacity);
    }
}

void ElementPool::release(void* block, std::size_t capacity) {
    const auto c = static_cast<std::int64_t>(capacity);
    // Check, then add: concurrent releases may overshoot the bound slightly.
    if (cached_.load(std::memory_order_relaxed) + c > static_cast<std::int64_t>(limit())) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        system_release(block, capacity);
        return;
    }
    cached_.fetch_add(c, std::memory_order_relaxed);
    Node* n = static_cast<Node*>(block);
    const int b = bin_of(capacity);
    Local& l = local();
    if (l.allocator) {
        n->next = l.free[b];
        l.free[b] = n;
        return;
    }
    Node* head = remote_[b].load(std::memory_order_relaxed);
    do {
        n->next = head;
    } while (!remote_[b].compare_exchange_weak(head, n, std::memory_order_release,
                                               std::memory_order_relaxed));
}

void ElementPool::trim() {
    Local& l = local();
    for (int b = 0; b < kBins; ++b) {
        Node* lists[2] = {l.free[b], remote_[b].exchange(nullptr, std::memory_order_acquire)};
        l.free[b] = nullptr;
        for (Node* n : lists) {
            while (n != nullptr) {
                Node* next = n->next;
                // Every block of a bin has the bin's capacity; recover it from
                // the bin index.
                std::size_t capacity = 0;
                if (b < 8) {
                    capacity = static_cast<std::size_t>(b + 1) * 16;
                } else {
                    const int g = (b - 8) / 8 + 7;
                    const int j = (b - 8) % 8;
                    capacity = static_cast<std::size_t>(8 + j + 1) << (g - 3);
                }
                cached_.fetch_sub(static_cast<std::int64_t>(capacity), std::memory_order_relaxed);
                system_release(n, capacity);
                n = next;
            }
        }
    }
}

void ElementPool::set_limit(std::size_t bytes) {
    limit_.store(bytes, std::memory_order_relaxed);
    if (cached_.load(std::memory_order_relaxed) > static_cast<std::int64_t>(bytes)) {
        trim();
    }
}

ElementPool::Stats ElementPool::stats() const {
    const std::int64_t c = cached_.load(std::memory_order_relaxed);
    return {hits_.load(std::memory_order_relaxed), misses_.load(std::memory_order_relaxed),
            static_cast<std::uint64_t>(c > 0 ? c : 0), dropped_.load(std::memory_order_relaxed)};
}

} // namespace viproc
