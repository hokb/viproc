#include "kernels/blas_threads.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>

namespace viproc::numpy {

namespace {

std::atomic<BlasSetThreads> g_set{nullptr};
std::atomic<const std::atomic<std::size_t>*> g_active{nullptr};
int g_alone = 1;
bool g_dynamic = false;
std::atomic<int> g_current{0};   // count set last (0: unknown / no control)
std::atomic<int> g_in_flight{0}; // BLAS calls running now
std::atomic<std::size_t> g_changes{0};
std::mutex g_set_mutex; // serializes changes (OpenBLAS's setter is not reentrant)
// Time [ns, steady clock] a BLAS call last found other asynchronous work. A call
// alone raises the count only kHoldNs after that: a stream of concurrent calls
// (a loop issuing independent products) often has a first call that starts
// before the next one is issued, and raising for it made OpenBLAS's threads
// compete with the workers (8 products of 300 x 300 per round: 5.6 ms vs.
// 4.9 ms with one thread throughout, 4 vCPUs).
std::atomic<std::int64_t> g_busy_ns{INT64_MIN / 2};
constexpr std::int64_t kHoldNs = 5'000'000;

std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void set_threads(BlasSetThreads set, int n) {
    std::lock_guard<std::mutex> lock(g_set_mutex);
    if (g_current.load(std::memory_order_relaxed) == n) {
        return;
    }
    set(n);
    g_current.store(n, std::memory_order_relaxed);
    g_changes.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

void configure_blas_threads(BlasSetThreads set, int alone_threads, bool dynamic,
                            const std::atomic<std::size_t>* active) {
    std::lock_guard<std::mutex> lock(g_set_mutex);
    g_alone = alone_threads < 1 ? 1 : alone_threads;
    g_dynamic = dynamic && set != nullptr && active != nullptr && g_alone > 1;
    g_active.store(active, std::memory_order_release);
    g_set.store(set, std::memory_order_release);
    g_current.store(0, std::memory_order_relaxed);
    if (set != nullptr) {
        // dynamic: start as plain NumPy (alone); fixed: one thread
        const int n = g_dynamic ? g_alone : 1;
        set(n);
        g_current.store(n, std::memory_order_relaxed);
    }
}

BlasCallScope::BlasCallScope() {
    const BlasSetThreads set = g_set.load(std::memory_order_acquire);
    if (set == nullptr || !g_dynamic) {
        return; // no control, or fixed at one thread since configure
    }
    counted_ = true;
    const int in_flight = g_in_flight.fetch_add(1, std::memory_order_acq_rel) + 1;
    const auto* active = g_active.load(std::memory_order_acquire);
    // this call's own task counts as active (a direct-path call does not)
    const bool alone =
        in_flight == 1 && active != nullptr && active->load(std::memory_order_acquire) <= 1;
    const std::int64_t now = now_ns();
    if (!alone) {
        g_busy_ns.store(now, std::memory_order_relaxed);
    }
    const bool quiet = now - g_busy_ns.load(std::memory_order_relaxed) > kHoldNs;
    const int want = alone && quiet ? g_alone : 1;
    const int cur = g_current.load(std::memory_order_relaxed);
    if (want == cur) {
        return;
    }
    if (want < cur || in_flight == 1) {
        set_threads(set, want); // lower any time, raise only as the only call in flight
    }
}

BlasCallScope::~BlasCallScope() {
    if (counted_) {
        g_in_flight.fetch_sub(1, std::memory_order_acq_rel);
    }
}

int blas_threads_current() { return g_current.load(std::memory_order_relaxed); }
std::size_t blas_threads_changes() { return g_changes.load(std::memory_order_relaxed); }

} // namespace viproc::numpy
