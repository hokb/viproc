#include "offload.hpp"

#include <algorithm>
#include <bit>
#include <chrono>

namespace viproc {

namespace {

// log2 size class: 1024..2047 elements -> 11, ...
int size_class(std::int64_t elements) {
    return elements <= 0 ? 0
                         : static_cast<int>(std::bit_width(static_cast<std::uint64_t>(elements)));
}

} // namespace

OffloadPolicy::Decision OffloadPolicy::decide(const char* site, std::int64_t elements) {
    if (elements < options_.sync_below) {
        ++inline_count_;
        return {false, nullptr};
    }
    if (!options_.adaptive) {
        ++offload_count_;
        return {true, nullptr};
    }
    SiteStats& s = sites_[{site, size_class(elements)}];
    // Decide on the previous issues first: their waits have been seen by now.
    maybe_switch(s);
    bool offload = true;
    if (s.inline_mode) {
        // Re-test now and then whether the site has overlap after all.
        offload = s.period > 0 && ++s.calls % s.period == 0;
    }
    if (!offload) {
        ++inline_count_;
        return {false, nullptr};
    }
    ++offload_count_;
    ++s.observed;
    return {true, &s};
}

int OffloadPolicy::cost_class(std::int64_t elements) {
    return std::min(size_class(elements), DirectCost::kClasses - 1);
}

double OffloadPolicy::estimate(const DirectCost& c, std::int64_t elements) {
    const int k = cost_class(elements);
    const double n = static_cast<double>(elements);
    if (c.classes[k].per > 0) {
        return c.classes[k].per * n;
    }
    int lo = k - 1;
    while (lo >= 0 && c.classes[lo].per <= 0) {
        --lo;
    }
    int hi = k + 1;
    while (hi < DirectCost::kClasses && c.classes[hi].per <= 0) {
        ++hi;
    }
    if (lo >= 0 && hi < DirectCost::kClasses) {
        const DirectCost::Class& a = c.classes[lo];
        const DirectCost::Class& b = c.classes[hi];
        const double ta = a.per * a.elements;
        const double tb = b.per * b.elements;
        return ta + (tb - ta) * (n - a.elements) / (b.elements - a.elements);
    }
    if (lo >= 0) {
        return c.classes[lo].per * n;
    }
    if (hi < DirectCost::kClasses) {
        return c.classes[hi].per * n; // underestimates: never promotes by accident
    }
    return 0;
}

void OffloadPolicy::record(DirectCost& c, double ns, std::int64_t elements) {
    record_sample(c, static_cast<float>(ns / static_cast<double>(elements)), elements);
}

void OffloadPolicy::record_sample(DirectCost& c, float per, std::int64_t elements) {
    // The smaller of two samples: one preempted or cold run (page faults,
    // caches) must not make a cheap loop look expensive. The first sample
    // alone decides nothing (min with 0).
    DirectCost::Class& s = c.classes[cost_class(elements)];
    s.per = std::min(s.last, per);
    s.last = per;
    s.elements = static_cast<float>(elements);
}

void OffloadPolicy::take_task_sample(DirectCost& c) {
    const std::uint64_t v = c.task_sample.exchange(0, std::memory_order_relaxed);
    record_sample(c, std::bit_cast<float>(static_cast<std::uint32_t>(v >> 32)),
                  static_cast<std::int64_t>(v & 0xffffffffu));
}

bool OffloadPolicy::check(ChainScore& s, double kernel_ns) {
    if (s.phase == 2) { // a new check
        s.phase = 0;
        s.count = 0;
        s.seen = 0;
        s.last_ns = 0;
    }
    const std::int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count();
    // The first periods of a variant are not counted: the switch, and as
    // tasks the warm-up (sleeping workers woken by the main thread, ~5 us
    // each, before the pool streams). With only the switch skipped the 200 x
    // 200 loop measured tasks as slow as the direct path on a 12700K, where
    // it runs 1.3x faster as tasks.
    const int warm = s.phase == 1 ? ChainScore::kWarmTask : 1;
    if (s.last_ns != 0 && ++s.seen > warm) {
        s.periods[s.phase][s.count++] = static_cast<float>(now - s.last_ns);
    }
    s.last_ns = now;
    if (s.count < ChainScore::kPeriods) {
        return s.phase == 1;
    }
    s.seen = 0;
    s.count = 0;
    s.last_ns = 0;
    if (++s.phase < 2) {
        return true; // the task variant next
    }
    auto median = [](float* v) {
        std::nth_element(v, v + ChainScore::kPeriods / 2, v + ChainScore::kPeriods);
        return v[ChainScore::kPeriods / 2];
    };
    const double direct = median(s.periods[0]);
    // As tasks the workers must keep up too: at most the direct time
    // (kernels included) spread over all of them. The chain's NumPy time
    // alone left out the workers' task overhead: with 1 worker on the VM the
    // 200 x 200 loop chose tasks and ran 20% slower.
    (void)kernel_ns;
    const double task =
        std::max<double>(median(s.periods[1]), direct / static_cast<double>(workers_));
    const bool t = task < 0.95 * direct;
    s.next = t == s.task && s.next != 0 ? std::min<std::uint32_t>(2 * s.next, 4096) : 64;
    s.task = t;
    s.calls = 0;
    return t;
}

void OffloadPolicy::on_blocked(SiteStats* stats, std::uint64_t distance) {
    if (stats == nullptr || distance > options_.window) {
        return;
    }
    ++stats->blocked;
}

void OffloadPolicy::maybe_switch(SiteStats& s) {
    if (s.observed < options_.sample) {
        return;
    }
    // Hysteresis: inline if >= 3/4 of async issues blocked right away, back to
    // async if <= 1/4 did.
    if (!s.inline_mode && 4 * s.blocked >= 3 * s.observed) {
        s.inline_mode = true;
        s.calls = 0;
        s.period = options_.explore_every;
    } else if (s.inline_mode && 4 * s.blocked <= s.observed) {
        s.inline_mode = false;
    } else if (s.inline_mode) {
        // Confirmed: explore less often.
        s.period = std::min(2 * s.period, options_.explore_max);
    }
    s.observed = 0;
    s.blocked = 0;
}

} // namespace viproc
