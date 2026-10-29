#include "offload.hpp"

#include <algorithm>
#include <bit>

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

void OffloadPolicy::record(DirectCost& c, double ns, std::int64_t elements) {
    // The smaller of two samples: one preempted or cold run (page faults,
    // caches) must not make a cheap loop look expensive. The first sample
    // alone decides nothing (min with 0).
    const auto per = static_cast<float>(ns / static_cast<double>(elements));
    c.ns_per_element = std::min(c.last, per);
    c.last = per;
    c.retest = false;
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
