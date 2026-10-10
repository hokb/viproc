#include "offload.hpp"

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
        offload = options_.explore_every > 0 && ++s.calls % options_.explore_every == 0;
    }
    if (!offload) {
        ++inline_count_;
        return {false, nullptr};
    }
    ++offload_count_;
    ++s.observed;
    return {true, &s};
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
    } else if (s.inline_mode && 4 * s.blocked <= s.observed) {
        s.inline_mode = false;
    }
    s.observed = 0;
    s.blocked = 0;
}

} // namespace viproc
