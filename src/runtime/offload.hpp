// Dispatch decision for tasks whose inputs are all ready: hand them to a
// worker, or run them right away on the main thread (should_offload).
//
// Hybrid policy (after the ILNumerics Accelerator's "scalar shortcut" and
// "adaptive fork suppression", see CLAUDE.md):
//  1. Below `sync_below` elements a task always runs inline: deterministic
//     and cheap, for ops whose hand-off costs more than the work.
//  2. Above it, per (issue site, size class) statistics: if the main thread
//     usually blocks on the result right after issuing (it needs the value
//     anyway, so there is nothing to overlap), the site runs inline. Unlike
//     the Accelerator the decision is reversible: in inline mode every
//     `explore_every`-th call still goes async to re-test.
//
// Main thread only: all state here is touched by the issuing thread alone.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <utility>

namespace viproc {

struct OffloadOptions {
    // Tasks with fewer output elements always run inline (N_min).
    std::int64_t sync_below = 2048;
    // Per-site adaptive decision above `sync_below`; off = always offload.
    bool adaptive = true;
    // A wait counts as "blocked right after issuing" if at most this many ops
    // were issued in between (measured from the main-thread decision that
    // started the work).
    std::uint64_t window = 8;
    // Async observations per decision, and the exploration period in inline
    // mode.
    int sample = 8;
    int explore_every = 16;
};

// Statistics of one (issue site, size class). Main thread only.
struct SiteStats {
    int observed = 0; // async issues since the last decision
    int blocked = 0;  // ... of which the main thread waited on right away
    bool inline_mode = false;
    int calls = 0; // calls in inline mode (for exploration)
};

class OffloadPolicy {
  public:
    explicit OffloadPolicy(OffloadOptions options) : options_(options) {}

    struct Decision {
        bool offload;
        SiteStats* stats; // to blame if the main thread blocks; may be null
    };

    // For a task whose inputs are all ready, issued at `site` (interned)
    // producing `elements` output elements.
    Decision decide(const char* site, std::int64_t elements);

    // The main thread is about to block on work it started with `stats`,
    // `distance` ops ago.
    void on_blocked(SiteStats* stats, std::uint64_t distance);

    const OffloadOptions& options() const { return options_; }
    std::uint64_t inline_count() const { return inline_count_; }
    std::uint64_t offload_count() const { return offload_count_; }

  private:
    struct KeyHash {
        std::size_t operator()(const std::pair<const char*, int>& k) const {
            return std::hash<const void*>()(k.first) * 31u + static_cast<std::size_t>(k.second);
        }
    };
    void maybe_switch(SiteStats& s);

    OffloadOptions options_;
    std::unordered_map<std::pair<const char*, int>, SiteStats, KeyHash> sites_;
    std::uint64_t inline_count_ = 0;
    std::uint64_t offload_count_ = 0;
};

} // namespace viproc
