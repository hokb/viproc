// Dispatch decision for tasks whose inputs are all ready: hand them to a
// worker, or run them right away on the main thread (should_offload).
//
// Hybrid policy (after the ILNumerics Accelerator's "scalar shortcut" and
// "adaptive fork suppression", see CLAUDE.md):
//  1. Below `sync_below` elements a task always runs inline: deterministic
//     and cheap, for ops whose hand-off costs more than the work. Unless
//     its loop is expensive: an op whose NumPy time (measured per loop on
//     the direct path) exceeds `direct_budget_ns` is a task like a large
//     one (sin of 1,500 doubles takes ~15 us, add ~0.6 us). Adaptive mode
//     only: the measurement is a time, not deterministic.
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

// NumPy time per element of a kernel that many ops share (one per resolved
// loop), measured on the direct path: the dispatch policy sends ops of an
// expensive loop to the task path although they have fewer than
// `sync_below` elements (OffloadPolicy::promote). Main thread only.
struct DirectCost {
    float ns_per_element = 0;   // min of the last two samples; 0: not known yet
    float last = 0;             // the last sample
    std::uint32_t calls = 0;    // direct runs, for sampling
    std::uint32_t promoted = 0; // ops sent to the task path, for re-tests
    bool retest = false;        // this direct run re-tests a promoted loop
};

struct OffloadOptions {
    // Tasks with fewer output elements always run inline (N_min).
    std::int64_t sync_below = 2048;
    // ... but not if the op's estimated NumPy time (DirectCost) is above
    // this. A promoted op makes the cheap ops that consume it tasks too:
    // `break_even.py` chains of sin/cos break even at ~2 us per op, but
    // `ind_cores_f32` (float32 sin of 1,000, ~2.2 us, then * 0.5 + 0.4)
    // drops from 0.6x to 0.15x as tasks. 5 us leaves it direct and still
    // promotes sin of 1,000 doubles (~8 us).
    double direct_budget_ns = 5000;
    // Direct runs of at least this many elements are timed (the first two
    // of a loop, then every 16th): below it the clock reads would cost more
    // than the decision gains.
    std::int64_t time_from = 256;
    // Per-site adaptive decision above `sync_below`, and expensive loops
    // below it as tasks; off = always offload above, always inline below.
    bool adaptive = true;
    // A sync point counts as "result needed right after issuing" if at most
    // this many ops were issued in between (measured from the main-thread
    // decision that started the work), whether the task had finished or not.
    std::uint64_t window = 2;
    // Async observations per decision, and the exploration period in inline
    // mode: it starts at explore_every and doubles (up to explore_max) each
    // time the exploration confirms inline mode.
    int sample = 8;
    int explore_every = 16;
    int explore_max = 1024;
};

// Statistics of one (issue site, size class). Main thread only.
struct SiteStats {
    int observed = 0; // async issues since the last decision
    int blocked = 0;  // ... of which the main thread waited on right away
    bool inline_mode = false;
    int calls = 0;  // calls in inline mode (for exploration)
    int period = 0; // current exploration period (0: explore_every)
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

    // True if an op of `elements` (< sync_below) on a kernel with this cost
    // estimate takes the task path anyway: its loop is expensive. Every
    // `kRetest`-th such op runs directly and is timed again, so a loop
    // measured slow by accident (cold memory, a preempted run) returns to
    // the direct path. Adaptive mode only: deterministic mode keeps rule 1
    // a fixed threshold.
    bool promote(DirectCost& c, std::int64_t elements) const {
        if (!options_.adaptive ||
            c.ns_per_element * static_cast<double>(elements) <= options_.direct_budget_ns) {
            return false;
        }
        if (++c.promoted % kRetest == 0 && elements >= options_.time_from) {
            c.retest = true;
            return false;
        }
        return true;
    }
    // True if this direct run is to be timed (record()): the first two of a
    // loop, then every 16th, and re-tests.
    bool time_direct(DirectCost& c, std::int64_t elements) const {
        if (!options_.adaptive || elements < options_.time_from) {
            return false;
        }
        const std::uint32_t k = c.calls++;
        return k < 2 || k % 16 == 0 || c.retest;
    }
    static void record(DirectCost& c, double ns, std::int64_t elements);

    // The main thread is about to block on work it started with `stats`,
    // `distance` ops ago.
    void on_blocked(SiteStats* stats, std::uint64_t distance);

    const OffloadOptions& options() const { return options_; }
    std::uint64_t inline_count() const { return inline_count_; }
    std::uint64_t offload_count() const { return offload_count_; }

  private:
    static constexpr std::uint32_t kRetest = 32;
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
