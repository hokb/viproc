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

#include <atomic>
#include <cstdint>
#include <unordered_map>
#include <utility>

namespace viproc {

// NumPy time per element of a kernel that many ops share (one per resolved
// loop), measured on the direct path: the dispatch policy sends ops of an
// expensive loop to the task path although they have fewer than
// `sync_below` elements (OffloadPolicy::promote). Main thread only, except
// `task_sample`.
struct DirectCost {
    float ns_per_element = 0;   // min of the last two samples; 0: not known yet
    float last = 0;             // the last sample
    std::uint32_t calls = 0;    // direct runs, for sampling
    std::uint32_t promoted = 0; // ops sent to the task path, for re-tests
    bool retest = false;        // the op being issued re-tests a promoted loop
    // ns per element of a re-test, measured by the task on its worker
    // (Task::time_kernel); 0: none arrived. Taken over by the next promote().
    std::atomic<float> task_sample{0};
};

struct OffloadOptions {
    // Tasks with fewer output elements always run inline (N_min).
    std::int64_t sync_below = 2048;
    // ... but not if the op's estimated NumPy time (DirectCost) is above
    // this. A promoted op makes the cheap ops that consume it tasks too, and
    // the main thread then pays the task path for all of them: `break_even.py`
    // chains of sin/cos break even at ~2 us per op, but with 2 us
    // `ind_cores_f32` (float32 sin of 1,000, ~2.2 us) fell from 0.6x to
    // 0.15x and with 5 us `expr8` (8 terms, 1,000 elements) from 1.0x to
    // 0.6x. 10 us promotes sin of 1,500 doubles (~15 us), not of 1,000.
    // These are times of a 4-vCPU VM. The budget stands for main-thread
    // costs, so the Python bridge sets it per machine, to 9 times the
    // measured time of a small direct op (`_scale_budget` in _ndarray.py,
    // vp_set_direct_budget_ns): with a fixed 10 us a 12700K (sin of 1,500
    // doubles 6.1 us) ran the whole loop of `array_pipelining.py` on the
    // main thread, no task at all.
    double direct_budget_ns = 10000;
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
    // `kRetest`-th such op is timed again (`c.retest`: its task times its
    // kernel on the worker), so a loop measured slow by accident (cold
    // memory, a preempted run) returns to the direct path. The re-test ran
    // directly on the main thread before: a sin of 1,500 doubles every 32nd
    // op stalled the issuing thread for ~15 us, 21% of its instructions in
    // `array_pipelining.py`'s loop. Adaptive mode only: deterministic mode
    // keeps rule 1 a fixed threshold.
    bool promote(DirectCost& c, std::int64_t elements) const {
        if (!options_.adaptive) {
            return false;
        }
        if (c.task_sample.load(std::memory_order_relaxed) > 0) {
            record_sample(c, c.task_sample.exchange(0, std::memory_order_relaxed));
        }
        if (c.ns_per_element * static_cast<double>(elements) <= options_.direct_budget_ns) {
            return false;
        }
        c.retest = ++c.promoted % kRetest == 0 && elements >= options_.time_from;
        return true;
    }
    // True if this direct run is to be timed (record()): the first two of a
    // loop, then every 16th.
    bool time_direct(DirectCost& c, std::int64_t elements) const {
        if (!options_.adaptive || elements < options_.time_from) {
            return false;
        }
        const std::uint32_t k = c.calls++;
        return k < 2 || k % 16 == 0;
    }
    static void record(DirectCost& c, double ns, std::int64_t elements);
    // The same for a sample in ns per element.
    static void record_sample(DirectCost& c, float ns_per_element);

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
