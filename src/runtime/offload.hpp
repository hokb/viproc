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

// NumPy time of a kernel that many ops share (one per resolved loop), per
// log2 size class of the op, measured on the direct path: the dispatch policy
// sends ops of an expensive loop to the task path although they have fewer
// than `sync_below` elements (OffloadPolicy::promote). A size without samples
// of its own is estimated from its neighbours: interpolated between the
// nearest measured classes below and above, extrapolated per element from the
// nearest one below (a kernel's time per element falls with the size, so this
// overestimates; above the largest sample only). Main thread only, except
// `task_sample`.
struct DirectCost {
    // Size classes: bit width of the element count (64..127 -> 7, ...).
    static constexpr int kClasses = 16;
    struct Class {
        float per = 0;      // ns per element, min of the last two samples; 0: not known
        float last = 0;     // the last sample
        float elements = 0; // elements of the last sample
        std::uint32_t calls = 0; // direct runs of this class, for sampling
    };
    Class classes[kClasses];
    std::uint32_t promoted = 0; // ops sent to the task path, for re-tests
    bool retest = false;        // the op being issued re-tests a promoted loop
    // A re-test, measured by the task on its worker (Task::time_kernel): the
    // float bits of its ns per element, shifted left by 32, | its elements;
    // 0: none arrived. Taken over by the next promote().
    std::atomic<std::uint64_t> task_sample{0};
};

struct OffloadOptions {
    // Tasks with fewer output elements always run inline (N_min).
    std::int64_t sync_below = 2048;
    // ... but not if the op's estimated NumPy time (DirectCost) is above
    // this: what the task path costs the main thread more than the direct
    // path, for the op and for the cheap ops consuming it (a promoted op makes
    // them tasks too). The Python bridge measures it per machine at start
    // (`_scale_budget` in _ndarray.py, vp_set_direct_budget_ns): a direct op,
    // a task of ready inputs and a task of a pending input, the budget is the
    // surcharge of the task plus two consumers' surcharges. The default is a
    // 4-vCPU VM's time.
    double direct_budget_ns = 10000;
    // Direct runs of at least this many elements are timed (per size class
    // the first two, then every 16th): below it the clock reads would cost
    // more than the decision gains, and such ops are never promoted. With
    // 256 a loop over 200 x 200 columns (`array_pipelining.py`) ran every sin
    // directly, at 1.3x of NumPy on a 12700K; as tasks 1.7x.
    std::int64_t time_from = 64;
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
        if (!options_.adaptive || elements < options_.time_from) {
            return false;
        }
        if (c.task_sample.load(std::memory_order_relaxed) != 0) {
            take_task_sample(c);
        }
        if (estimate(c, elements) <= options_.direct_budget_ns) {
            return false;
        }
        c.retest = ++c.promoted % kRetest == 0;
        return true;
    }
    // True if this direct run is to be timed (record()): per size class the
    // first two, then every 16th.
    bool time_direct(DirectCost& c, std::int64_t elements) const {
        if (!options_.adaptive || elements < options_.time_from) {
            return false;
        }
        const std::uint32_t k = c.classes[cost_class(elements)].calls++;
        return k < 2 || k % 16 == 0;
    }
    // Estimated NumPy time (ns) of `elements` on this kernel; 0: unknown.
    static double estimate(const DirectCost& c, std::int64_t elements);
    static void record(DirectCost& c, double ns, std::int64_t elements);
    // The same for a sample in ns per element.
    static void record_sample(DirectCost& c, float ns_per_element, std::int64_t elements);
    static int cost_class(std::int64_t elements);

    // The main thread is about to block on work it started with `stats`,
    // `distance` ops ago.
    void on_blocked(SiteStats* stats, std::uint64_t distance);

    const OffloadOptions& options() const { return options_; }
    std::uint64_t inline_count() const { return inline_count_; }
    std::uint64_t offload_count() const { return offload_count_; }

  private:
    static constexpr std::uint32_t kRetest = 32;
    static void take_task_sample(DirectCost& c);
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
