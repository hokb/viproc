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
        float per = 0;           // ns per element, min of the last two samples; 0: not known
        float last = 0;          // the last sample
        float elements = 0;      // elements of the last sample
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
    // What a small op costs the main thread more as a task of a pending input
    // than directly (a promoted op's follower, ChainScore); measured by the
    // bridge (vp_set_task_costs).
    double chain_ns = 0;
    // Ops after an issue whose readers count as its followers.
    std::uint64_t chain_window = 32;
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

// The ops that follow one issue site's op in the program's data flow (a
// chain: the ops reading its result, their results, ...), observed for a
// window of ops after each issue (OffloadPolicy::credit). An op of the site
// that goes to the task path takes the small ones among them along (their
// input is pending then): it saves the main thread their kernel time and
// costs it their task issue. Main thread only.
struct ChainScore {
    const char* site = nullptr;
    float kernel_ns = 0; // followers' NumPy time per issue (decaying average)
    float followers = 0; // followers per issue (decaying average)
    float acc_ns = 0;    // ... of the current issue
    std::uint16_t acc_followers = 0;
    std::uint16_t issues = 0; // issues seen, saturating
    // A/B check of a promotion the estimate proposes: the main thread's time
    // between two issues at the site (a loop's iteration), kPeriods of them
    // directly, then as tasks; the faster variant holds for `next` issues
    // (64, doubling up to 4096 while the decision holds).
    static constexpr int kPeriods = 8;
    static constexpr int kWarmTask = 16; // periods as tasks before counting
    std::int64_t last_ns = 0;            // the previous issue during a check; 0: none
    float periods[2][kPeriods] = {};     // [direct, task]
    std::uint8_t count = 0;              // periods of the current variant counted
    std::uint8_t seen = 0;               // ... and not counted (warm-up)
    std::uint8_t phase = 0;              // 0: check direct, 1: check task, 2: decided
    bool task = false;                   // the decision
    std::uint32_t calls = 0;             // issues since the decision
    std::uint32_t next = 0;              // ... until the next check
};

// Statistics of one (issue site, size class). Main thread only.
struct SiteStats {
    int observed = 0; // async issues since the last decision
    int blocked = 0;  // ... of which the main thread waited on right away
    bool inline_mode = false;
    int calls = 0;        // calls in inline mode (for exploration)
    int period = 0;       // current exploration period (0: explore_every)
    bool explore = false; // prefers_inline() chose this call to explore
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
    // For an op promoted below `sync_below` (promote()): true if its site
    // runs inline (rule 2), so the op takes the direct path after all: run
    // inline, a task costs the main thread more than the direct op. Else,
    // and on an exploration call, decide() offloads it.
    bool prefers_inline(const char* site);

    // True if an op of `elements` (< sync_below) on a kernel with this cost
    // estimate takes the task path anyway: its loop is expensive. Every
    // `kRetest`-th such op is timed again (`c.retest`: its task times its
    // kernel on the worker), so a loop measured slow by accident (cold
    // memory, a preempted run) returns to the direct path. The re-test ran
    // directly on the main thread before: a sin of 1,500 doubles every 32nd
    // op stalled the issuing thread for ~15 us, 21% of its instructions in
    // `array_pipelining.py`'s loop. Adaptive mode only: deterministic mode
    // keeps rule 1 a fixed threshold.
    //
    // The op's own time is not all the main thread saves: the small ops
    // following it in the data flow become tasks too. Per issue site the
    // policy keeps their NumPy time and count (ChainScore, credit()) and
    // promotes if own + followers' time > direct_budget_ns (the task's
    // surcharge) + followers x chain_ns (a follower's surcharge as a task of
    // a pending input). A loop over 200 x 200 columns, `sum(abs(sin(A[:,
    // i])))`: the sin alone (~0.8 us on a 12700K) costs less than a task's
    // issue (~1 us), with abs and sum (~1 us) more. Without observations the
    // site counts 2 followers of no time.
    bool promote(DirectCost& c, std::int64_t elements, const char* site) {
        if (!options_.adaptive || elements < options_.time_from) {
            return false;
        }
        if (c.task_sample.load(std::memory_order_relaxed) != 0) {
            take_task_sample(c);
        }
        ChainScore& s = chain_of(site, false);
        const double own = estimate(c, elements);
        const bool known = s.site == site && s.issues >= 2;
        const double gain = own + (known ? s.kernel_ns : 0.0);
        const double cost =
            options_.direct_budget_ns + (known ? s.followers : 2.0) * options_.chain_ns;
        if (gain <= cost) {
            return false;
        }
        // The estimate's costs are those of `a * b`: a site's own ops may
        // cost more as tasks. A close promotion is checked by measuring both.
        if (known && gain < 4 * cost &&
            !(s.phase == 2 && ++s.calls < s.next ? s.task : check(s, gain))) {
            return false;
        }
        c.retest = ++c.promoted % kRetest == 0;
        return true;
    }
    // A new issue at `site` (with no chain among its inputs): the
    // observations of its previous issue are folded into its score.
    void start_chain(const char* site) {
        ChainScore& s = chain_of(site, true);
        if (s.issues <= 1) {
            s.kernel_ns = s.acc_ns;
            s.followers = s.acc_followers;
        } else {
            s.kernel_ns = 0.75f * s.kernel_ns + 0.25f * s.acc_ns;
            s.followers = 0.75f * s.followers + 0.25f * static_cast<float>(s.acc_followers);
        }
        s.issues = s.issues < 0xffff ? s.issues + 1 : s.issues;
        s.acc_ns = 0;
        s.acc_followers = 0;
    }
    // A small op of estimated NumPy time `ns` follows the op issued at `site`.
    void credit(const char* site, double ns) {
        ChainScore& s = chain_of(site, false);
        if (s.site == site) {
            s.acc_ns += static_cast<float>(ns);
            s.acc_followers = s.acc_followers < 0xffff ? s.acc_followers + 1 : s.acc_followers;
        }
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
    void set_workers(std::size_t n) { workers_ = n; }
    std::uint64_t inline_count() const { return inline_count_; }
    std::uint64_t offload_count() const { return offload_count_; }

  private:
    static constexpr std::uint32_t kRetest = 32;
    static void take_task_sample(DirectCost& c);
    // One step of the A/B check: true if this issue runs as a task.
    bool check(ChainScore& s, double kernel_ns);
    static constexpr int kChains = 64; // direct-mapped by site
    ChainScore& chain_of(const char* site, bool claim) {
        ChainScore& s = chains_[(reinterpret_cast<std::uintptr_t>(site) >> 4) % kChains];
        if (claim && s.site != site) {
            s = ChainScore{};
            s.site = site;
        }
        return s;
    }
    struct KeyHash {
        std::size_t operator()(const std::pair<const char*, int>& k) const {
            return std::hash<const void*>()(k.first) * 31u + static_cast<std::size_t>(k.second);
        }
    };
    void maybe_switch(SiteStats& s);

    OffloadOptions options_;
    std::size_t workers_ = 1;
    ChainScore chains_[kChains];
    std::unordered_map<std::pair<const char*, int>, SiteStats, KeyHash> sites_;
    std::uint64_t inline_count_ = 0;
    std::uint64_t offload_count_ = 0;
};

} // namespace viproc
