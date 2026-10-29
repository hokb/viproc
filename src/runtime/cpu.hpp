#pragma once

#include <cstddef>

namespace viproc {

// Number of physical cores this process may run on (SMT siblings counted
// once; affinity respected where the OS reports it). The default worker
// count: a second hardware thread on a core shares its execution units and
// caches, and NumPy's SIMD loops already keep one thread per core busy.
// Falls back to std::thread::hardware_concurrency() if the topology is
// unknown. Thread-safe; reads the OS topology on every call.
std::size_t physical_core_count();

// The runtime's default worker count: one per physical core but one, at
// least 1. The main thread issues the program's ops and needs a core of its
// own: with a worker on every core it shared one with a worker, and its
// issue rate fell with every worker added (fine-grained loops are bound by
// it, docs/optimizations.md). Thread-safe.
std::size_t default_worker_count();

// Time of a fixed piece of pointer-chasing scalar code on the calling
// thread (the kind of code the issue path is), in ns: the best of a few
// runs of ~8 us each. A measure of the calling core's single-thread
// speed, to scale per-op time budgets that stand for main-thread costs
// (OffloadOptions::direct_budget_ns) to the machine. Thread-safe; 0.2-1 ms.
double issue_probe_ns();

// issue_probe_ns() on the machine the budgets were calibrated on (4-vCPU
// Xeon VM at 2.8 GHz, GCC Release; docs/benchmarks.md).
inline constexpr double kIssueProbeReferenceNs = 7840;

// issue_probe_ns() / kIssueProbeReferenceNs, measured once per process and
// clamped to [0.2, 5]: < 1 on a faster core. Thread-safe.
double issue_speed_scale();

} // namespace viproc
