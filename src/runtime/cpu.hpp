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

} // namespace viproc
