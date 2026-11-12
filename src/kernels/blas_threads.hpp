// Thread count of the BLAS library NumPy's loops call (OpenBLAS), decided per
// BLAS call by the task that makes it (docs/design/goals.md, "BLAS threads").
//
// OpenBLAS's count is process-wide (no per-thread setting in NumPy's wheels).
// Dynamic mode: a BLAS call that runs alone (no other BLAS call in flight, no
// other active task of the runtime, and none found for 5 ms) gets the count
// found at init (OpenBLAS's own parallelism, as plain NumPy); as soon as there
// is more asynchronous work, calls run on one thread and viproc's workers
// parallelize instead.
// Fixed mode: one thread for every call (results of NumPy with
// OPENBLAS_NUM_THREADS=1, reproducible). The count is lowered at any time (a
// running call keeps the split it started with) and raised only by a call
// that is the only one in flight.
#pragma once

#include <atomic>
#include <cstddef>

namespace viproc::numpy {

using BlasSetThreads = void (*)(int);

// Installs the setter (OpenBLAS's set_num_threads; nullptr: no control) and
// the policy. `active` is the runtime's count of active tasks (read only; it
// must outlive the configuration: reset with nullptr before the runtime
// goes). Main thread, while no kernel runs a BLAS call.
void configure_blas_threads(BlasSetThreads set, int alone_threads, bool dynamic,
                            const std::atomic<std::size_t>* active);

// Around every loop call that may call BLAS (matmul, gufuncs): sets the count
// for this call. Any thread; no lock unless the count changes.
class BlasCallScope {
  public:
    BlasCallScope();
    ~BlasCallScope();
    BlasCallScope(const BlasCallScope&) = delete;
    BlasCallScope& operator=(const BlasCallScope&) = delete;

  private:
    bool counted_ = false;
};

// Diagnostics: the count viproc set last (0: no control) and how often it
// changed it.
int blas_threads_current();
std::size_t blas_threads_changes();

} // namespace viproc::numpy
