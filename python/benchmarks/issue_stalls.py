"""Where does the main thread stall while issuing? Per-op issue times of a
dependent chain on large arrays (24 MB results), with the element pool's
misses: an op that waits on a system allocation or on the CPU shows up as
one slow issue.

    python python/benchmarks/issue_stalls.py [n] [runs]
"""

import os
import sys
import time

import numpy as np

import viproc


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(f"cpus {os.cpu_count()}, workers {viproc.workers()}, n {n}")
    x = viproc.asarray(np.random.default_rng(0).standard_normal(n))
    ops = (lambda v: v * v, lambda v: v + 1.0, np.sqrt)
    for run in range(runs):
        viproc.wait_all()
        misses = viproc.pool_stats()["misses"]
        times, y = [], x
        for _ in range(20):
            for f in ops:
                t = time.perf_counter()
                y = f(y)
                times.append(1e3 * (time.perf_counter() - t))
        t = time.perf_counter()
        y.wait()
        wait = 1e3 * (time.perf_counter() - t)
        slow = [f"{k}:{t:.1f}" for k, t in enumerate(times) if t > 0.5]
        print(f"run {run}: issue {sum(times):.1f} ms, wait {wait:.1f} ms, "
              f"misses {viproc.pool_stats()['misses'] - misses}, ops > 0.5 ms (index:ms) {slow}")


if __name__ == "__main__":
    main()
