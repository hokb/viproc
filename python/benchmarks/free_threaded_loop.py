"""array_pipelining.py's column loop with a thread pool on free-threaded CPython.

The manual alternative to viproc for the loop of array_pipelining.py: a
thread pool over column blocks (a Parallel.For). With the GIL it is slower
than the plain loop (array_pipelining.py, `numpy_threads`); on a
free-threaded CPython (3.13t and later, no GIL) the threads run NumPy's
calls in parallel. Plain NumPy only, no viproc: run it with a free-threaded
interpreter that has NumPy installed, e.g.

    uv python install 3.13t
    uv venv -p 3.13t /tmp/ft && uv pip install -p /tmp/ft numpy
    /tmp/ft/bin/python python/benchmarks/free_threaded_loop.py

Speedups are vs. the plain loop on the same interpreter. Only the loop with
independent iterations (`C[i] = ...`) can be split this way; `loop_dep`
(`C[:i+1] = ...`) gives wrong results unless its iterations keep their order.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy


def counter(n):
    return numpy.arange(1.0, n * n + 1.0).reshape(n, n).T


def loop(A, C):
    for i in range(A.shape[1]):
        C[i] = numpy.sum(numpy.abs(numpy.sin(A[:, i])), axis=0)


def parallel_for(pool, threads, A, C):
    n = A.shape[1]

    def block(k):
        for i in range(k * n // threads, (k + 1) * n // threads):
            C[i] = numpy.sum(numpy.abs(numpy.sin(A[:, i])), axis=0)

    list(pool.map(block, range(threads)))


def sample(fn, reps):
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1e3


def median(xs):
    return sorted(xs)[len(xs) // 2]


def main(threads=4):
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"Python {sys.version.split()[0]}, GIL enabled: {gil}, NumPy {numpy.__version__}, "
          f"{threads} threads")
    print("n,numpy_ms,threads_ms,speedup,identical")
    for n, reps in ((200, 200), (500, 50), (1000, 20), (1500, 10), (2000, 5), (3000, 3)):
        A = counter(n)
        C = numpy.zeros(n)
        loop(A, C)
        ref = C.copy()
        t0 = median([sample(lambda: loop(A, C), reps) for _ in range(5)])
        with ThreadPoolExecutor(threads) as pool:
            C2 = numpy.zeros(n)
            parallel_for(pool, threads, A, C2)
            t1 = median([sample(lambda: parallel_for(pool, threads, A, C2), reps)
                         for _ in range(5)])
        print(f"{n},{t0:.2f},{t1:.2f},{t0 / t1:.2f},{numpy.array_equal(C2, ref)}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
