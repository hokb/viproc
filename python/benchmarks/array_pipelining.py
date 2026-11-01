"""Array pipelining: ILNumerics' loop-parallelization benchmark on numpy2x.

The program of ILNumerics' "array pipelining" demo (getting-started part IV,
https://ilnumerics.net/accelerator-array-pipelining.html; code in
github.com/ILNumerics/decentralized-array-execution-artifacts2026,
Appendix "Part 2a/2b Loop Parallelization"), as plain NumPy code:

  loop        for i: C[i] = sum(abs(sin(A[:, i])), axis=0)       (2a)
              1,500 independent iterations of small array instructions
  loop_dep    for i: C[0:i+1] = sum(abs(sin(A[:, i])), axis=0)   (2b)
              the same, but every iteration overwrites the results of
              all earlier ones: only the original order gives the right C
  vectorized  C = sum(abs(sin(A)), axis=0)                       (2a)

A is ILNumerics' counter(1.0, 1.0, n, n): 1, 2, 3, ... in column-major
order, so A[:, i] is a contiguous column. The program runs unchanged on
NumPy and on numpy2x: its functions take the module as `np`, and the numpy2x
runs pass the numpy2x module, which is what `python -m numpy2x prog.py`
(`import numpy` gives numpy2x) and `import numpy2x as np` give a program.

Nothing in the loop is annotated or split by hand: numpy2x issues the
4 x n array instructions of the loop from the main thread and runs them on
its workers as soon as their inputs are ready (array instruction-level
parallelism). The loop's iterations are independent in `loop`; in
`loop_dep` the writes into C keep their order (C's alias group), the
sin/abs/sum work of different iterations still overlaps. ILNumerics
compares with a Parallel.For over the columns; `numpy_threads` is that for
NumPy (a thread pool over column blocks, which NumPy runs without the GIL
inside its loops), for `loop` only: for `loop_dep` it would be wrong.

Every result is checked bit for bit against NumPy.

    PYTHONPATH=build/python python3 python/benchmarks/array_pipelining.py
    ... --sizes 200:500,1500:10 --samples 7
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy

import numpy2x


def counter(np, n):
    """ILNumerics' counter<double>(1.0, 1.0, n, n): 1, 2, ... column by column."""
    return np.arange(1.0, n * n + 1.0).reshape(n, n).T


def loop(np, A, C):
    for i in range(A.shape[1]):
        C[i] = np.sum(np.abs(np.sin(A[:, i])), axis=0)


def loop_dep(np, A, C):
    for i in range(A.shape[1]):
        C[0:i + 1] = np.sum(np.abs(np.sin(A[:, i])), axis=0)


def vectorized(np, A, C):
    C[...] = np.sum(np.abs(np.sin(A)), axis=0)


PROGRAMS = {"loop": loop, "loop_dep": loop_dep, "vectorized": vectorized}


def parallel_for(pool, threads, A, C):
    """Parallel.For over the columns of `loop` (NumPy, column blocks)."""
    n = A.shape[1]

    def block(k):
        for i in range(k * n // threads, (k + 1) * n // threads):
            C[i] = numpy.sum(numpy.abs(numpy.sin(A[:, i])), axis=0)

    list(pool.map(block, range(threads)))


def sample(fn, reps, sync):
    """ms per run: `reps` runs and the final sync, as ILNumerics' run()."""
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t) / reps * 1e3


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def measure(n, reps, workers, samples, threads):
    rows = {}
    A = counter(numpy, n)
    expected = {}
    for name, prog in PROGRAMS.items():
        C = numpy.zeros(n)
        prog(numpy, A, C)
        expected[name] = C.copy()
        t = [sample(lambda: prog(numpy, A, C), reps, lambda: None) for _ in range(samples)]
        rows[name, "numpy"] = (median(t), True, 0)

    with ThreadPoolExecutor(threads) as pool:
        C = numpy.zeros(n)
        parallel_for(pool, threads, A, C)
        t = [sample(lambda: parallel_for(pool, threads, A, C), reps, lambda: None)
             for _ in range(samples)]
        rows["loop", "numpy_threads"] = (median(t), numpy.array_equal(C, expected["loop"]), 0)

    for w in workers:
        numpy2x.init()  # its fixed 2 workers
        vA = counter(numpy2x, n)
        for name, prog in PROGRAMS.items():
            vC = numpy2x.zeros(n)
            numpy2x.wait_all()
            f0 = numpy2x._ndarray.fallback_calls
            s0 = numpy2x.offload_stats()
            prog(numpy2x, vA, vC)  # warm-up and check
            same = numpy.array_equal(numpy.asarray(vC), expected[name])
            fallbacks = numpy2x._ndarray.fallback_calls - f0
            syncs = numpy2x.offload_stats()["syncs"] - s0["syncs"] - 1  # the check's own
            t = [sample(lambda: prog(numpy2x, vA, vC), reps, vC.wait) for _ in range(samples)]
            same = same and numpy.array_equal(numpy.asarray(vC), expected[name])
            rows[name, f"numpy2x w={w}"] = (median(t), same, fallbacks + max(syncs, 0))
        del vA, vC
        numpy2x.shutdown()
    return rows


def report(n, reps, rows, workers, threads):
    lines = [f"\nA: float64 [{n}, {n}], {reps} runs per sample, median of the samples "
             f"(ms per run; speedup vs. NumPy)", ""]
    cols = ["numpy", "numpy_threads"] + [f"numpy2x w={w}" for w in workers]
    head = ["program"] + [c if c != "numpy_threads" else f"NumPy {threads} threads"
                          for c in cols]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|---" + "|---:" * len(cols) + "|")
    t_np = {name: rows[name, "numpy"][0] for name in PROGRAMS}
    for name in PROGRAMS:
        cells = [f"`{name}`"]
        for c in cols:
            if (name, c) not in rows:
                cells.append("–")
                continue
            t, same, extra = rows[name, c]
            cell = f"{t:.2f}" if c == "numpy" else f"{t:.2f} ({t_np[name] / t:.2f}x)"
            if not same:
                cell += " **differs**"
            if extra:
                cell += f" [{extra} syncs/fallbacks]"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sizes", default="200:500,1500:10",
                   help="n:reps,... (A is n x n; reps runs per sample, as ILNumerics)")
    p.add_argument("--threads", type=int, default=None,
                   help="threads of the NumPy Parallel.For (default: cores)")
    p.add_argument("--samples", type=int, default=7)
    args = p.parse_args()
    cores = numpy2x.physical_cores()
    workers = [2]  # numpy2x's fixed worker count
    threads = args.threads or cores
    numpy2x.init()
    budget = numpy2x._numpy2x.direct_budget_ns() / 1e3
    numpy2x.shutdown()
    print(f"array pipelining, {os.cpu_count()} CPUs, {cores} physical cores, "
          f"NumPy {numpy.__version__}, numpy2x {numpy2x.__version__}, Python {sys.version.split()[0]}, "
          f"direct-path budget {budget:.1f} us")
    for spec in args.sizes.split(","):
        n, reps = (int(x) for x in spec.split(":"))
        rows = measure(n, reps, workers, args.samples, threads)
        print(report(n, reps, rows, workers, threads), flush=True)


if __name__ == "__main__":
    main()
