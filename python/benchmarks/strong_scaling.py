"""Strong-scaling benchmark: one fixed array program, NumPy vs. viproc at
1, 2, 4, ... worker threads.

The program has `chains` independent dependency chains of `length` ops each,
interleaved the way a sequential program would issue them, and combines the
chains at the end. Only array ILP can make it faster: every single op runs as
one NumPy loop call, exactly as in NumPy.

Usage: PYTHONPATH=build/python python3 python/benchmarks/strong_scaling.py
"""

import argparse
import os
import time

import numpy as np

import viproc


def program(xs, length):
    """Sequential NumPy code; runs unchanged on NumPy or viproc arrays."""
    xs = list(xs)
    for _ in range(length):
        for i, x in enumerate(xs):
            xs[i] = np.sin(x) * 0.5 + np.cos(x) * 0.25
    total = xs[0]
    for x in xs[1:]:
        total = total + x
    return total


def best_of(fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--size", type=int, default=1_000_000, help="elements per array")
    p.add_argument("--chains", type=int, default=8)
    p.add_argument("--length", type=int, default=10)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--workers", type=str, default="1,2,4")
    p.add_argument("--sync-below", type=int, default=None,
                   help="ops with ready inputs and fewer elements run inline (default: viproc's)")
    p.add_argument("--deterministic", action="store_true", help="disable the adaptive policy")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    data = [rng.standard_normal(args.size) for _ in range(args.chains)]
    ops = args.chains * args.length * 5 + args.chains - 1

    expected = program(data, args.length)
    t_np = best_of(lambda: program(data, args.length), args.repeat)
    print(f"{ops} ops on {args.size:,} float64 elements, {args.chains} independent chains, "
          f"{os.cpu_count()} CPUs")
    print(f"{'numpy':>12}: {t_np * 1e3:8.1f} ms")

    for w in (int(s) for s in args.workers.split(",")):
        viproc.init(workers=w, sync_below=args.sync_below,
                    adaptive=False if args.deterministic else None)
        vdata = [viproc.asarray(d) for d in data]
        viproc.wait_all()

        def run():
            program(vdata, args.length).wait()

        result = program(vdata, args.length)
        assert np.array_equal(np.asarray(result), expected), "result differs from NumPy"
        t = best_of(run, args.repeat)
        print(f"{'viproc w=' + str(w):>12}: {t * 1e3:8.1f} ms   speedup vs numpy {t_np / t:5.2f}x")
    viproc.shutdown()


if __name__ == "__main__":
    main()
