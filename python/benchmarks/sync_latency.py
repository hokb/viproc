"""Latency of "issue an op, then read its result right away" vs. NumPy.

This is the pattern the dispatch policy targets (see "Dispatch decision" in
CLAUDE.md): there is nothing to overlap, so handing the op to a worker only
adds latency. viproc should stay close to NumPy here.

Usage: PYTHONPATH=build/python python3 python/benchmarks/sync_latency.py
"""

import argparse
import time

import numpy as np

import viproc


def per_op_us(fn, n):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e6


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sizes", type=str, default="8,1000,10000,100000")
    p.add_argument("--ops", type=int, default=2000)
    p.add_argument("--deterministic", action="store_true", help="disable the adaptive policy")
    args = p.parse_args()
    viproc.init(adaptive=False if args.deterministic else None)
    print(f"{'elements':>10} {'numpy':>10} {'viproc':>10}   (us per op, result read right away)")
    for size in (int(s) for s in args.sizes.split(",")):
        a, b = np.random.rand(size), np.random.rand(size)
        va, vb = viproc.asarray(a), viproc.asarray(b)
        t_np = per_op_us(lambda: (a * b)[0], args.ops)
        t_vp = per_op_us(lambda: (va * vb).wait(), args.ops)
        print(f"{size:>10} {t_np:>10.2f} {t_vp:>10.2f}")
    print("dispatch:", viproc.offload_stats())


if __name__ == "__main__":
    main()
