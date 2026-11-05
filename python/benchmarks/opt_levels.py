"""BLAS-heavy programs at optimization levels O1 and O2 vs. NumPy.

O2 (docs/design/goals.md) runs every BLAS call on one thread and lets
viproc's workers overlap independent calls; O1 keeps NumPy's BLAS threading
(bit-identical) and serializes large LAPACK calls. Per case and size: NumPy's
time (its default BLAS threads), viproc O1 and O2 as speedups vs. NumPy, and
O2's largest relative deviation from NumPy's result.

    PYTHONPATH=build/python python3 python/benchmarks/opt_levels.py [--quick]
"""

import argparse
import time

import numpy as np

import viproc


def products(ms, p):
    """`p` independent chains of products, 3 steps each."""
    out = []
    for i in range(p):
        x = ms[i]
        for _ in range(3):
            x = (x @ ms[(i + 1) % p]) * (1.0 / x.shape[0])
        out.append(x)
    return out


def inverses(ms, p):
    return [np.linalg.inv(m) for m in ms[:p]]


def solves(ms, p):
    return [np.linalg.solve(m, m[:, :4]) for m in ms[:p]]


def one_product(ms, p):
    """A single large product: nothing to overlap (O2's worst case)."""
    return [ms[0] @ ms[1]]


CASES = {"products": products, "inv": inverses, "solve": solves, "one_product": one_product}


def run(fn, ms, p, reps):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        r = fn(ms, p)
        r = [np.asarray(x) for x in r]  # sync
        best = min(best, time.perf_counter() - t)
    return best, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--p", type=int, default=8, help="independent ops per case")
    args = ap.parse_args()
    sizes = [64, 200, 500] if args.quick else [64, 128, 200, 300, 500, 1000]
    reps = 3 if args.quick else 7
    rng = np.random.default_rng(0)
    print(f"cores {viproc.physical_cores()}, p = {args.p}")
    print(f"{'case':12} {'n':>5} {'numpy ms':>9} {'O1':>6} {'O2':>6} {'O2 rel.dev':>10}")
    for name, fn in CASES.items():
        for n in sizes:
            ms_np = [rng.random((n, n)) + n * np.eye(n) for _ in range(args.p)]
            with viproc.disabled():
                t_np, ref = run(fn, ms_np, args.p, reps)
            res = {}
            for level in (1, 2):
                viproc.init(level=level)
                ms = [viproc.asarray(m) for m in ms_np]
                t, r = run(fn, ms, args.p, reps)
                res[level] = (t, r)
            viproc.init()
            dev = max(float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))
                      for a, b in zip(res[2][1], ref))
            print(f"{name:12} {n:5d} {t_np * 1e3:9.2f} {t_np / res[1][0]:6.2f} "
                  f"{t_np / res[2][0]:6.2f} {dev:10.1e}")


if __name__ == "__main__":
    main()
