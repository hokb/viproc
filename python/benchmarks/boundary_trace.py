"""Where the boundary between numpy.ndarray and viproc arrays costs scaling.

Runs ILNumerics' k-means program (suite case `kmeans_points`) with X as a
viproc array (arrays), as `python -m viproc` with an ndarray X (module) and
under `viproc.enable_hook()` with an ndarray X (hook): the whole program
against NumPy, then the per-point statement split into its ops, with the
main thread's time per op, the syncs it waited for and the kind of result.

    PYTHONPATH=build/python python3 python/benchmarks/boundary_trace.py [--d 1000]

docs/design/ndarray-hook.md ("Where ndarray state stops scaling") explains
the output.
"""

import argparse
import time

import numpy

import viproc


def kmeans(np, X, k, reps=1):
    """The suite's kmeans_points program."""
    n = np.size(X, 1)
    centers = X[:, np.linspace(0, n - 2, k, dtype=np.int64)].copy()
    classes = np.zeros_like(X, shape=(1, n))
    for _ in range(reps):
        for i in range(n):
            dist = np.sqrt(np.sum(np.square(X[:, i, np.newaxis] - centers), axis=0))
            classes[0, i] = dist.argmin()
        for i in range(k):
            found = np.nonzero(classes == i)[1]
            in_class = X[:, found]
            if in_class.size > 0:
                centers[:, i] = in_class.mean(axis=1)
    return classes, centers


def kind(r):
    mod = type(r).__module__.split(".")[0]
    if mod == "numpy" and getattr(r, "ndim", 0) and not (
            r.flags.c_contiguous or r.flags.f_contiguous):
        return "ndarray, strided"
    return {"numpy": "ndarray", "viproc": "viproc"}.get(mod, mod)


def setup(mode, X0):
    np = numpy if mode == "hook" else viproc
    X = viproc.asarray(X0) if mode == "arrays" else X0
    if mode != "arrays":
        viproc.enable_hook()
    return np, X


def whole(mode, X0, k):
    t = time.perf_counter()
    ref = kmeans(numpy, X0, k)
    tn = time.perf_counter() - t
    np, X = setup(mode, X0)
    try:
        s0 = dict(viproc.offload_stats())
        t = time.perf_counter()
        r = kmeans(np, X, k)
        viproc.wait_all()
        tv = time.perf_counter() - t
        s1 = viproc.offload_stats()
    finally:
        viproc.enable_hook(False)
    same = all(numpy.array_equal(numpy.asarray(a), b) for a, b in zip(r, ref))
    print(f"{mode:6s}: NumPy {tn * 1e3:6.1f} ms, viproc {tv * 1e3:6.1f} ms, {tn / tv:5.2f}x;"
          f" tasks {s1['tasks'] - s0['tasks']}, syncs {s1['syncs'] - s0['syncs']},"
          f" identical {same}")


def per_op(mode, X0, k, points):
    np, X = setup(mode, X0)
    try:
        n = np.size(X, 1)
        centers = X[:, np.linspace(0, n - 2, k, dtype=np.int64)].copy()
        classes = np.zeros_like(X, shape=(1, n))
        steps = [
            ("col  = X[:, i, np.newaxis]", "col", lambda e: X[:, e["i"], np.newaxis]),
            ("dif  = col - centers", "dif", lambda e: e["col"] - centers),
            ("sq   = np.square(dif)", "sq", lambda e: np.square(e["dif"])),
            ("s    = np.sum(sq, axis=0)", "s", lambda e: np.sum(e["sq"], axis=0)),
            ("dist = np.sqrt(s)", "dist", lambda e: np.sqrt(e["s"])),
            ("m    = dist.argmin()", "m", lambda e: e["dist"].argmin()),
            ("classes[0, i] = m", None, None),
        ]
        spent = {s: 0.0 for s, _, _ in steps}
        syncs = {s: 0 for s, _, _ in steps}
        kinds = {}
        for i in range(points):
            e = {"i": i}
            for s, name, f in steps:
                s0 = viproc.offload_stats()["syncs"]
                t = time.perf_counter()
                if f is None:
                    classes[0, i] = e["m"]
                else:
                    e[name] = f(e)
                spent[s] += time.perf_counter() - t
                syncs[s] += viproc.offload_stats()["syncs"] - s0
                if name is not None:
                    kinds[s] = kind(e[name])
        viproc.wait_all()
    finally:
        viproc.enable_hook(False)
    print(f"{mode}: X {kind(X)}, centers {kind(centers)}, classes {kind(classes)}; "
          f"main thread per point:")
    for s, _, _ in steps:
        print(f"   {s:28s} {spent[s] / points * 1e6:7.1f} us   syncs {syncs[s] / points:.1f}"
              f"   {kinds.get(s, '')}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--d", type=int, default=1000, help="dimensions = points (X is d x d)")
    p.add_argument("--k", type=int, default=50, help="clusters")
    p.add_argument("--points", type=int, default=300, help="points traced per op")
    args = p.parse_args()
    viproc.init()
    X0 = numpy.random.default_rng(0).uniform(0, 10, size=(args.d, args.d))
    print(f"k-means, X {args.d} x {args.d}, k = {args.k}")
    for mode in ("arrays", "module", "hook"):
        whole(mode, X0, args.k)
    for mode in ("arrays", "module", "hook"):
        per_op(mode, X0, args.k, args.points)


if __name__ == "__main__":
    main()
