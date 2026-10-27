"""Does NumPy compute a strided ndarray view as it computes its dense copy?

Under the ndarray hook (docs/design/ndarray-hook.md) a strided plain-NumPy
operand stays in NumPy at optimization level O1: viproc could only import
a dense copy (the user may write the view's memory later), and NumPy's
results can depend on the strides of the call. This script measures where
they do on this machine and NumPy build: element-wise ufuncs (one and two
operands, a scalar), reductions and matmul, each on strided views against
the same values copied densely. Plain NumPy only; run with
``python3 python/benchmarks/strided_identity.py``.
"""

import platform
import warnings

import numpy as np

DTYPES = (np.float32, np.float64, np.complex64, np.complex128, np.int32, np.int64)


def values(rng, shape, dt):
    x = rng.standard_normal(shape) * 3
    if np.issubdtype(dt, np.complexfloating):
        x = x + 1j * rng.standard_normal(shape)
    return x.astype(dt)


def same(f, args):
    try:
        a = f(*args)
        b = f(*[np.array(v) if isinstance(v, np.ndarray) else v for v in args])
    except Exception:
        return None
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(2)
    total = 0
    diff = {}

    def record(kind, dt, ok):
        nonlocal total
        if ok is None:
            return
        total += 1
        if not ok:
            diff.setdefault(kind, set()).add(np.dtype(dt).name)

    for name in sorted(dir(np)):
        u = getattr(np, name)
        if not isinstance(u, np.ufunc) or u.signature or u.nout != 1:
            continue
        for dt in DTYPES:
            for rows in (40000, 300, 17, 3):
                a, b = values(rng, (rows, 16), dt), values(rng, (rows, 16), dt)
                for x, y in ((a[:, 1], b[:, 2]), (a[:, 1], b[:, 2].copy()),
                             (a[::2, ::3], b[::2, ::3]), (a.T[::-1], b.T[::-1]),
                             (a[:, 3], dt(1.7))):
                        record("ufunc " + name, dt, same(u, [x, y][:u.nin]))
    for f in ("sum", "prod", "mean", "var", "std", "max", "cumsum"):
        for dt in DTYPES:
            for rows in (300, 40000, 100003):
                a = values(rng, (rows, 16), dt)
                for v, axis in ((a[:, 1], None), (a[::3], 0), (a[::3], 1), (a.T[::2], None),
                                (a[:, ::2], 0)):
                    record(f, dt, same(lambda x: getattr(np, f)(x, axis=axis), [v]))
    for dt in (np.float32, np.float64, np.complex64, np.complex128):
        for n in (64, 300, 700):
            a, b = values(rng, (n, 2 * n), dt), values(rng, (2 * n, n), dt)
            for x, y in ((a[:, ::2], b[::2]), (a[:, :n], b[:n]), (a.T[::2].T, b)):
                record("matmul", dt, same(np.matmul, [x, y]))

    print(f"NumPy {np.__version__}, {platform.machine()}, {platform.processor() or '?'}")
    print(f"{total} cases, {sum(len(v) for v in diff.values())} (function, dtype) pairs differ")
    for kind, dts in sorted(diff.items()):
        print(f"  {kind}: {', '.join(sorted(dts))}")


if __name__ == "__main__":
    main()
