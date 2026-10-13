"""Cases of the viproc benchmark suite (suite.py).

Every case is plain, sequential NumPy code. It runs unchanged on NumPy
arrays and on viproc arrays (NumPy dispatches through the array protocols),
so the same function measures both.

A case declares:

- `width`: its parallel width P, the number of independent instruction
  streams in the program (independent chains, tiles, batches, clusters).
  An ideal executor reaches a speedup of min(cores, P) on data large enough
  to amortize the per-op overhead. `"cores"` makes the program as wide as
  the machine.
- `variants`: (size class, layout) pairs to run, or (size class, layout,
  dtype) for the cases of category `dtypes`; `make` then gets the dtype.
- `make(rng, shape_of, width)`: the input arrays (NumPy, base arrays plus a
  layout view, see `Layout`); `run(data, reps)`: the program, repeated
  `reps` times along its dependency chains (the suite picks `reps` so one
  NumPy run takes a fixed time).

Size classes: `scalar` (0-d), `tiny` (8 elements), `small` (1,000),
`medium` (100,000: cache sized), `large` (scaled with the machine's RAM per
core, see suite.py). Layouts: `vec` (1-d contiguous), `C`/`F` (2-d row /
column major), `strided` (2-d view of every other column), `col`/`row`
((n, 1) / (1, n) vectors, for broadcasting).
"""

from dataclasses import dataclass, field

import numpy as np

SIZE_ELEMENTS = {"scalar": 1, "tiny": 8, "small": 1_000, "medium": 100_000}


@dataclass
class Layout:
    """How an input array is laid out: a base shape and a view of it."""

    name: str

    def base_shape(self, n):
        if self.name == "scalar":
            return ()
        if self.name == "vec":
            return (n,)
        if self.name == "col":
            return (n, 1)
        if self.name == "row":
            return (1, n)
        rows, cols = matrix_shape(n)
        if self.name == "C":
            return (rows, cols)
        if self.name == "F":
            return (cols, rows)  # transposed below: a column-major view
        if self.name == "strided":
            return (rows, 2 * cols)  # every other column below
        raise ValueError(self.name)

    def view(self, a):
        if self.name == "F":
            return a.T
        if self.name == "strided":
            return a[:, ::2]
        return a


def matrix_shape(n):
    """(rows, cols) with rows * cols close to n."""
    if n >= 100_000:
        cols = 1000
    elif n >= 1000:
        cols = 40
    else:
        cols = max(1, min(4, n))
    return max(1, n // cols), cols


@dataclass
class Case:
    name: str
    category: str
    description: str
    width: object  # int, or "cores"
    variants: list
    make: object
    run: object
    dtype: object = np.float64
    tags: list = field(default_factory=list)


CASES = []


def case(name, category, description, width, variants, dtype=np.float64):
    def register(fn):
        make, run = fn()
        CASES.append(Case(name, category, description, width, variants, make, run, dtype))
        return fn

    return register


def uniform(rng, shape, dtype=np.float64, lo=0.5, hi=1.5):
    return rng.uniform(lo, hi, size=shape).astype(dtype)


# --- element-wise chains ----------------------------------------------------


@case("chains_wide", "chains",
      "one independent chain of sin/cos updates per core (strong_scaling.py)",
      "cores", [("small", "vec"), ("medium", "vec"), ("medium", "C"), ("medium", "F"),
                ("medium", "strided"), ("large", "vec"), ("large", "C")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape) for _ in range(width)]

    def run(xs, reps):
        xs = list(xs)
        for _ in range(reps):
            for i, x in enumerate(xs):
                xs[i] = np.sin(x) * 0.5 + np.cos(x) * 0.25
        total = xs[0]
        for x in xs[1:]:
            total = total + x
        return total

    return make, run


@case("chains_8", "chains", "8 independent chains: fixed width, as strong_scaling.py",
      8, [("medium", "vec"), ("large", "vec")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape) for _ in range(width)]

    def run(xs, reps):
        xs = list(xs)
        for _ in range(reps):
            for i, x in enumerate(xs):
                xs[i] = np.sqrt(x * x + 1.0) - x * 0.5
        return xs

    return make, run


@case("chain_serial", "chains", "a single dependency chain: no parallelism, pure overhead",
      1, [("scalar", "scalar"), ("tiny", "vec"), ("small", "vec"), ("medium", "vec"),
          ("large", "vec")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape)]

    def run(xs, reps):
        x = xs[0]
        for _ in range(reps):
            x = np.sqrt(x * 1.5 + 0.25) - 0.1
        return x

    return make, run


@case("expression_tree", "chains",
      "one wide expression: 8 independent subterms combined at the end", 8,
      [("small", "vec"), ("medium", "vec"), ("large", "vec")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape) for _ in range(2)]

    def run(xy, reps):
        x, y = xy
        acc = x
        for _ in range(reps):
            terms = [np.exp(-x * 0.5), np.sin(y), np.cos(x * y), np.sqrt(x + y),
                     np.log1p(x), np.tanh(y), x * x * 0.3, np.arctan(y)]
            s = terms[0]
            for t in terms[1:]:
                s = s + t
            acc = acc * 0.5 + s * 0.01
        return acc

    return make, run


@case("inplace_updates", "chains", "per-core chains of in-place operators and out=",
      "cores", [("small", "vec"), ("medium", "vec"), ("large", "vec")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape) for _ in range(width)] + [uniform(rng, shape)]

    def run(data, reps):
        *xs, y = data
        xs = [x.copy() for x in xs]  # mutated below
        for _ in range(reps):
            for x in xs:
                x *= 0.99
                x += y
                np.sqrt(x, out=x)
        return xs

    return make, run


@case("dtype_mix", "chains", "per-core chains in float32, int64 and complex128",
      "cores", [("medium", "vec"), ("large", "vec")])
def _():
    def make(rng, shape, width):
        out = []
        for i in range(width):
            kind = i % 3
            if kind == 0:
                out.append(uniform(rng, shape, np.float32))
            elif kind == 1:
                out.append(rng.integers(0, 1000, size=shape, dtype=np.int64))
            else:
                out.append(uniform(rng, shape) + 1j * uniform(rng, shape))
        return out

    def run(xs, reps):
        xs = list(xs)
        for _ in range(reps):
            for i, x in enumerate(xs):
                if x.dtype == np.int64:
                    xs[i] = (x * 3 + 7) % 1009
                elif x.dtype == np.float32:
                    xs[i] = np.sin(x) * np.float32(0.5) + np.float32(0.4)
                else:
                    xs[i] = x * (0.6 + 0.2j) + 0.1
        return xs

    return make, run


@case("broadcast_outer", "chains",
      "outer combinations of column and row vectors ((n,1) op (1,n)), per core",
      "cores", [("small", "col"), ("medium", "col")])
def _():
    def make(rng, shape, width):
        n = shape[0]
        side = max(2, int(round(n ** 0.5)))
        return [(uniform(rng, (side, 1)), uniform(rng, (1, side))) for _ in range(width)]

    def run(pairs, reps):
        out = []
        for col, row in pairs:
            m = col * row
            for _ in range(reps):
                m = m * 0.5 + col - row
            out.append(m)
        return out

    return make, run


# --- indexing ---------------------------------------------------------------


@case("stencil_1d", "indexing", "explicit heat equation on one rod per core (slices)",
      "cores", [("small", "vec"), ("medium", "vec"), ("large", "vec")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape) for _ in range(width)]

    def run(us, reps):
        us = [u.copy() for u in us]
        for _ in range(reps):
            for u in us:
                u[1:-1] = u[1:-1] + 0.25 * (u[:-2] - 2.0 * u[1:-1] + u[2:])
        return us

    return make, run


@case("stencil_2d", "indexing", "5-point Jacobi stencil on one grid per core (2-d slices)",
      "cores", [("medium", "C"), ("medium", "F"), ("large", "C")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape) for _ in range(width)]

    def run(gs, reps):
        gs = list(gs)
        for _ in range(reps):
            for i, g in enumerate(gs):
                inner = 0.25 * (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
                g = g.copy()
                g[1:-1, 1:-1] = inner
                gs[i] = g
        return gs

    return make, run


@case("strided_views", "indexing",
      "per-core chains on strided, reversed and newaxis views", "cores",
      [("medium", "vec"), ("large", "vec")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape) for _ in range(width)]

    def run(xs, reps):
        out = []
        for x in xs:
            a = x[::2]
            b = x[::-1][::2]
            for _ in range(reps):
                a = a * 0.5 + b[::-1] * 0.25 + 0.1
            out.append(a[:, None] * 2.0)
        return out

    return make, run


@case("gather", "indexing", "advanced indexing: gather with index arrays, per core",
      "cores", [("small", "vec"), ("medium", "vec"), ("large", "vec")])
def _():
    def make(rng, shape, width):
        n = shape[0]
        xs = [uniform(rng, shape) for _ in range(width)]
        idx = rng.integers(0, n, size=n)
        return xs + [idx]

    def run(data, reps):
        *xs, idx = data
        out = []
        for x in xs:
            for _ in range(reps):
                x = x[idx] * 0.5 + 0.25
            out.append(x)
        return out

    return make, run


@case("boolean_mask", "indexing", "boolean masks: selection and np.where, per core",
      "cores", [("small", "vec"), ("medium", "vec"), ("large", "vec")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape) for _ in range(width)]

    def run(xs, reps):
        out = []
        for x in xs:
            for _ in range(reps):
                x = np.where(x > 1.0, x - 0.5, x + 0.4)
            out.append((x, x[x > 1.0]))
        return out

    return make, run


@case("scatter", "indexing", "advanced assignment x[idx] = values, per core",
      "cores", [("medium", "vec"), ("large", "vec")])
def _():
    def make(rng, shape, width):
        n = shape[0]
        return [uniform(rng, shape) for _ in range(width)] + [rng.permutation(n)[: n // 4]]

    def run(data, reps):
        *xs, idx = data
        xs = [x.copy() for x in xs]
        for _ in range(reps):
            for x in xs:
                x[idx] = x[idx] * 0.5 + 1.0
        return xs

    return make, run


# --- reductions -------------------------------------------------------------


@case("reduce_full", "reductions", "sum / max / mean of a computed array, per core",
      "cores", [("small", "vec"), ("medium", "vec"), ("large", "vec")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape) for _ in range(width)]

    def run(xs, reps):
        out = []
        for x in xs:
            acc = 0.0
            for _ in range(reps):
                y = np.sin(x) * 2.0
                acc = acc + y.sum() + y.max() - y.mean()
            out.append(acc)
        return out

    return make, run


@case("reduce_axis", "reductions", "axis reductions (sum, std) on 2-d data, per core",
      "cores", [("medium", "C"), ("medium", "F"), ("large", "C")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape) for _ in range(width)]

    def run(xs, reps):
        out = []
        for x in xs:
            for _ in range(reps):
                r = x.sum(axis=0) + x.std(axis=0)
                c = x.sum(axis=1)
            out.append((r, c))
        return out

    return make, run


@case("softmax_rows", "reductions", "row-wise softmax (max, exp, sum, divide), per core",
      "cores", [("small", "C"), ("medium", "C"), ("large", "C")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape, lo=-3, hi=3) for _ in range(width)]

    def run(xs, reps):
        out = []
        for x in xs:
            for _ in range(reps):
                e = np.exp(x - x.max(axis=1, keepdims=True))
                p = e / e.sum(axis=1, keepdims=True)
            out.append(p)
        return out

    return make, run


# --- algorithms -------------------------------------------------------------


@case("kmeans_points", "algorithms",
      "k-means as in the ILNumerics Accelerator benchmark: a loop over the points, "
      "per point the distances to all centers and argmin; per cluster nonzero and mean",
      "cores", [("small", "C"), ("medium", "C"), ("large", "C")])
def _():
    # The NumPy program of github.com/hokb/decentralized-array-execution-
    # artifacts2026 (Appendix, Part 3 KMeans), with a fixed iteration count
    # (reps) instead of the convergence test. Every point's distance chain
    # is independent of the others: the parallel width is the number of
    # points; the per-cluster updates need the labels (nonzero syncs).
    def make(rng, shape, width):
        n = int(np.prod(shape))
        d = max(2, int(round(n ** 0.5)))  # d dimensions x d points
        return [rng.uniform(0, 10, size=(d, d)), np.array(min(50, max(2, d // 10)))]

    def run(data, reps):
        X, k = data
        k = int(k)
        n = np.size(X, 1)
        centers = X[:, np.linspace(0, n - 2, k, dtype=np.int64)].copy()
        # zeros_like: an array of X's kind (np.zeros would make a NumPy
        # array, and each classes[0, i] = ... would sync on the argmin)
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

    return make, run


@case("kmeans", "algorithms",
      "k-means, k = 8, 2-d points: per-cluster distances, labels, centroid update", 8,
      [("small", "vec"), ("medium", "vec"), ("large", "vec")])
def _():
    k = 8

    def make(rng, shape, width):
        return [uniform(rng, shape, lo=0, hi=10), uniform(rng, shape, lo=0, hi=10)]

    def run(pts, reps):
        x, y = pts
        cx = [float(v) for v in np.linspace(1, 9, k)]
        cy = [float(v) for v in np.linspace(9, 1, k)]
        labels = None
        for _ in range(reps):
            dist = [(x - cx[j]) ** 2 + (y - cy[j]) ** 2 for j in range(k)]
            best = dist[0]
            labels = np.zeros(x.shape, dtype=np.int64)
            for j in range(1, k):
                closer = dist[j] < best
                best = np.minimum(best, dist[j])
                labels = np.where(closer, j, labels)
            for j in range(k):
                m = labels == j
                cnt = int(m.sum())
                if cnt:
                    cx[j] = float((x * m).sum()) / cnt
                    cy[j] = float((y * m).sum()) / cnt
        return labels, np.array(cx), np.array(cy)

    return make, run


@case("black_scholes", "algorithms", "Black-Scholes call/put prices, one batch per core",
      "cores", [("small", "vec"), ("medium", "vec"), ("large", "vec")])
def _():
    def make(rng, shape, width):
        return [(uniform(rng, shape, lo=50, hi=150), uniform(rng, shape, lo=50, hi=150),
                 uniform(rng, shape, lo=0.1, hi=2.0)) for _ in range(width)]

    def cdf(d):
        # Abramowitz-Stegun 26.2.17, element-wise only
        t = 1.0 / (1.0 + 0.2316419 * np.abs(d))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
                    + t * (-1.821255978 + t * 1.330274429))))
        w = 1.0 - 0.3989422804014327 * np.exp(-0.5 * d * d) * poly
        return np.where(d < 0, 1.0 - w, w)

    def run(batches, reps):
        r, v = 0.02, 0.3
        out = []
        for s, k, t in batches:
            for _ in range(reps):
                sq = v * np.sqrt(t)
                d1 = (np.log(s / k) + (r + 0.5 * v * v) * t) / sq
                d2 = d1 - sq
                disc = k * np.exp(-r * t)
                call = s * cdf(d1) - disc * cdf(d2)
                put = call - s + disc
            out.append((call, put))
        return out

    return make, run


@case("mandelbrot", "algorithms", "Mandelbrot iteration on one complex tile per core",
      "cores", [("small", "vec"), ("medium", "vec")])
def _():
    def make(rng, shape, width):
        n = shape[0]
        tiles = []
        for i in range(width):
            re = np.linspace(-2.0, 0.5, n) + 0.01 * i
            im = np.full(n, -1.0 + 2.0 * (i + 0.5) / width)
            tiles.append(re + 1j * im)
        return tiles

    def run(cs, reps):
        out = []
        for c in cs:
            z = c * 0.0
            count = np.zeros(c.shape)
            for _ in range(reps):
                z = z * z + c
                inside = np.abs(z) <= 2.0
                count = count + inside
                z = z * inside  # escaped points restart at 0 (bounded values)
            out.append(count)
        return out

    return make, run


@case("nbody_forces", "algorithms",
      "pairwise forces via broadcasting (n,1)-(1,n), x and y components", 2,
      [("small", "vec"), ("medium", "vec")])
def _():
    def make(rng, shape, width):
        m = max(2, int(round(shape[0] ** 0.5)))
        return [uniform(rng, (m,), lo=0, hi=1), uniform(rng, (m,), lo=0, hi=1)]

    def run(pos, reps):
        x, y = pos
        for _ in range(reps):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            inv = 1.0 / np.sqrt(dx * dx + dy * dy + 0.01) ** 3
            ax = (dx * inv).sum(axis=1)
            ay = (dy * inv).sum(axis=1)
            x = x + 1e-6 * ax
            y = y + 1e-6 * ay
        return x, y

    return make, run


@case("polynomial_horner", "algorithms",
      "Horner evaluation of a degree-reps polynomial, one input per core", "cores",
      [("small", "vec"), ("medium", "vec"), ("large", "vec")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape, lo=-1, hi=1) for _ in range(width)]

    def run(xs, reps):
        out = []
        for x in xs:
            p = x * 0.0 + 1.0
            for i in range(reps):
                p = p * x + (0.5 / (i + 1))
            out.append(p)
        return out

    return make, run


@case("matmul_batch", "algorithms", "independent matrix products (BLAS), per core",
      "cores", [("small", "C"), ("medium", "C")])
def _():
    def make(rng, shape, width):
        side = max(2, int(round(shape[0] * shape[1]) ** 0.5))
        return [uniform(rng, (side, side)) for _ in range(width)]

    def run(ms, reps):
        out = []
        for m in ms:
            p = m
            for _ in range(reps):
                p = (p @ m) * (1.0 / m.shape[0])
            out.append(p)
        return out

    return make, run


@case("monte_carlo_pi", "algorithms", "Monte Carlo estimate of pi, one batch per core",
      "cores", [("medium", "vec"), ("large", "vec")])
def _():
    def make(rng, shape, width):
        return [(uniform(rng, shape, lo=0, hi=1), uniform(rng, shape, lo=0, hi=1))
                for _ in range(width)]

    def run(batches, reps):
        out = []
        for u, v in batches:
            est = 0.0
            for _ in range(reps):
                inside = (u * u + v * v) <= 1.0
                est = est + 4.0 * inside.mean()
                u = (u + 0.618034) % 1.0
            out.append(est / reps)
        return out

    return make, run


# --- scalars and tiny data: overhead ------------------------------------------


@case("scalar_newton", "scalars", "Newton iteration for sqrt on 0-d arrays", 1,
      [("scalar", "scalar")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape, lo=2, hi=3)]

    def run(data, reps):
        a = data[0]
        x = a * 1.0
        for _ in range(reps):
            x = x - (x * x - a) / (2.0 * x)
            x = x + 0.0 * a  # keep the chain alive after convergence
        return x

    return make, run


@case("tiny_vectors", "scalars", "arithmetic on 8-element vectors with Python scalars", 1,
      [("tiny", "vec"), ("tiny", "C")])
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape), uniform(rng, shape)]

    def run(ab, reps):
        a, b = ab
        for _ in range(reps):
            a = (a * b + 1.0) / (b + 2.0)
            b = np.maximum(a, b) - 0.5 * a
        return a, b

    return make, run


# --- element types ----------------------------------------------------------
# Variants carry a third entry, the dtype (a name, or "a+b" for a mixed-type
# program); `make` then receives it. float16 and the mixed programs (type
# promotion casts an operand) are paths viproc does not run asynchronously
# yet; they show up as fallbacks.

DTYPES = ["float64", "float32", "int32", "int16", "complex128", "float16"]


def typed(rng, shape, dtype):
    dt = np.dtype(dtype)
    if dt.kind in "iu":
        return rng.integers(0, 1000, size=shape).astype(dt)
    if dt.kind == "c":
        return (uniform(rng, shape) + 1j * uniform(rng, shape)).astype(dt)
    return uniform(rng, shape, dt)


def typed_step(x):
    """One update that keeps the dtype (Python scalars are weak, NEP 50) and
    bounded values."""
    kind = x.dtype.kind
    if kind in "iu":
        return (x * 3 + 7) % 1009
    if kind == "c":
        return x * (0.6 + 0.2j) + 0.1
    return np.sqrt(x * 0.75 + 0.5) * 0.9


@case("dtype_chains", "dtypes", "per-core chains of dtype-preserving updates",
      "cores", [(s, "vec", d) for s in ("small", "medium", "large") for d in DTYPES])
def _():
    def make(rng, shape, width, dtype):
        return [typed(rng, shape, dtype) for _ in range(width)]

    def run(xs, reps):
        xs = list(xs)
        for _ in range(reps):
            for i, x in enumerate(xs):
                xs[i] = typed_step(x)
        return xs

    return make, run


@case("dtype_reduce", "dtypes", "per-core update, then mean and max reductions",
      "cores", [("medium", "vec", d) for d in DTYPES])
def _():
    def make(rng, shape, width, dtype):
        return [typed(rng, shape, dtype) for _ in range(width)]

    def run(xs, reps):
        out = []
        for x in xs:
            acc = []
            for _ in range(reps):
                x = typed_step(x)
                acc.append((x.mean(), np.abs(x).max()))
            out.append(acc[-1])
        return out

    return make, run


@case("dtype_promote", "dtypes", "mixed-type arithmetic: type promotion casts an operand",
      "cores", [("medium", "vec", d) for d in
                ("int16+float32", "int32+float64", "float16+float32", "float32+complex128")])
def _():
    def make(rng, shape, width, dtype):
        a, b = dtype.split("+")
        return [(typed(rng, shape, a), typed(rng, shape, b)) for _ in range(width)]

    def run(pairs, reps):
        out = []
        for a, b in pairs:
            r = a * b
            for _ in range(reps):
                r = r * 0.5 + a - b
            out.append(r)
        return out

    return make, run
