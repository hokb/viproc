"""Cases of the shape map benchmark (shape_map.py).

Every case is plain, sequential NumPy code on 2-d data of shape
(rows, cols); it runs unchanged on NumPy and on viproc arrays. The map
measures every case at every grid point, so together they should span the
range viproc meets in practice:

- **dependent**: one dependency chain of 1, 2 or 3 unary ops per step
  (no parallelism at all: pure per-op overhead vs. NumPy).
- **independent**: 2, 3, `cores` and 8 independent chains (parallel width
  P = number of chains), float64 and float32, cheap and expensive ops.
- **mixed**: expressions of unary and binary ops with independent
  subterms (P = 2 and 8), in-place updates, broadcasting of a row and a
  column vector, a 5-point stencil on slices.
- **reductions**: axis reductions feeding back into element-wise ops,
  per-core chains with a full reduction each.
- **sync**: the same kinds of programs with a sync point (a Python float
  read from the data) after every step: the main thread cannot run ahead.
- **expressions**: longer expressions as they occur in numerical code: a
  polynomial, a Gaussian, a Black-Scholes-like formula, element-wise
  conditionals (`np.where`, masked assignment), a row-wise softmax.
- **loops**: Python loops over rows, columns or column blocks of one
  array, each iteration a small array program on a slice. Independent
  iterations (the loop of ILNumerics' "Part 2a": `out[i] = sum(abs(sin(
  A[:, i])))`), order-dependent writes (ILNumerics' "Part 2b":
  `out[:i+1] = ...`, later iterations overwrite earlier ones), a recurrence
  over rows (`x[i]` from `x[i-1]`), a scalar accumulator over independent
  row reductions, and disjoint `out=` blocks of one result array (viproc
  orders writes to one storage, even to disjoint regions). P is the number
  of independent iterations (1 for the recurrence).
- **control**: conditionals in Python: branches on the loop index (no
  sync), on a reduction of the data (sync every step), and a Newton
  iteration with a convergence test every 4th step.

`run(data, reps)` repeats the program `reps` times along its dependency
chains: every step reads the previous step's result (`x = f(x)`). The
step functions are chosen to keep values bounded and away from zero
(no overflow, no subnormal numbers, which would slow down NumPy and viproc
alike and measure the FPU instead of the executor).

`width` is the parallel width P; `mem` the peak number of (rows, cols)
arrays a run keeps alive when its whole program is issued at once (inputs,
viproc copies and temporaries in flight, rounded up); shape_map.py skips
grid points that would not fit into RAM.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Case:
    name: str
    group: str
    description: str
    width: object  # int, or "cores"
    mem: int
    make: object  # make(rng, shape, width) -> list of NumPy arrays
    run: object  # run(data, reps) -> result (array or list of arrays)
    scalar_ok: bool = True  # also meaningful on 0-d data
    complexity: str = ""  # key of CLASSES, set from COMPLEXITY below


CASES = []
GROUPS = ["single", "dependent", "independent", "mixed", "reductions", "sync", "expressions",
          "loops", "control", "algorithms"]

# Complexity classes (shape_map.py --classes: one image each), orthogonal to
# the groups. An op counts as binary when it combines two arrays; a Python or
# NumPy scalar operand (`x * 0.5`, `x ** 3`) is a constant of a unary step.
CLASSES = {
    "single": "single ops: one op per step (x = op(x))",
    "unary": "chains without binary ops: every op reads one array",
    "binary": "chains with binary ops: ops combine two or more arrays",
    "algorithms": "whole algorithms (k-means, power iteration, ...)",
}


def case(name, group, description, width, mem, scalar_ok=True):
    def register(fn):
        make, run = fn()
        CASES.append(Case(name, group, description, width, mem, make, run, scalar_ok))
        return fn

    return register


def uniform(rng, shape, dtype=np.float64, lo=0.5, hi=1.5):
    return np.asarray(rng.uniform(lo, hi, size=shape)).astype(dtype)


def inputs(n, dtype=np.float64):
    def make(rng, shape, width):
        return [uniform(rng, shape, dtype) for _ in range(width if n is None else n)]

    return make


def sync(x):
    """A sync point as a NumPy program has it: read a value into Python."""
    return float(np.max(x))


# --- single ops (P = 1) -------------------------------------------------------------
# One op per step, each reading the previous step's result: what one op costs
# in a stream of ops (dep1_abs below is the cheapest of them).


@case("op_cos", "single", "x = cos(x): 1 expensive unary op per step", 1, 6)
def _():
    def run(d, reps):
        x = d[0]
        for _ in range(reps):
            x = np.cos(x)
        return x

    return inputs(1), run


@case("op_sub", "single", "x = y - x: 1 binary op per step", 1, 8)
def _():
    def run(d, reps):
        x, y = d
        for _ in range(reps):
            x = y - x
        return x

    return inputs(2), run


@case("op_div", "single", "x = y / x: 1 binary op per step", 1, 8)
def _():
    def run(d, reps):
        x, y = d
        for _ in range(reps):
            x = y / x
        return x

    return inputs(2), run


@case("op_matmul", "single", "x = x @ q, q orthogonal (cols, cols): 1 matmul per step", 1, 8,
      scalar_ok=False)
def _():
    def make(rng, shape, width):
        q, _ = np.linalg.qr(rng.standard_normal((shape[1], shape[1])))
        return [uniform(rng, shape), q]

    def run(d, reps):
        x, q = d
        for _ in range(reps):
            x = x @ q
        return x

    return make, run


# --- dependent chains (P = 1) ---------------------------------------------------


@case("dep1_abs", "dependent", "x = abs(x): 1 cheap op per step", 1, 6)
def _():
    def run(d, reps):
        x = d[0]
        for _ in range(reps):
            x = np.abs(x)
        return x

    return inputs(1), run


@case("dep2_abs_pow", "dependent", "x = abs(x ** 0.5): 2 ops per step", 1, 6)
def _():
    def run(d, reps):
        x = d[0]
        for _ in range(reps):
            x = np.abs(np.power(x, 0.5))
        return x

    return inputs(1), run


@case("dep3_cos_pow_neg", "dependent", "x = cos((-x) ** 3): 3 ops per step", 1, 6)
def _():
    def run(d, reps):
        x = d[0]
        for _ in range(reps):
            x = np.cos(np.power(-x, 3))
        return x

    return inputs(1), run


# --- independent chains ---------------------------------------------------------


@case("ind2", "independent", "2 independent chains: abs(a), cos(b)", 2, 8)
def _():
    def run(d, reps):
        a, b = d
        for _ in range(reps):
            a = np.abs(a)
            b = np.cos(b)
        return [a, b]

    return inputs(2), run


@case("ind3", "independent", "3 independent chains: abs(a), cos(b), sqrt(c)", 3, 10)
def _():
    def run(d, reps):
        a, b, c = d
        for _ in range(reps):
            a = np.abs(a)
            b = np.cos(b)
            c = np.sqrt(c)
        return [a, b, c]

    return inputs(3), run


@case("ind_cores_3ops", "independent", "one chain per core, cos((-x) ** 3) per step",
      "cores", 18)
def _():
    def run(xs, reps):
        xs = list(xs)
        for _ in range(reps):
            for i, x in enumerate(xs):
                xs[i] = np.cos(np.power(-x, 3))
        return xs

    return inputs(None), run


@case("ind8_cheap", "independent", "8 chains of a cheap 2-op step (x * 0.5 + 0.5)", 8, 24)
def _():
    def run(xs, reps):
        xs = list(xs)
        for _ in range(reps):
            for i, x in enumerate(xs):
                xs[i] = x * 0.5 + 0.5
        return xs

    return inputs(8), run


@case("ind_cores_f32", "independent", "one float32 chain per core, sin(x) * 0.5 + 0.4",
      "cores", 14)
def _():
    def run(xs, reps):
        xs = list(xs)
        for _ in range(reps):
            for i, x in enumerate(xs):
                xs[i] = np.sin(x) * np.float32(0.5) + np.float32(0.4)
        return xs

    return inputs(None, np.float32), run


# --- mixed unary/binary expressions --------------------------------------------------


@case("expr2", "mixed", "x = sin(x) + cos(x + pi) * 3: two subterms", 2, 10)
def _():
    def run(d, reps):
        x = d[0]
        for _ in range(reps):
            x = np.sin(x) + np.cos(x + np.pi) * 3
        return x

    return inputs(1), run


@case("expr8", "mixed", "x = mean of 8 independent unary/binary terms of x and y", 8, 22)
def _():
    def run(d, reps):
        x, y = d
        for _ in range(reps):
            t = [np.sin(x), np.cos(y), np.exp(-x), np.sqrt(y), np.tanh(x * y),
                 np.arctan(x - y), np.log1p(x * x), y * 0.3]
            s = t[0]
            for u in t[1:]:
                s = s + u
            x = s * 0.125
        return x

    return inputs(2), run


@case("inplace", "mixed", "per core: x *= 0.99; x += y; sqrt(x, out=x)", "cores", 12)
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape) for _ in range(width + 1)]

    def run(d, reps):
        *xs, y = d
        xs = [x.copy() for x in xs]  # mutated below
        for _ in range(reps):
            for x in xs:
                x *= 0.99
                x += y
                np.sqrt(x, out=x)
        return xs

    return make, run


@case("broadcast", "mixed",
      "m = m * 0.5 + col - row: (rows, cols) with (rows, 1) and (1, cols)", 1, 8,
      scalar_ok=False)
def _():
    def make(rng, shape, width):
        rows, cols = shape
        return [uniform(rng, shape), uniform(rng, (rows, 1)), uniform(rng, (1, cols))]

    def run(d, reps):
        m, col, row = d
        for _ in range(reps):
            m = m * 0.5 + col - row
        return m

    return make, run


@case("stencil", "mixed", "5-point stencil on slices (Jacobi step, interior)", 1, 12,
      scalar_ok=False)
def _():
    def run(d, reps):
        x = d[0].copy()
        for _ in range(reps):
            x[1:-1, 1:-1] = 0.25 * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])
        return x

    return inputs(1), run


# --- reductions ------------------------------------------------------------------------


@case("center_cols", "reductions", "x = abs(x - mean(x, axis=0) * 0.5) + 0.1", 1, 8,
      scalar_ok=False)
def _():
    def run(d, reps):
        x = d[0]
        for _ in range(reps):
            x = np.abs(x - x.mean(axis=0) * 0.5) + 0.1
        return x

    return inputs(1), run


@case("normalize_rows", "reductions", "x = x / max(x, axis=1, keepdims) + 0.5", 1, 8,
      scalar_ok=False)
def _():
    def run(d, reps):
        x = d[0]
        for _ in range(reps):
            x = x / np.max(x, axis=1, keepdims=True) + 0.5
        return x

    return inputs(1), run


@case("sum_cores", "reductions", "per core: x = cos(x * 0.5); s += sum(x)", "cores", 12)
def _():
    def run(xs, reps):
        xs = list(xs)
        sums = [np.sum(x) for x in xs]
        for _ in range(reps):
            for i, x in enumerate(xs):
                xs[i] = np.cos(x * 0.5)
                sums[i] = sums[i] + np.sum(xs[i])
        return xs + [np.asarray(s) for s in sums]

    return inputs(None), run


# --- sync after every step ---------------------------------------------------------------


@case("dep3_sync", "sync", "x = cos((-x) ** 3), then read max(x) into Python", 1, 6)
def _():
    def run(d, reps):
        x = d[0]
        for _ in range(reps):
            x = np.cos(np.power(-x, 3))
            sync(x)
        return x

    return inputs(1), run


@case("ind3_sync", "sync", "3 independent chains, read max of each step into Python", 3, 10)
def _():
    def run(d, reps):
        a, b, c = d
        for _ in range(reps):
            a = np.abs(a)
            b = np.cos(b)
            c = np.sqrt(c)
            sync(a), sync(b), sync(c)
        return [a, b, c]

    return inputs(3), run


@case("converge", "sync",
      "iterate x = cos(x) until a max-change test in Python (fixed step count)", 1, 8)
def _():
    def run(d, reps):
        x = d[0]
        for _ in range(reps):
            prev, x = x, np.cos(x)
            if float(np.max(np.abs(x - prev))) < 0.0:  # never true: same work every run
                break
        return x

    return inputs(1), run


# --- longer expressions ----------------------------------------------------------------


@case("expr_poly", "expressions",
      "x = 0.75 + 0.25 * tanh(((0.1 x - 0.2) x + 0.3) x - 0.4): Horner, 9 ops", 1, 6)
def _():
    def run(d, reps):
        x = d[0]
        for _ in range(reps):
            x = 0.75 + 0.25 * np.tanh(((0.1 * x - 0.2) * x + 0.3) * x - 0.4)
        return x

    return inputs(1), run


@case("expr_gauss", "expressions", "x = exp(-(x - mu) ** 2 / 0.5) + 0.5", 1, 8)
def _():
    def run(d, reps):
        x, mu = d
        for _ in range(reps):
            x = np.exp(-((x - mu) ** 2) / 0.5) + 0.5
        return x

    return inputs(2), run


@case("expr_option", "expressions",
      "Black-Scholes-like call price of S, K, T (log, sqrt, exp, tanh; 3 subterms), "
      "S = 1 + 0.4 tanh(price)", 3, 16)
def _():
    def run(d, reps):
        s, k, t = d
        v, r = 0.3, 0.05
        for _ in range(reps):
            sq = v * np.sqrt(t)
            d1 = (np.log(s / k) + (r + 0.5 * v * v) * t) / sq
            d2 = d1 - sq
            c = s * (0.5 + 0.5 * np.tanh(d1)) - k * np.exp(-r * t) * (0.5 + 0.5 * np.tanh(d2))
            s = 1.0 + 0.4 * np.tanh(c)
        return s

    return inputs(3), run


@case("expr_where", "expressions", "x = where(x > 1, x * 0.9, x * 1.1 + 0.05)", 1, 8)
def _():
    def run(d, reps):
        x = d[0]
        for _ in range(reps):
            x = np.where(x > 1.0, x * 0.9, x * 1.1 + 0.05)
        return x

    return inputs(1), run


@case("expr_mask", "expressions", "x = x * 1.07; x[x > 1.4] = 0.6 (masked assignment)", 1, 8,
      scalar_ok=False)
def _():
    def run(d, reps):
        x = d[0].copy()
        for _ in range(reps):
            x = x * 1.07
            x[x > 1.4] = 0.6
        return x

    return inputs(1), run


@case("expr_softmax", "expressions",
      "x = 0.5 + softmax(x) along rows (max, exp, sum with keepdims)", 1, 8, scalar_ok=False)
def _():
    def run(d, reps):
        x = d[0]
        for _ in range(reps):
            e = np.exp(x - np.max(x, axis=1, keepdims=True))
            x = 0.5 + e / np.sum(e, axis=1, keepdims=True)
        return x

    return inputs(1), run


# --- loops over slices -------------------------------------------------------------------


@case("loop_cols", "loops",
      "for i in cols: out[i] = sum(abs(sin(A[:, i]))) (independent, ILNumerics Part 2a)",
      "cols", 6, scalar_ok=False)
def _():
    def run(d, reps):
        a = d[0]
        out = a[0] * 0.0
        for _ in range(reps):
            for i in range(a.shape[1]):
                out[i] = np.sum(np.abs(np.sin(a[:, i])))
        return out

    return inputs(1), run


@case("loop_cols_dep", "loops",
      "for i in cols: out[:i+1] = sum(abs(sin(A[:, i]))) (order-dependent, ILNumerics Part 2b)",
      "cols", 6, scalar_ok=False)
def _():
    def run(d, reps):
        a = d[0]
        out = a[0] * 0.0
        for _ in range(reps):
            for i in range(a.shape[1]):
                out[:i + 1] = np.sum(np.abs(np.sin(a[:, i])))
        return out

    return inputs(1), run


@case("loop_rows", "loops", "for i in rows: out[i] = cos(A[i] * 0.5) * A[i] (independent)",
      "rows", 8, scalar_ok=False)
def _():
    def run(d, reps):
        a = d[0]
        out = a * 0.0
        for _ in range(reps):
            for i in range(a.shape[0]):
                out[i] = np.cos(a[i] * 0.5) * a[i]
        return out

    return inputs(1), run


@case("loop_scan", "loops",
      "for i in rows: x[i] = cos(x[i-1]) * 0.5 + x[i] * 0.5 (recurrence, in place)", 1, 8,
      scalar_ok=False)
def _():
    def run(d, reps):
        x = d[0].copy()
        for _ in range(reps):
            for i in range(1, x.shape[0]):
                x[i] = np.cos(x[i - 1]) * 0.5 + x[i] * 0.5
        return x

    return inputs(1), run


@case("loop_accum", "loops", "s = 0; for i in rows: s = s + sum(sqrt(A[i])) (scalar carry)",
      "rows", 6, scalar_ok=False)
def _():
    def run(d, reps):
        a = d[0]
        s = 0.0
        for _ in range(reps):
            for i in range(a.shape[0]):
                s = s + np.sum(np.sqrt(a[i]))
        return np.asarray(s)

    return inputs(1), run


@case("loop_blocks_out", "loops",
      "for j in cols step 4: tanh(A[:, j:j+4] * 0.5, out=B[:, j:j+4]) (disjoint writes)",
      "cols/4", 8, scalar_ok=False)
def _():
    def run(d, reps):
        a = d[0]
        b = a * 0.0
        for _ in range(reps):
            for j in range(0, a.shape[1], 4):
                np.tanh(a[:, j:j + 4] * 0.5, out=b[:, j:j + 4])
        return b

    return inputs(1), run


# --- conditionals ---------------------------------------------------------------------------


@case("cond_index", "control",
      "2 chains, branch on the step index k % 3: sqrt / square / abs(sin) (no sync)", 2, 10)
def _():
    def run(d, reps):
        a, b = d
        for k in range(reps):
            if k % 3 == 0:
                a = np.sqrt(a) + 0.3
                b = b * b * 0.5 + 0.4
            elif k % 3 == 1:
                a = a * a * 0.5 + 0.4
                b = np.abs(np.sin(b)) + 0.5
            else:
                a = np.abs(np.sin(a)) + 0.5
                b = np.sqrt(b) + 0.3
        return [a, b]

    return inputs(2), run


@case("cond_data", "control",
      "if max(x) > 1.2: x = x * 0.9 else: x = x * 1.05 + 0.02 (sync every step)", 1, 6)
def _():
    def run(d, reps):
        x = d[0]
        for _ in range(reps):
            if np.max(x) > 1.2:
                x = x * 0.9
            else:
                x = x * 1.05 + 0.02
        return x

    return inputs(1), run


@case("cond_newton", "control",
      "Newton x = (x + a / x) / 2 for sqrt(a), convergence test every 4th step (sync)", 1, 8)
def _():
    def run(d, reps):
        a = d[0]
        x = a * 1.0
        for k in range(reps):
            x = 0.5 * (x + a / x)
            if k % 4 == 3 and float(np.max(np.abs(x * x - a))) < 0.0:  # never: fixed work
                break
        return x

    return inputs(1), run


# --- whole algorithms -------------------------------------------------------------------


@case("kmeans_points", "algorithms",
      "k-means of ILNumerics' Part 3: rows = dimensions, cols = points, k = cols / 10 "
      "(2..50); per point distances and argmin, per cluster nonzero (sync) and mean",
      "cols", 8, scalar_ok=False)
def _():
    # The NumPy program of github.com/hokb/decentralized-array-execution-
    # artifacts2026 (Appendix, Part 3 KMeans), as suite_cases.kmeans_points,
    # with a fixed iteration count (reps) instead of the convergence test.
    def make(rng, shape, width):
        return [rng.uniform(0, 10, size=shape)]

    def run(d, reps):
        X = d[0]
        n = np.size(X, 1)
        k = min(50, max(2, n // 10))
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
        return [classes, centers]

    return make, run


@case("kmeans_vec", "algorithms",
      "k-means, k = 8, of the 2-d points (x, y) (each of the shape): distances, labels by "
      "minimum and where, centroids from masked sums (sync per cluster)", 8, 30)
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
        return [labels, np.array(cx), np.array(cy)]

    return make, run


@case("power_iter", "algorithms",
      "power iteration on A^T A: v = A.T @ (A @ v); v = v / norm(v)", 1, 4, scalar_ok=False)
def _():
    def make(rng, shape, width):
        return [uniform(rng, shape), uniform(rng, (shape[1],))]

    def run(d, reps):
        a, v = d
        for _ in range(reps):
            v = a.T @ (a @ v)
            v = v / np.linalg.norm(v)
        return v

    return make, run


@case("logreg", "algorithms",
      "logistic regression, gradient descent: rows = samples, cols = features; "
      "p = 1 / (1 + exp(-X @ w)); w -= 0.1 * X.T @ (p - y) / rows", 1, 4, scalar_ok=False)
def _():
    def make(rng, shape, width):
        x = uniform(rng, shape, lo=-1.0, hi=1.0)
        y = (rng.uniform(size=shape[0]) < 0.5).astype(np.float64)
        return [x, y, uniform(rng, (shape[1],), lo=-0.1, hi=0.1)]

    def run(d, reps):
        x, y, w = d
        n = x.shape[0]
        for _ in range(reps):
            p = 1.0 / (1.0 + np.exp(-(x @ w)))
            w = w - 0.1 * (x.T @ (p - y)) / n
        return w

    return make, run


@case("mandelbrot", "algorithms",
      "Mandelbrot iteration on a complex tile: z = where(|z| <= 2, z * z + c, z); "
      "count += |z| <= 2", 1, 20)
def _():
    def make(rng, shape, width):
        rows, cols = shape if shape else (1, 1)
        re = np.linspace(-2.0, 0.5, cols)
        im = np.linspace(-1.25, 1.25, rows)
        c = (re + 1j * im[:, None]).reshape(shape)
        return [c]

    def run(d, reps):
        c = d[0]
        z = c * 0.0
        count = c.real * 0.0
        for _ in range(reps):
            inside = np.abs(z) <= 2.0
            z = np.where(inside, z * z + c, z)
            count += inside
        return [z, count]

    return make, run


COMPLEXITY = {
    "single": ["op_cos", "op_sub", "op_div", "op_matmul", "dep1_abs"],
    "unary": ["dep2_abs_pow", "dep3_cos_pow_neg", "ind2", "ind3", "ind_cores_3ops",
              "ind8_cheap", "ind_cores_f32", "dep3_sync", "ind3_sync", "loop_cols",
              "loop_cols_dep", "loop_blocks_out", "cond_data"],
    "binary": ["expr2", "expr8", "inplace", "broadcast", "stencil", "center_cols",
               "normalize_rows", "sum_cores", "converge", "expr_poly", "expr_gauss",
               "expr_option", "expr_where", "expr_mask", "expr_softmax", "loop_rows",
               "loop_scan", "loop_accum", "cond_index", "cond_newton"],
    "algorithms": ["kmeans_points", "kmeans_vec", "power_iter", "logreg", "mandelbrot"],
}
for _cls, _names in COMPLEXITY.items():
    for _c in CASES:
        if _c.name in _names:
            _c.complexity = _cls
assert all(c.complexity for c in CASES), [c.name for c in CASES if not c.complexity]
assert sum(map(len, COMPLEXITY.values())) == len(CASES)
