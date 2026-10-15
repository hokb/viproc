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


CASES = []
GROUPS = ["dependent", "independent", "mixed", "reductions", "sync"]


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
