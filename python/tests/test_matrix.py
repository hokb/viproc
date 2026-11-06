"""A parameter matrix against NumPy: every op runs on NumPy and on viproc
with the same inputs, and must give the same values (bit for bit), dtype,
shape, strides, warnings and errors.

Dimensions (hokb/viproc#62 review): shapes (0-d scalars; empty arrays of
1, 2, 3 and 8 dimensions; vectors; matrices; 3-d and 8-d arrays, with
extent-1 axes), layouts (C, F, transposed, stepped, reversed), dtypes,
element values (NaN, +inf, -inf, -0.0, integer extremes), ready or
pending inputs (the direct path or a task: "sync / async"), and per op
group its own parameters (broadcasting pairs, axes and keepdims, casts
that overflow, indices and masks). The suite runs at O1; `VIPROC_OPT=2`
runs it at O2 (values may deviate there: those checks are marked)."""

import itertools
import warnings
import zlib

import numpy as np
import pytest

import viproc

SHAPES = {
    "0d": (),
    "e1": (0,),
    "e2": (2, 0),
    "e3": (0, 3, 1),
    "e8": (1, 2, 0, 1, 2, 1, 1, 2),
    "v1": (1,),
    "v7": (7,),
    "v3000": (3000,),
    "m34": (3, 4),
    "m151": (1, 5, 1),
    "t234": (2, 3, 4),
    "t8": (2, 1, 2, 1, 2, 1, 2, 3),
    "big": (60, 70),
}
VIEWS = {
    "C": lambda a: a,
    "F": None,  # the base in F order
    "T": lambda a: a.T,
    "step": lambda a: a[..., ::2] if a.ndim else a,
    "rev": lambda a: a[::-1] if a.ndim else a,
}
DTYPES = ["b1", "i4", "i8", "f4", "f8", "c8", "c16"]
MODES = ["ready", "pending"]


def values(rng, shape, dtype):
    """Random elements of `dtype` with the special values mixed in."""
    n = int(np.prod(shape))
    dt = np.dtype(dtype)
    if dt.kind == "b":
        x = rng.integers(0, 2, n).astype(bool)
    elif dt.kind == "i":
        info = np.iinfo(dt)
        x = rng.integers(-50, 50, n).astype(dt)
        special = np.array([0, -1, 1, info.min, info.max, info.min + 1, info.max - 1], dt)
        k = min(n // 3, len(special))
        x[rng.choice(n, k, replace=False)] = special[:k] if n else x[:0]
    else:
        real = (rng.standard_normal(n) * 10).astype(dt.char.lower() if dt.kind == "c" else dt)
        special = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1e30, -1e30, 2.5, -2.5, 0.5])
        k = min(n // 3, len(special))
        if k:
            real[rng.choice(n, k, replace=False)] = special[:k]
        x = real
        if dt.kind == "c":
            imag = (rng.standard_normal(n) * 10).astype(real.dtype)
            if k:
                imag[rng.choice(n, k, replace=False)] = special[:k][::-1]
            x = real + 1j * imag
        x = x.astype(dt)
    return x.reshape(shape)


def make(a, view, mode="ready"):
    """The NumPy operand: `a` (C order), in F order or viewed. Pending
    operands are copies made by np.where (to_viproc): their NumPy
    counterpart is made the same way (np.where gives empty results zero
    strides and extent-1 axes the strides of its iteration order)."""
    if view == "F":
        a = np.asfortranarray(a)
    if mode == "pending":
        a = np.where(True, a, a)
    return a if view == "F" else VIEWS[view](a)


_COND = []


def _pending_true():
    """A 0-d bool viproc array that is True but most likely still pending."""
    x = viproc.asarray(np.full(200_000, 0.5))
    return np.sin(np.sin(x)).max() < 2.0


def to_viproc(a, base, view, mode):
    """The viproc operand of the NumPy operand `a` = view(base): ready
    (imported), or pending (an exact copy behind a slow op), then viewed."""
    if mode == "ready":
        return viproc.asarray(a)
    v = viproc.asarray(np.asfortranarray(base) if view == "F" else base)
    v = np.where(_pending_true(), v, v)  # exact, pending, order K
    return v if view == "F" else VIEWS[view](v)


def same_bits(e, g):
    e, g = np.asarray(e), np.asarray(g)
    if e.dtype != g.dtype or e.shape != g.shape:
        return False
    if e.dtype.kind in "fc":  # NaNs: any payload NumPy may produce
        en, gn = np.isnan(e), np.isnan(g)
        if not np.array_equal(en, gn):
            return False
        e, g = np.where(en, 0, e), np.where(gn, 0, g)
    return np.ascontiguousarray(e).tobytes() == np.ascontiguousarray(g).tobytes()


def run(fn, args):
    """fn(*args): (results, error, warnings), results synced."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = fn(*args)
            rs = r if isinstance(r, tuple) else (r,)
            rs = tuple(np.asarray(x) if isinstance(x, viproc.ndarray) else x for x in rs)
            viproc.wait_all()
            err = None
        except Exception as ex:  # noqa: BLE001 - compared below
            rs, err = None, ex
            try:
                viproc.wait_all()
            except Exception:  # noqa: BLE001 - the same error, raised once more
                pass
    return rs, err, sorted({(x.category.__name__, str(x.message)) for x in w})


def seed(key):
    """A seed that does not change between runs (hash() of str does)."""
    return zlib.crc32(repr(key).encode())


def check(fn, np_args, vp_args, strides=True, values_=True, label=None):
    """fn on NumPy and on viproc: the same results (values, dtype, shape,
    strides), the same error (type and message) and the same warnings."""
    try:
        _check(fn, np_args, vp_args, strides, values_)
    except AssertionError as ex:
        raise AssertionError(f"{label or fn.__name__}: {ex}") from None


def _check(fn, np_args, vp_args, strides, values_):
    e, eerr, ew = run(fn, np_args)
    g, gerr, gw = run(fn, vp_args)
    if eerr is not None or gerr is not None:
        assert eerr is not None and gerr is not None, (eerr, gerr, e, g)
        assert isinstance(gerr, type(eerr)), (type(eerr), type(gerr), eerr, gerr)
        assert str(gerr).startswith(str(eerr)), (str(eerr), str(gerr))
        return
    assert len(e) == len(g)
    for x, y in zip(e, g):
        assert not isinstance(y, viproc.ndarray)
        xa, ya = np.asarray(x), np.asarray(y)
        assert xa.dtype == ya.dtype and xa.shape == ya.shape, (xa.dtype, ya.dtype, xa.shape, ya.shape)
        if strides:
            assert xa.strides == ya.strides, (xa.strides, ya.strides)
        if values_:
            assert same_bits(xa, ya), (xa, ya)
        assert isinstance(x, np.generic) == isinstance(y, np.generic) or ya.ndim == 0
    assert ew == gw, (ew, gw)


O2 = None


def deviates():
    """O2 may compute deviating values (goals.md, optimization levels)."""
    return viproc.opt_level() >= 2


def cases(*dims):
    return list(itertools.product(*dims))


# --- element-wise unary ---------------------------------------------------

UNARY = {
    "negative": np.negative, "absolute": np.absolute, "sqrt": np.sqrt, "exp": np.exp,
    "sin": np.sin, "floor": np.floor, "rint": np.rint, "sign": np.sign, "isnan": np.isnan,
    "isinf": np.isinf, "signbit": np.signbit, "square": np.square, "logical_not": np.logical_not,
    "invert": np.invert,
}


@pytest.mark.parametrize("shape, view, dtype, mode",
                         cases(SHAPES, VIEWS, DTYPES, MODES))
def test_unary(shape, view, dtype, mode):
    rng = np.random.default_rng(seed((shape, view, dtype)))
    base = values(rng, SHAPES[shape], dtype)
    a = make(base, view, mode)
    for name, f in UNARY.items():
        check(f, (a,), (to_viproc(a, base, view, mode),), values_=not deviates(), label=name)


# --- element-wise binary with broadcasting -------------------------------

PAIRS = [
    ((), ()), ((), (7,)), ((7,), ()), ((0,), ()), ((0,), (1,)), ((1,), (0,)), ((2, 0), (0,)),
    ((2, 0), (2, 1)), ((3, 4), (4,)), ((3, 1), (1, 4)), ((1, 5, 1), (3, 1, 4)),
    ((2, 3, 4), (3, 1)), ((2, 1, 2, 1, 2, 1, 2, 3), (2, 1, 3)), ((60, 70), (70,)),
    ((60, 1), (1, 70)), ((3000,), (3000,)), ((3, 4), (3,)), ((2, 0), (3,)),
    ((0, 3, 1), (2,)),
]
BINARY = {
    "add": np.add, "subtract": np.subtract, "multiply": np.multiply, "divide": np.divide,
    "floor_divide": np.floor_divide, "remainder": np.remainder, "power": np.power,
    "maximum": np.maximum, "fmin": np.fmin, "less": np.less, "equal": np.equal,
    "logical_and": np.logical_and, "arctan2": np.arctan2, "bitwise_and": np.bitwise_and,
}
DTYPE_PAIRS = [("f8", "f8"), ("i4", "f8"), ("i8", "i8"), ("i4", "i8"), ("b1", "i4"),
               ("f4", "f4"), ("f4", "c8"), ("c16", "f8"), ("b1", "b1"), ("i8", "f4")]


@pytest.mark.parametrize("pair, dtypes, views, modes",
                         cases(PAIRS, DTYPE_PAIRS, [("C", "C"), ("F", "rev"), ("T", "step")],
                               [("ready", "ready"), ("pending", "ready"),
                                ("pending", "pending")]),
                         ids=lambda x: str(x).replace(" ", ""))
def test_binary(pair, dtypes, views, modes):
    rng = np.random.default_rng(seed((pair, dtypes, views)))
    bases = [values(rng, s, d) for s, d in zip(pair, dtypes)]
    if views[1] in ("step", "rev"):  # views that keep the shape: only on 1-d+ data
        bases[1] = values(rng, pair[1][:-1] + (2 * pair[1][-1],), dtypes[1]) \
            if views[1] == "step" and pair[1] else bases[1]
    nps = [make(b, v, m) for b, v, m in zip(bases, views, modes)]
    if views[0] == "T":
        bases[0] = values(rng, pair[0][::-1], dtypes[0])
        nps[0] = make(bases[0], "T", modes[0])
    vps = [to_viproc(a, b, v, m) for a, b, v, m in zip(nps, bases, views, modes)]
    for name, f in BINARY.items():
        check(f, tuple(nps), tuple(vps), values_=not deviates(), label=name)
    # Python scalars (NEP 50: weak), also special values
    for s in (2, -1, 2.5, float("nan"), float("inf"), -0.0, 1j, True):
        for f in (np.add, np.multiply, np.maximum, np.less):
            check(lambda x: f(x, s), (nps[0],), (vps[0],), values_=not deviates(),
                  label=f"{f.__name__}(x, {s!r})")
            check(lambda x: f(s, x), (nps[0],), (vps[0],), values_=not deviates(),
                  label=f"{f.__name__}({s!r}, x)")


# --- reductions: axes, keepdims ("scalar dimension removal") ---------------

REDUCTIONS = {
    "sum": np.sum, "prod": np.prod, "min": np.min, "max": np.max, "any": np.any,
    "all": np.all, "mean": np.mean, "std": np.std, "argmax": np.argmax,
}


@pytest.mark.parametrize("shape, view, dtype, mode, keepdims",
                         cases(SHAPES, ["C", "F", "T"], ["b1", "i4", "i8", "f4", "f8", "c16"],
                               MODES, [False, True]))
def test_reductions(shape, view, dtype, mode, keepdims):
    rng = np.random.default_rng(seed((shape, view, dtype)))
    base = values(rng, SHAPES[shape], dtype)
    a = make(base, view, mode)
    axes = [None] + list(range(-a.ndim, a.ndim))[:3] + ([(0, a.ndim - 1)] if a.ndim > 1 else [])
    for (name, f), axis in itertools.product(REDUCTIONS.items(), axes):
        if name == "argmax" and isinstance(axis, tuple):
            continue
        kw = {"axis": axis, "keepdims": keepdims}
        check(lambda x: f(x, **kw), (a,), (to_viproc(a, base, view, mode),),
              values_=not deviates(), label=f"{name}({kw})")


# --- casts: overflow, rounding, NaN and infinities --------------------------

CASTS = [("f8", "i4"), ("f8", "i8"), ("f4", "i4"), ("f8", "b1"), ("i8", "i4"), ("i4", "i8"),
         ("i8", "f4"), ("i8", "f8"), ("c16", "f8"), ("f8", "f4"), ("b1", "f8"), ("f8", "c8")]


@pytest.mark.parametrize("shape, view, cast, mode",
                         cases(SHAPES, ["C", "F", "rev"], CASTS, MODES))
def test_casts(shape, view, cast, mode):
    rng = np.random.default_rng(seed((shape, view, cast)))
    base = values(rng, SHAPES[shape], cast[0])
    if np.dtype(cast[0]).kind == "f":  # large values: overflow on the way to integers
        base = base * np.asarray(1e9, base.dtype)
    a = make(base, view, mode)
    v = to_viproc(a, base, view, mode)
    for order in ("K", "C", "F"):
        check(lambda x: x.astype(cast[1], order=order), (a,), (v,), label=f"astype {order}")
    check(lambda x: np.round(x, 1), (a,), (v,), values_=not deviates())
    check(lambda x: np.round(x, -2), (a,), (v,), values_=not deviates())
    if np.dtype(cast[0]).kind != "c":
        check(lambda x: np.clip(x, -3, 3), (a,), (v,))
        check(lambda x: np.floor(x).astype(cast[1]), (a,), (v,))
    check(lambda x: np.add(x, x, dtype=cast[1], casting="unsafe"), (a,), (v,),
          values_=not deviates())


# --- indexing: integers remove dimensions, slices, index arrays, masks ------

def keys(shape):
    nd = len(shape)
    out = [(), (Ellipsis,), (None,)]
    if nd:
        n0 = shape[0]
        out += [(slice(None),), (slice(1, None),), (slice(None, None, -2),), (Ellipsis, None)]
        if n0:
            out += [(0,), (-1,), (n0,), (-n0 - 1,), (np.array([0, -1, 0]),),
                    (np.array([n0]),), (np.array([], np.intp),), (n0 - 1, Ellipsis)]
        if nd > 1 and shape[1]:
            out += [(slice(None), 0), (slice(None), np.array([[0], [shape[1] - 1]])),
                    (np.array([0]), slice(None), ) if n0 else (slice(None),)]
    return out


@pytest.mark.parametrize("shape, view, dtype, mode",
                         cases(SHAPES, ["C", "F", "T"], ["b1", "i8", "f8", "c16"], MODES))
def test_indexing(shape, view, dtype, mode):
    rng = np.random.default_rng(seed((shape, view, dtype)))
    base = values(rng, SHAPES[shape], dtype)
    a = make(base, view, mode)
    v = to_viproc(a, base, view, mode)
    for key in keys(a.shape):
        check(lambda x: x[key], (a,), (v,), label=f"x[{key!r}]")
    if dtype in ("f8", "c16"):  # masks from NaN and the infinities
        check(lambda x: x[np.isnan(x)], (a,), (v,))
        check(lambda x: x[~np.isfinite(x)], (a,), (v,))
        check(lambda x: np.where(np.isnan(x), 0, x), (a,), (v,))
