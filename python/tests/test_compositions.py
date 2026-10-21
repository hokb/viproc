"""NumPy functions viproc composes of its asynchronous ops (_compose.py,
hokb/viproc#25 step 2): bit-identical to NumPy (values, dtype, shape and
layout) over dtypes, layouts and sizes, and on the asynchronous path."""

import warnings

import numpy as np
import pytest

import viproc
from async_path import SIZE, assert_async, deterministic, pending

rng = np.random.default_rng(2)

DTYPES = [np.float64, np.float32, np.complex128, np.complex64, np.int64, np.int32, np.bool_]


def sample(shape, dtype, special=True):
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return rng.random(shape) > 0.5
    if dtype.kind in "iu":
        return rng.integers(-50, 50, shape).astype(dtype)
    x = (rng.standard_normal(shape) * 3).astype(dtype)
    if dtype.kind == "c":
        x = x + 1j * (rng.standard_normal(shape) * 3).astype(dtype)
    if special and x.size > 8:
        flat = x.reshape(-1)
        flat[1], flat[3], flat[5], flat[7] = np.nan, np.inf, -np.inf, -0.0
    return x


def layouts(x):
    """(name, NumPy array, viproc array) for several layouts of x."""
    v = viproc.asarray(x)
    yield "c", x, v
    if x.ndim >= 2:
        yield "t", x.T, v.T
        yield "step", x[::2, ::-1], v[::2, ::-1]
    elif x.ndim == 1:
        yield "rev", x[::-3], v[::-3]


def same(r, expected):
    if isinstance(expected, (tuple, list)):
        assert type(r) is type(expected) and len(r) == len(expected)
        for a, b in zip(r, expected):
            same(a, b)
        return
    if not isinstance(expected, (np.ndarray, np.generic)):  # bool, shape, dtype, ...
        assert type(r) is type(expected) and r == expected
        return
    got, want = np.asarray(r), np.asarray(expected)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert np.array_equal(got, want, equal_nan=True)
    # NumPy's layout (extent-1 axes may have any stride)
    assert [s for s, n in zip(got.strides, got.shape) if n > 1] == [
        s for s, n in zip(want.strides, want.shape) if n > 1]


def check(fn, *arrays, **kwargs):
    """fn on NumPy arrays and on viproc arrays (all layouts of the first one)
    give the same result or raise the same error type, and the same warnings."""
    first, rest = arrays[0], arrays[1:]
    for _, x, v in layouts(first):
        vr = [viproc.asarray(a) if isinstance(a, np.ndarray) else a for a in rest]
        with warnings.catch_warnings(record=True) as want_w:
            warnings.simplefilter("always")
            try:
                want = fn(x, *rest, **kwargs)
            except Exception as e:  # noqa: BLE001
                with pytest.raises(type(e)):
                    fn(v, *vr, **kwargs)
                continue
        with warnings.catch_warnings(record=True) as got_w:
            warnings.simplefilter("always")
            got = fn(v, *vr, **kwargs)
        same(got, want)
        assert messages(got_w) == messages(want_w)


def messages(caught):
    # NumPy's floating-point warnings ("invalid value encountered in ...") are
    # not reported by viproc yet (CLAUDE.md, FP flags); the others are code's.
    return sorted({str(w.message) for w in caught if "encountered in" not in str(w.message)})


SHAPES = [(37,), (6, 7), (3000,), (60, 50)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_elementwise(shape, dtype):
    x = sample(shape, dtype)
    if np.dtype(dtype).kind != "c":
        check(np.clip, x, -2, 2)
        check(lambda a: np.clip(a, max=1), x)
        check(lambda a: np.clip(a, min=-1), x)
        check(lambda a: a.clip(-1, None), x)
    for d in (0, 1, 2, -1, 20):
        if np.dtype(dtype).kind != "b":
            check(np.round, x, d)
            check(np.around, x, d)
    check(np.angle, x)
    check(lambda a: np.angle(a, deg=True), x)
    check(np.sinc, x)
    check(np.nan_to_num, x)
    check(lambda a: np.nan_to_num(a, nan=7.0, posinf=1e3, neginf=-1e3), x)
    y = sample(shape, dtype)
    check(np.isclose, x, y)
    check(np.isclose, x, x.copy())
    check(lambda a: np.isclose(a, a, equal_nan=True), x)
    check(np.allclose, x, x.copy())
    check(lambda a: np.allclose(a, a, equal_nan=True), x)
    check(np.array_equal, x, x.copy())
    check(lambda a: np.array_equal(a, a.copy(), equal_nan=True), x)
    check(np.array_equal, x, y)
    if np.dtype(dtype).kind in "fc":
        check(lambda a: np.polyval([1.5, -2.0, 0.25], a), x)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_reductions(shape, dtype):
    x = sample(shape, dtype)
    kinds = np.dtype(dtype).kind
    if kinds != "c":
        check(np.ptp, x)
        check(lambda a: np.ptp(a, axis=0), x)
    if x.ndim == 2:
        check(np.trace, x)
        check(lambda a: np.trace(a, 1), x)
        check(lambda a: np.trace(a, -1, dtype=np.float64), x)
    check(np.average, x)
    check(lambda a: np.average(a, axis=0, returned=True), x)
    w = np.abs(sample(x.shape, np.float64, False)) + 0.5
    check(lambda a, w: np.average(a, weights=w), x, w)
    check(np.diff, x)
    check(lambda a: np.diff(a, 2, axis=0), x)
    for f in (np.nansum, np.nanprod, np.nanmax, np.nanmin, np.nanmean, np.nanvar, np.nanstd,
              np.nanargmax, np.nanargmin):
        if kinds == "c" and f in (np.nanmax, np.nanmin, np.nanargmax, np.nanargmin):
            continue
        check(f, x)
        check(lambda a: f(a, axis=0), x)
        if f not in (np.nanargmax, np.nanargmin):
            check(lambda a: f(a, axis=-1, keepdims=True), x)
    check(lambda a: np.nanvar(a, ddof=1), x)
    check(lambda a: np.nanstd(a, axis=0, ddof=2), x)


def test_nan_functions_on_all_nan_slices():
    x = np.full((4, 5), np.nan)
    x[1, 2] = 3.0
    for f in (np.nanmax, np.nanmean, np.nanvar, np.nanstd, np.nanargmax):
        check(lambda a: f(a, axis=0), x)
        check(f, x)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.complex128, np.int64])
def test_norm(dtype):
    for shape in [(37,), (3000,), (60, 50)]:
        x = sample(shape, dtype, special=False)
        check(np.linalg.norm, x)
        check(lambda a: np.linalg.norm(a, keepdims=True), x)
        for ord in (None, 2, 1, np.inf, -np.inf, 0, 3, 0.5):
            check(lambda a: np.linalg.norm(a, ord, axis=0), x)
        if x.ndim == 2:
            check(lambda a: np.linalg.norm(a, "fro"), x)
            check(lambda a: np.linalg.norm(a, 2, axis=1), x)


@pytest.mark.parametrize("dtype", DTYPES)
def test_products(dtype):
    a, b = sample((40,), dtype), sample((30,), dtype)
    check(np.outer, a, b)
    check(np.outer, sample((4, 5), dtype), b)
    if np.dtype(dtype).kind in "fc":
        c = sample((40,), dtype, special=False)
        check(np.inner, c, c[::-1].copy())


@pytest.mark.parametrize("dtype", DTYPES)
def test_copies_views_and_new_arrays(dtype):
    x = sample((6, 7), dtype)
    for order in ("K", "A", "C", "F"):
        check(lambda a: np.copy(a, order=order), x)
    check(np.imag, x) if np.dtype(dtype).kind == "c" else None
    for f in (np.zeros_like, np.ones_like, np.empty_like):
        check(lambda a: f(a).shape, x)
        check(lambda a: f(a).strides, x)
        check(lambda a: f(a, dtype=np.float32).dtype, x)
    check(np.zeros_like, x)
    check(np.ones_like, x)
    check(lambda a: np.full_like(a, 3), x)
    check(lambda a: np.zeros_like(a, order="F"), x)
    check(lambda a: np.ones_like(a, shape=(3, 2)), x)
    for k in (0, 1, -2, 9):
        check(lambda a: np.triu(a, k), x)
        check(lambda a: np.tril(a, k), x)
        check(lambda a: np.diag(a, k), x)
        check(lambda a: np.diag(a, k), sample((7,), dtype))
    check(lambda a: np.triu(a), sample((3, 4, 5), dtype))
    check(np.broadcast_arrays, sample((1, 7), dtype), sample((6, 1), dtype))
    check(lambda a, b: np.meshgrid(a, b), sample((5,), dtype), sample((3,), dtype))
    check(lambda a, b: np.meshgrid(a, b, indexing="ij", sparse=True), sample((5,), dtype),
          sample((3,), dtype))
    check(lambda a: np.split(a, 3, axis=1), sample((6, 9), dtype))
    check(lambda a: np.array_split(a, 4), x)
    check(lambda a: np.array_split(a, [1, 4], axis=1), x)
    check(lambda a: np.split(a, 4), x)  # NumPy's ValueError


def test_diag_result_is_writable_and_views_follow_numpy():
    v = viproc.asarray(np.arange(4.0))
    d = np.diag(v)
    d[0, 1] = 5.0
    e = np.diag(np.arange(4.0))
    e[0, 1] = 5.0
    same(d, e)
    m = viproc.asarray(np.arange(12.0).reshape(3, 4))
    assert not np.diag(m).flags.writeable  # NumPy: a read-only view


# --- asynchronous path ------------------------------------------------------------


@pytest.fixture
def det():
    with deterministic():
        yield


def big(shape=(SIZE,), dtype=np.float64, special=False):
    x = sample(shape, dtype, special)
    return x, pending(viproc.asarray(x))


ASYNC = [  # name, call, inputs (shape, dtype), ops, small_steps, syncs[, min_tasks]
    ("clip", lambda a: np.clip(a, -1, 1), [((SIZE,), "f8")], 1, False, 0),
    ("clip_max", lambda a: a.clip(max=1), [((SIZE,), "f8")], 1, False, 0),
    ("round0", np.round, [((SIZE,), "f8")], 1, False, 0),
    ("round2", lambda a: np.round(a, 2), [((SIZE,), "f8")], 3, False, 0),
    ("round2_strided", lambda a: np.round(a[::2].T, 2), [((100, 100), "f8")], 3, False, 0),
    # cast to float64, divide, rint, multiply, cast back
    ("round_int", lambda a: np.round(a, -1), [((SIZE,), "i8")], 5, False, 0),
    ("round_complex", lambda a: a.round(1), [((SIZE,), "c16")], 9, False, 0),
    ("angle", np.angle, [((SIZE,), "c16")], 1, False, 0),
    ("sinc", np.sinc, [((SIZE,), "f8")], 5, False, 0),
    ("nan_to_num", np.nan_to_num, [((SIZE,), "f8")], 12, False, 0),
    ("isclose", np.isclose, [((SIZE,), "f8"), ((SIZE,), "f8")], 10, False, 0),
    ("polyval", lambda a: np.polyval([1.0, 2.0, 3.0], a), [((SIZE,), "f8")], 6, False, 0),
    ("ptp", np.ptp, [((SIZE,), "f8")], 3, True, 0),
    ("ptp_axis", lambda a: np.ptp(a, axis=0), [((100, 100), "f8")], 3, True, 0),
    ("trace", np.trace, [((3000, 3000), "f8")], 1, False, 0),
    ("average", np.average, [((SIZE,), "f8")], 2, True, 0),
    ("diff", np.diff, [((SIZE,), "f8")], 1, False, 0),
    ("diff2", lambda a: np.diff(a, 2, axis=0), [((100, 100), "f8")], 2, False, 0),
    ("nansum", np.nansum, [((SIZE,), "f8")], 4, False, 0),
    ("nanprod_axis", lambda a: np.nanprod(a, axis=0), [((100, 100), "f8")], 4, False, 0),
    ("nanmax", np.nanmax, [((SIZE,), "f8")], 3, True, 1),  # fmax.reduce, isnan, any
    ("nanmean", np.nanmean, [((SIZE,), "f8")], 10, True, 1),
    ("nanstd_axis", lambda a: np.nanstd(a, axis=1), [((100, 100), "f8")], 17, True, 1),
    ("nanargmax", np.nanargmax, [((SIZE,), "f8")], 6, True, 1),
    ("norm", np.linalg.norm, [((SIZE,), "f8")], 2, True, 0, 0),
    ("norm_fro", np.linalg.norm, [((100, 100), "f8")], 2, True, 0, 0),
    ("norm_axis", lambda a: np.linalg.norm(a, axis=1), [((100, 100), "f8")], 4, True, 0),
    ("outer", np.outer, [((100,), "f8"), ((100,), "f8")], 1, False, 0),
    # norm, inner: a dot of two vectors takes the direct path once its inputs
    # are ready (the policy sees one output element, not the work; as a full
    # argmax, #26), so no task needs to leave the main thread
    ("inner", np.inner, [((SIZE,), "f8"), ((SIZE,), "f8")], 1, True, 0, 0),
    ("copy", np.copy, [((100, 100), "f8")], 1, False, 0),
    ("copy_F", lambda a: np.copy(a.T, order="K"), [((100, 100), "f8")], 1, False, 0),
    ("triu", np.triu, [((100, 100), "f8")], 1, False, 0),
    ("tril", lambda a: np.tril(a, -3), [((100, 100), "f8")], 1, False, 0),
    ("diag", np.diag, [((3000,), "f8")], 1, False, 0),
    ("meshgrid", np.meshgrid, [((100,), "f8"), ((50,), "f8")], 2, False, 0),
    ("allclose", np.allclose, [((SIZE,), "f8"), ((SIZE,), "f8")], 11, False, 1),
    ("array_equal", np.array_equal, [((SIZE,), "f8"), ((SIZE,), "f8")], 2, False, 1),
    # views and new arrays: no task, no sync of the pending argument
    ("imag", np.imag, [((SIZE,), "c16")], 0, False, 0),
    ("zeros_like", np.zeros_like, [((SIZE,), "f8")], 0, False, 0),
    ("full_like", lambda a: np.full_like(a, 2.5), [((SIZE,), "f8")], 0, False, 0),
    ("empty_like", lambda a: np.empty_like(a).shape, [((SIZE,), "f8")], 0, False, 0),
    ("diag_2d", np.diag, [((100, 100), "f8")], 0, False, 0),
    ("broadcast_arrays", np.broadcast_arrays, [((1, SIZE), "f8"), ((3, 1), "f8")], 0, False,
     0),
    ("split", lambda a: np.split(a, 5), [((SIZE,), "f8")], 0, False, 0),
    ("array_split", lambda a: np.array_split(a, 3), [((SIZE,), "f8")], 0, False, 0),
]


@pytest.mark.parametrize("call, inputs, ops, small_steps, syncs, min_tasks",
                         [c[1:] + (1,) * (len(c) == 6) for c in ASYNC], ids=[c[0] for c in ASYNC])
def test_async_path(det, call, inputs, ops, small_steps, syncs, min_tasks):
    pairs = [big(shape, dtype) for shape, dtype in inputs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = call(*(x for x, _ in pairs))
        got = assert_async(call, *(v for _, v in pairs), ops=ops, small_steps=small_steps,
                           syncs=syncs, min_tasks=min_tasks)
    same(got, want)
