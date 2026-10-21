"""where= on ufunc calls, ufunc.at, ufunc.reduceat and reductions into out=
(hokb/viproc#25 step 5c): bit-identical to NumPy (values, dtypes, layouts,
errors) and on the asynchronous path."""

import warnings

import numpy as np
import pytest

import viproc
from async_path import SIZE, assert_async
from test_compositions import DTYPES, big, check, det, same, sample  # noqa: F401 (det: fixture)
from test_joining import path  # noqa: F401 (fixture)

rng = np.random.default_rng(5)
SHAPES = [(7,), (4, 5), (3, 4, 2), (3000,)]


def both(fn, *arrays):
    """fn(*arrays) mutates its first argument: on NumPy copies and on viproc
    arrays, the first one in each layout; same results or the same error
    type (AsyncError subclasses NumPy's)."""
    base = arrays[0]
    rest = arrays[1:]
    for key in [(), (slice(None, None, -1),), "t"]:
        if key == "t" and base.ndim < 2:
            continue
        x = np.array(base)
        v = viproc.asarray(np.array(base))
        x, v = (x.T, v.T) if key == "t" else (x[key], v[key])
        vr = [viproc.asarray(a) if isinstance(a, np.ndarray) else a for a in rest]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fn(x, *rest)
            except Exception as e:  # noqa: BLE001
                with pytest.raises(type(e)):
                    fn(v, *vr)
                    viproc.wait_all()
                continue
            fn(v, *vr)
        same(v, x)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_where(shape, dtype, path):
    x, y = sample(shape, dtype), sample(shape, dtype)
    mask = rng.random(shape) > 0.4
    out = sample(shape, dtype)
    for f in (np.add, np.multiply, np.maximum, np.arctan2, np.power):
        if f in (np.arctan2,) and np.dtype(dtype).kind in "c":
            continue
        both(lambda o, a, b, m: f(a, b, out=o, where=m), out, x, y, mask)
        both(lambda o, a, m: f(a, 2, out=o, where=m), out, x, mask)
        both(lambda o, a, m: f(a, a[..., :1], out=o, where=m[..., :1]), out, x, mask)
    for f in (np.sin, np.exp, np.sqrt, np.negative, np.absolute):
        if dtype == np.bool_ and f is np.negative:
            continue
        both(lambda o, a, m: f(a, out=o, where=m), out, x, mask)
        both(lambda o, a: f(a, out=o, where=False), out, x)
        both(lambda o, a: f(a, out=o, where=True), out, x)
    # cast into out with where=
    o64 = sample(shape, np.complex128)
    both(lambda o, a, m: np.add(a, 1, out=o, where=m), o64, x, mask)


def test_where_without_out(path):
    """Without out= NumPy leaves masked-off elements undefined: only the
    selected ones and the result's dtype and layout are compared."""
    x = sample((60, 50), np.float64)
    m = rng.random((60, 50)) > 0.5
    for layout in (lambda a: a, lambda a: a.T, lambda a: a[::2, ::-1]):
        with warnings.catch_warnings(record=True) as want_w:
            warnings.simplefilter("always")
            want = np.sin(layout(x), where=layout(m))
        with warnings.catch_warnings(record=True) as got_w:
            warnings.simplefilter("always")
            got = np.sin(layout(viproc.asarray(x)), where=layout(viproc.asarray(m)))
        assert [str(w.message) for w in got_w] == [str(w.message) for w in want_w
                                                   if "encountered" not in str(w.message)]
        assert got_w[0].filename == __file__
        g = np.asarray(got)
        assert g.dtype == want.dtype and g.shape == want.shape and g.strides == want.strides
        assert np.array_equal(g[layout(m)], want[layout(m)], equal_nan=True)


@pytest.mark.parametrize("shape", [(10,), (10, 3), (10, 2, 3)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_at(shape, dtype, path):
    a = sample(shape, dtype)
    n = shape[0]
    idx = rng.integers(-n, n, 50)
    vals = sample((50,) + shape[1:], dtype)
    for f in (np.add, np.subtract, np.multiply, np.maximum, np.minimum, np.fmax, np.power,
              np.true_divide, np.bitwise_xor, np.logical_and):
        both(lambda t, v: f.at(t, idx, v), a, vals)
        both(lambda t: f.at(t, idx, 3), a)
        both(lambda t: f.at(t, idx.tolist(), 2.5), a)
        both(lambda t, v: f.at(t, (idx,), v[:1]), a, vals)
        both(lambda t: f.at(t, 4, 2), a)
    for f in (np.negative, np.absolute, np.sqrt, np.logical_not):
        both(lambda t: f.at(t, idx), a)
    both(lambda t: np.add.at(t, idx.astype(np.int32), 1), a)
    both(lambda t: np.add.at(t, [n], 1), a)  # IndexError, nothing changed
    both(lambda t: np.add.at(t, idx, np.ones(3)), a)  # shapes do not broadcast


def test_at_out_of_bounds_changes_nothing():
    a = viproc.asarray(np.zeros((5, 2)))  # 2-d: not NumPy's indexed loop
    idx = viproc.asarray(np.array([1, 2, 9]))
    with pytest.raises(IndexError):
        np.add.at(a, idx, 1.0)
        viproc.wait_all()
    with pytest.raises(IndexError):  # a failed array stays failed (error-handling.md)
        a.numpy()


@pytest.mark.parametrize("shape", SHAPES[:3] + [(40, 30)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_reduceat(shape, dtype, path):
    x = sample(shape, dtype)
    for axis in range(len(shape)):
        n = shape[axis]
        for ind in ([0], [0, n - 1], [n - 1, 0, 1], list(range(n)), [1, 1, 3 % n, 0],
                    rng.integers(0, n, 7).tolist()):
            for f in (np.add, np.multiply, np.maximum, np.minimum, np.logical_or):
                check(lambda a: f.reduceat(a, ind, axis=axis), x)
            check(lambda a: np.add.reduceat(a, ind, axis, np.float32), x)
            check(lambda a: np.add.reduceat(a, np.array(ind, np.int32), axis=axis), x)
    check(lambda a: np.add.reduceat(a, [shape[0]]), x)  # NumPy's IndexError
    check(lambda a: np.add.reduceat(a, [-1]), x)
    check(lambda a: np.add.reduceat(a, []), x)


@pytest.mark.parametrize("dtype", DTYPES)
def test_reduce_into_out(dtype, path):
    x = sample((40, 30), dtype)
    acc = np.dtype(np.int64) if np.dtype(dtype).kind in "bi" else np.dtype(dtype)
    for axis, shape in [(0, (30,)), (1, (40,)), (None, ())]:
        for f, d in [(np.sum, acc), (np.prod, acc), (np.max, np.dtype(dtype)),
                     (np.min, np.dtype(dtype))]:
            for order in ("C", "F"):
                out = np.zeros(shape, d, order=order)
                vout = viproc.asarray(out)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = f(x, axis=axis, out=out)
                    vr = f(viproc.asarray(x), axis=axis, out=vout)
                assert vr is vout and r is out
                same(vout, out)
        out = np.zeros((40, 1), acc)
        vout = viproc.asarray(out)
        np.add.reduce(x, axis=1, keepdims=True, out=out)
        assert np.add.reduce(viproc.asarray(x), axis=1, keepdims=True, out=vout) is vout
        same(vout, out)
    # out= of another dtype or layout: NumPy runs it
    out = np.zeros(30, np.float32)
    vout = viproc.asarray(out)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            np.sum(x, axis=0, out=out)
        except TypeError:
            with pytest.raises(TypeError):
                np.sum(viproc.asarray(x), axis=0, out=vout)
        else:
            np.sum(viproc.asarray(x), axis=0, out=vout)
            same(vout, out)


def test_reduce_into_view_of_input():
    x = sample((40, 30), np.float64)
    v = viproc.asarray(x)
    np.add.reduce(x, axis=0, out=x[0])
    np.add.reduce(v, axis=0, out=v[0])
    same(v, x)


ASYNC = [  # name, call(v) -> result, input (shape, dtype), tasks
    ("where_out", lambda a, o, m: np.sin(a, out=o, where=m), ((SIZE,), "f8"), 1),
    ("where_new", lambda a, o, m: np.add(a, 1.0, where=m), ((SIZE,), "f8"), 1),
    ("where_cast", lambda a, o, m: np.multiply(a, 2, out=o, where=m, casting="unsafe"),
     ((SIZE,), "i8"), 1),
]


@pytest.mark.parametrize("call, inp, tasks", [c[1:] for c in ASYNC], ids=[c[0] for c in ASYNC])
def test_where_async_path(det, call, inp, tasks):
    x, v = big(*inp)
    m = rng.random(inp[0]) > 0.5
    o = np.zeros(inp[0], np.float64)
    vo, vm = viproc.asarray(o), viproc.asarray(m)
    viproc.wait_all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = call(x, o, m)
        got = assert_async(call, v, vo, vm, ops=tasks)
    if want is o:
        assert got is vo
        same(vo, o)
    else:
        assert np.array_equal(np.asarray(got)[m], want[m])


def test_at_async_path(det):
    a = np.zeros((1000, 8))
    va = viproc.asarray(a)
    idx = rng.integers(0, 1000, SIZE)
    x, v = big((SIZE, 8), np.float64)
    vidx = viproc.asarray(idx)
    viproc.wait_all()
    np.add.at(a, idx, x)
    assert assert_async(lambda: np.add.at(va, vidx, v), ops=1) is None
    same(va, a)


def test_reduceat_async_path(det):
    x, v = big((100, 60), np.float64)
    ind = [0, 7, 7, 30, 59]
    same(assert_async(lambda a: np.add.reduceat(a, ind, axis=1), v), np.add.reduceat(x, ind, 1))
    xi, vi = big((SIZE,), np.int32)  # cast to int64 first
    same(assert_async(lambda a: np.add.reduceat(a, ind), vi, ops=2), np.add.reduceat(xi, ind))


def test_reduce_out_async_path(det):
    x, v = big((100, 60), np.float64)
    out = np.zeros(60)
    vout = viproc.asarray(out)
    viproc.wait_all()
    np.sum(x, axis=0, out=out)
    assert assert_async(lambda a: np.sum(a, axis=0, out=vout), v) is vout
    same(vout, out)
