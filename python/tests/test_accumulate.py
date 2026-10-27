"""ufunc.accumulate (np.cumsum, np.cumprod, np.nancumsum, np.nancumprod) and
ufunc.outer (_ufuncs.py, hokb/viproc#25 step 5): bit-identical to NumPy
(values, dtype, shape and layout) over ufuncs, dtypes, layouts, axes and
dtype=, and on the asynchronous path."""

import warnings

import numpy as np
import pytest

import viproc
from async_path import SIZE, assert_async, counters
from test_compositions import DTYPES, big, check, det, same, sample  # noqa: F401 (det: fixture)
from test_joining import path  # noqa: F401 (fixture)

SHAPES = [(7,), (4, 5), (3, 4, 2)]

UFUNCS = [np.add, np.multiply, np.subtract, np.maximum, np.minimum, np.fmax, np.logical_and,
          np.logical_or, np.bitwise_xor, np.divide, np.true_divide, np.hypot, np.arctan2]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_accumulate(shape, dtype, path):
    x = sample(shape, dtype)
    for ufunc in UFUNCS:
        for axis in range(-len(shape), len(shape)):
            check(lambda a: ufunc.accumulate(a, axis=axis), x)
        check(lambda a: ufunc.accumulate(a), x)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cumsum_cumprod(shape, dtype, path):
    x = sample(shape, dtype)
    for f in (np.cumsum, np.cumprod, np.nancumsum, np.nancumprod):
        check(f, x)
        for axis in range(-len(shape), len(shape)):
            check(lambda a: f(a, axis=axis), x)
        for d in (np.float64, np.float32, np.int64, np.int32, np.complex64, np.bool_):
            check(lambda a: f(a, axis=0, dtype=d), x)
    check(lambda a: a.cumsum(axis=-1), x)
    check(lambda a: a.cumprod(), x)
    check(lambda a: np.add.accumulate(a, dtype=np.float32), x)
    check(lambda a: np.multiply.accumulate(a, 0, np.int64), x)


def test_accumulate_errors_and_edges(path):
    x = sample((4, 5), np.float64)
    check(lambda a: np.cumsum(a, axis=2), x)  # AxisError
    check(lambda a: np.add.accumulate(a, axis=None), x)
    check(lambda a: np.cumsum(a[:0]), x)
    check(lambda a: np.cumsum(a[:, :0], axis=0), x)
    check(lambda a: np.cumsum(a[0, 0]), x)  # 0-d
    check(lambda a: np.add.accumulate(a[0, 0]), x)
    check(lambda a: np.cumsum(a, dtype="U3"), x)
    check(lambda a: np.isnan.accumulate(a), x)  # unary: ValueError
    out, vout = np.zeros((4, 5)), viproc.asarray(np.zeros((4, 5)))  # out=: fallback
    np.cumsum(x, axis=0, out=out)
    np.cumsum(viproc.asarray(x), axis=0, out=vout)
    same(vout, out)


@pytest.mark.parametrize("dtype", DTYPES)
def test_outer(dtype, path):
    x, y = sample((5,), dtype), sample((3, 2), dtype)
    for ufunc in (np.add, np.multiply, np.subtract, np.maximum, np.equal, np.logical_xor):
        check(lambda a, b: ufunc.outer(a, b), x, y)
        check(lambda a, b: ufunc.outer(b, a), x, y)
        check(lambda a: ufunc.outer(a, 2), x)
        check(lambda a: ufunc.outer(a, a.tolist()), x)
        check(lambda a: ufunc.outer(np.arange(3), a), x)
    check(lambda a, b: np.add.outer(a, b, dtype=np.complex128), x, y)
    check(lambda a: np.outer(a, a), x)
    check(lambda a: np.matmul.outer(a, a), x)  # gufunc: NumPy's error


ASYNC = [  # name, call, input (shape, dtype), tasks
    ("cumsum", np.cumsum, ((SIZE,), "f8"), 1),
    ("cumsum_axis0", lambda a: np.cumsum(a, axis=0), ((100, 60), "f8"), 1),
    ("cumsum_axis1", lambda a: a.cumsum(axis=1), ((100, 60), "f8"), 1),
    ("cumsum_ravel", lambda a: np.cumsum(a.T), ((100, 60), "f8"), 2),  # C copy, then cumsum
    ("cumsum_int32", np.cumsum, ((SIZE,), "i4"), 2),  # cast to int64 first
    ("cumsum_bool", lambda a: np.cumsum(a > 0), ((SIZE,), "f8"), 3),  # >, cast, cumsum
    ("cumprod", lambda a: a.cumprod(axis=0), ((100, 60), "c16"), 1),
    ("nancumsum", np.nancumsum, ((SIZE,), "f8"), 3),  # isnan, where, cumsum
    ("nancumsum_int", np.nancumsum, ((SIZE,), "i8"), 1),
    ("accumulate", lambda a: np.maximum.accumulate(a, axis=1), ((100, 60), "f4"), 1),
    ("accumulate_dtype", lambda a: np.add.accumulate(a, dtype=np.float32), ((SIZE,), "f8"), 2),
    ("outer", lambda a: np.multiply.outer(a[:80], a[:70]), ((SIZE,), "f8"), 1),
]


@pytest.mark.parametrize("call, inp, tasks", [c[1:] for c in ASYNC], ids=[c[0] for c in ASYNC])
def test_async_path(det, call, inp, tasks):
    x, v = big(*inp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = call(x)
        got = assert_async(call, v, ops=tasks)
    same(got, want)


def test_small_ready_inputs_run_in_numpy(det):
    """Small ready inputs: NumPy's function right away, no task, no fallback."""
    x = sample((100,), np.float64, special=False)
    v = viproc.asarray(x)
    viproc.wait_all()
    before = counters()
    r = np.cumsum(v)
    after = counters()
    assert type(r) is viproc.ndarray
    assert after["tasks"] == before["tasks"] and after["fallbacks"] == before["fallbacks"]
    same(r, np.cumsum(x))
