"""Generalized ufuncs (vecdot, matvec, vecmat; matmul with casts, dtype=,
out=, axes=) on NumPy's gufunc loops (hokb/viproc#25 step 6a): bit-identical
to NumPy (values, dtypes, layouts, errors) and on the asynchronous path."""

import warnings

import numpy as np
import pytest

import viproc
from async_path import assert_async, pending
from test_compositions import DTYPES, check, det, same, sample  # noqa: F401 (det: fixture)
from test_joining import path  # noqa: F401 (fixture)

rng = np.random.default_rng(6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_vector_gufuncs(dtype, path):
    a = sample((5, 4, 3), dtype, special=False)
    b = sample((5, 3), dtype, special=False)
    m = sample((3, 4), dtype, special=False)
    check(np.vecdot, a, a)
    check(lambda x, y: np.vecdot(x, y), a, a[0])
    check(lambda x, y: np.vecdot(y, x), a, b[:, None, :])
    check(lambda x, y: np.matvec(x, y), a, b[:, None, :])
    check(lambda x, y: np.vecmat(x[:, :, 0], y), a, m)
    check(lambda x: np.vecdot(x, x, axis=0), a)
    check(lambda x: np.vecdot(x, x, axis=-2, keepdims=True), a)
    check(lambda x: np.vecdot(x, x, axes=[(1,), (1,), ()]), a)
    check(lambda x: np.matvec(x, x[..., 0], axes=[(1, 2), (1,), (0,)]), a)
    check(lambda x: np.vecdot(x[0, 0], x[0, 0]), a)  # 0-d result: NumPy's scalar
    check(lambda x: np.vecdot(x[:, :0], x[:, :0]), a)  # empty batch
    check(lambda x: np.vecdot(x[..., :0], x[..., :0]), a)  # empty core: zeros
    check(lambda x, y: np.vecdot(x, y), a, a[:, :2])  # NumPy's ValueError


@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_casts(dtype, path):
    a = sample((5, 4, 3), dtype, special=False)
    check(lambda x: np.matmul(x, x.transpose(0, 2, 1).astype(np.float32)), a)
    check(lambda x: np.matmul(x, x[0, 0]), a)  # 1-d operand: missing flexible dims
    check(lambda x: np.matmul(x[0, 0], x.transpose(0, 2, 1)), a)
    check(lambda x: np.matmul(x.T, x, axes=[(0, 1), (1, 2), (0, 1)]), a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ComplexWarning
        for d in (np.float32, np.complex128, np.int64):
            check(lambda x: np.matmul(x, x.transpose(0, 2, 1), dtype=d), a)
    check(lambda x: np.matmul(x, 2), a)  # NumPy's ValueError for a scalar


@pytest.mark.parametrize("dtype", [np.float64, np.complex64, np.int32])
def test_out(dtype, path):
    a = sample((5, 4, 3), dtype, special=False)
    other = np.complex64 if a.dtype.kind == "c" else np.float32
    for f, args, shape in [(np.vecdot, (a, a), (5, 4)),
                           (np.matvec, (a, a[:, 0]), (5, 4)),
                           (np.matmul, (a, a.transpose(0, 2, 1).astype(other)), None)]:
        want = f(*args)
        shape = want.shape if shape is None else shape
        for order in ("C", "F"):
            o = np.zeros(shape, want.dtype, order=order)
            vo = viproc.asarray(o)
            r = f(*args, out=o)
            vr = f(*[viproc.asarray(x) for x in args], out=vo)
            assert r is o and vr is vo
            same(vo, o)
            vr = f(*[viproc.asarray(x) for x in args], out=(vo,))
            assert vr is vo
            same(vo, o)
    # out= aliasing an input, of another dtype, through axes=
    x = viproc.asarray(a)
    o = np.vecdot(a, a)
    vo = np.vecdot(x, x)
    np.vecdot(o[:, :, None], o[:, :, None], out=o)
    np.vecdot(vo[:, :, None], vo[:, :, None], out=vo)
    same(vo, o)
    o32 = np.zeros((5, 4), np.complex128)
    vo32 = viproc.asarray(o32)
    np.vecdot(a, a, out=o32)
    np.vecdot(x, x, out=vo32)
    same(vo32, o32)
    o = np.zeros((4, 5), a.dtype)
    vo = viproc.asarray(o)
    np.vecdot(a, a, axes=[(2,), (2,), ()], out=o.T)
    np.vecdot(x, x, axes=[(2,), (2,), ()], out=vo.T)
    same(vo, o)


def test_larger(path):
    """Sizes and layouts NumPy hands to BLAS (matmul) and long batches."""
    a = sample((300, 40, 30), np.float64, special=False)
    check(np.vecdot, a, a)
    check(lambda x: np.matvec(x, x[:, :, 0].astype(np.float32)), a.transpose(0, 2, 1))
    check(lambda x: np.matmul(x, x.astype(np.float32).transpose(0, 2, 1)), a)
    check(lambda x: np.vecmat(x[:, :, 0], x), a)


ASYNC = [  # name, call, inputs
    ("vecdot", lambda x, y: np.vecdot(x, y), [(100, 50), (100, 50)]),
    ("vecdot_bcast", lambda x, y: np.vecdot(x, y), [(100, 50), (50,)]),
    ("vecdot_axis", lambda x, y: np.vecdot(x, y, axis=0), [(100, 50), (100, 50)]),
    ("matvec", lambda x, y: np.matvec(x, y), [(10, 30, 20), (10, 20)]),
    ("vecmat", lambda x, y: np.vecmat(x, y), [(10, 30), (10, 30, 20)]),
    ("matmul_dtype", lambda x, y: np.matmul(x, y, dtype=np.float32), [(60, 70), (70, 80)]),
]


@pytest.mark.parametrize("name,call,shapes", ASYNC, ids=[c[0] for c in ASYNC])
def test_async(name, call, shapes, det):
    xs = [sample(s, np.float64, special=False) for s in shapes]
    vs = [pending(viproc.asarray(x)) for x in xs]
    want = call(*xs)
    # inputs cast to the loop dtype first (matmul_dtype): one task each
    ops = 1 + (len(xs) if name == "matmul_dtype" else 0)
    got = assert_async(call, *vs, ops=ops)
    same(got, want)


def test_async_mixed_matmul(det):
    x = sample((60, 70), np.float64, special=False)
    y = sample((70, 80), np.float32, special=False)
    vx, vy = pending(viproc.asarray(x)), pending(viproc.asarray(y))
    got = assert_async(np.matmul, vx, vy, ops=2)  # the cast of y, the matmul
    same(got, np.matmul(x, y))


def test_async_out(det):
    x = sample((100, 50), np.float64, special=False)
    o = np.zeros(100)
    vx, vo = pending(viproc.asarray(x)), pending(viproc.asarray(o))
    got = assert_async(lambda: np.vecdot(vx, vx, out=vo), ops=1)
    assert got is vo
    same(vo, np.vecdot(x, x, out=o))
