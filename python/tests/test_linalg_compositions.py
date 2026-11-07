"""numpy.linalg functions NumPy composes of products, inverses and solves
(matrix_power, multi_dot, tensordot, tensorsolve, tensorinv, the
numpy.linalg aliases matmul/vecdot/outer/tensordot/diagonal/trace/
matrix_transpose) and numpy.fft's fftshift/ifftshift: ported without
NumPy's syncing asarray, bit-identical to NumPy (values, dtypes, layouts,
errors) and without the eager fallback."""

import numpy as np
import pytest

import viproc
from test_linalg import check, matrices

L = np.linalg
FLOATS = [np.float64, np.float32, np.complex128, np.complex64]


@pytest.mark.parametrize("dtype", FLOATS + [np.int64])
def test_matrix_power(dtype):
    a = matrices((6, 6), dtype) / 4
    for n in (0, 1, 2, 3, 4, 5, 13):
        check(lambda x, n=n: L.matrix_power(x, n), a)
    b = matrices((3, 5, 5), dtype) / 4
    check(lambda x: L.matrix_power(x, 6), b)
    if np.dtype(dtype).kind in "fc":
        check(lambda x: L.matrix_power(x, -3), a + 4 * np.eye(6, dtype=dtype))
    check(lambda x: L.matrix_power(x, 2.0), a)  # TypeError: exponent must be an integer
    check(lambda x: L.matrix_power(x, 2), matrices((3, 4), dtype))  # LinAlgError
    v = viproc.asarray(a)
    assert L.matrix_power(v, 1) is v  # NumPy returns its argument


@pytest.mark.parametrize("dtype", FLOATS)
def test_multi_dot(dtype):
    a, b, c, d = (matrices(s, dtype) for s in [(5, 8), (8, 3), (3, 9), (9, 4)])
    check(lambda x: L.multi_dot([x, b]), a)
    check(lambda x: L.multi_dot([x, b, c]), a)
    check(lambda x: L.multi_dot([x, b, c, d]), a)
    v, w = matrices((5,), dtype), matrices((4,), dtype)
    check(lambda x: L.multi_dot([x, a, b, c, d]), v)
    check(lambda x: L.multi_dot([x, a, b, c, d, w]), v)
    check(lambda x: L.multi_dot([x]), a)  # ValueError


@pytest.mark.parametrize("dtype", FLOATS)
def test_tensordot(dtype):
    a, b = matrices((3, 4, 5), dtype), matrices((4, 5, 6), dtype)
    check(lambda x: np.tensordot(x, b), a)
    check(lambda x: np.tensordot(x, b, axes=([1, 2], [0, 1])), a)
    check(lambda x: np.tensordot(x, b, axes=1), matrices((3, 4), dtype))
    check(lambda x: np.tensordot(x, b, axes=([-2], [0])), a)
    check(lambda x: L.tensordot(x, b, axes=2), a)
    check(lambda x: np.tensordot(x, b, axes=([0], [0])), a)  # shape-mismatch
    check(lambda x: np.tensordot(x, b, axes=([1, 1], [0, 1])), a)  # duplicate axes


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_tensorsolve_tensorinv(dtype):
    a = matrices((6, 2, 3), dtype) + 0  # prod(a.shape[1:]) == 6
    b = matrices((6,), dtype)
    check(lambda x: L.tensorsolve(x, b), a)
    a4, b4 = matrices((2, 3, 6), dtype), matrices((2, 3), dtype)
    check(lambda x: L.tensorsolve(x, b4, axes=(2,)), a4)
    check(lambda x: L.tensorsolve(x, b4, axes=(0,)), a4)
    check(lambda x: L.tensorsolve(x, b), matrices((6, 2, 4), dtype))  # LinAlgError
    t = matrices((4, 6, 8, 3), dtype)
    check(lambda x: L.tensorinv(x, ind=2), t)
    check(lambda x: L.tensorinv(x, ind=1), matrices((24, 4, 6), dtype))
    check(lambda x: L.tensorinv(x, ind=0), t)  # ValueError


@pytest.mark.parametrize("dtype", FLOATS)
def test_linalg_aliases(dtype):
    a, b = matrices((3, 4, 5), dtype), matrices((3, 5, 2), dtype)
    v, w = matrices((7,), dtype), matrices((5,), dtype)
    check(lambda x: L.matmul(x, b), a)
    check(lambda x: L.vecdot(x, a), a)
    check(lambda x: L.vecdot(x, a, axis=1), a)
    check(lambda x: L.outer(x, w), v)
    check(lambda x: L.outer(x, w), a)  # ValueError: one-dimensional
    check(lambda x: L.diagonal(x, offset=1), a)
    check(lambda x: L.trace(x), a)
    check(L.matrix_transpose, a)


@pytest.mark.parametrize("shape", [(8,), (7,), (4, 5), (3, 4, 6)])
def test_fftshift(shape):
    x = matrices(shape, np.complex128)
    for f in (np.fft.fftshift, np.fft.ifftshift):
        check(f, x)
        check(lambda a, f=f: f(a, axes=0), x)
        check(lambda a, f=f: f(a, axes=(-1,)), x)


def test_no_fallback():
    a = viproc.asarray(matrices((40, 40), np.float64) / 40)
    b = viproc.asarray(matrices((40,), np.float64))
    before = viproc._ndarray.fallback_calls
    L.matrix_power(a, 7)
    L.multi_dot([a, a, a, a])
    np.tensordot(a, a, axes=1)
    L.tensorsolve(a + 3 * np.eye(40), b)
    L.tensorinv(a.reshape(4, 10, 4, 10) + 0, ind=2)
    L.matmul(a, a)
    np.fft.fftshift(a)
    viproc.wait_all()
    assert viproc._ndarray.fallback_calls == before
