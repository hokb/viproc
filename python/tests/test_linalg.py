"""numpy.linalg on NumPy's _umath_linalg gufuncs (hokb/viproc#25 step 6b):
bit-identical to NumPy (values, dtypes, layouts, errors) and on the
asynchronous path; LinAlgError from a loop raised asynchronously."""

import numpy as np
import pytest

import viproc
from async_path import assert_async, pending
from test_compositions import DTYPES, det, same  # noqa: F401 (det: fixture)
from test_compositions import check as _check
from test_joining import path  # noqa: F401 (fixture)

L = np.linalg
rng = np.random.default_rng(7)


def check(fn, *arrays):
    """test_compositions.check, with errors of the asynchronous op raised
    inside the call (wait_all), where NumPy raises them."""
    def waited(*args):
        r = fn(*args)
        viproc.wait_all()
        return r
    _check(waited, *arrays)


def matrices(shape, dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return rng.random(shape) > 0.5
    a = rng.standard_normal(shape) * 3
    if dtype.kind == "c":
        a = a + 1j * rng.standard_normal(shape)
    return a.astype(dtype)


def hermitian_pd(a):
    n = a.shape[-1]
    return a @ np.conj(np.swapaxes(a, -1, -2)) + n * np.eye(n, dtype=a.dtype)


SQUARE = [(4, 4), (3, 5, 5), (2, 2, 3, 3), (1, 1), (0, 3, 3), (2, 0, 0)]


@pytest.mark.parametrize("shape", SQUARE)
@pytest.mark.parametrize("dtype", DTYPES)
def test_square(shape, dtype, path):
    a = matrices(shape, dtype)
    for f in (L.det, L.slogdet, L.inv, L.eigvals, L.eig, L.eigvalsh, L.eigh, L.svdvals):
        check(f, a)
    check(lambda x: L.eigh(x, UPLO="u"), a)
    check(lambda x: L.eigvalsh(x, "U"), a)
    if a.size and np.dtype(dtype).kind != "b":
        h = hermitian_pd(a)
        check(L.cholesky, h)
        check(lambda x: L.cholesky(x, upper=True), h)
        check(L.eigh, h)


@pytest.mark.parametrize("shape", [(4, 6), (6, 4), (3, 5, 2), (2, 0, 3), (2, 3, 0)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_svd(shape, dtype, path):
    a = matrices(shape, dtype)
    check(L.svd, a)
    check(lambda x: L.svd(x, full_matrices=False), a)
    check(lambda x: L.svd(x, compute_uv=False), a)
    check(L.svdvals, a)


@pytest.mark.parametrize("dtype", DTYPES)
def test_solve(dtype, path):
    a = matrices((3, 4, 4), dtype)
    for b in (matrices((3, 4, 2), np.float64), matrices((4,), np.float32),
              matrices((4, 3), dtype), matrices((3, 4), dtype)):
        check(L.solve, a, b)
    check(L.solve, a[0], matrices((2, 4, 1), dtype))


def test_errors(path):
    check(L.inv, np.ones(3))  # NumPy's LinAlgError at the call
    check(L.inv, np.ones((3, 4)))
    check(L.solve, np.ones((3, 4)), np.ones(3))
    check(lambda x: L.eigh(x, UPLO="X"), np.eye(3))  # ValueError
    check(L.svd, np.ones(3))
    # errors from the loop: NumPy's LinAlgError, raised asynchronously
    check(L.inv, np.ones((2, 3, 3)))
    check(L.solve, np.zeros((3, 3)), np.ones(3))
    check(L.cholesky, -np.eye(3))
    check(L.eig, np.full((3, 3), np.nan))
    check(L.eigvals, np.array([[1.0, np.inf], [0, 1]]))
    check(L.det, np.full((3, 3), np.nan))  # no error: NaN


ASYNC = [  # name, call, matrices, tasks
    ("det", L.det, (60, 10, 10), 1),
    ("slogdet", L.slogdet, (60, 10, 10), 1),
    ("inv", L.inv, (60, 10, 10), 1),
    ("solve", lambda a: L.solve(a, a[..., :4]), (60, 10, 10), 1),
    ("cholesky", lambda a: L.cholesky(hermitian_pd(a)), (60, 10, 10), 4),  # conj, @, +
    ("eigvals", L.eigvals, (60, 10, 10), 1),
    ("eig", L.eig, (60, 10, 10), 1),
    ("eigh", L.eigh, (60, 10, 10), 1),
    ("eigvalsh", L.eigvalsh, (60, 10, 10), 1),
    ("svd", L.svd, (60, 10, 8), 1),
    ("svdvals", L.svdvals, (60, 8, 10), 1),
    # single precision: the input cast to double first, the results cast back
    ("inv_f4", L.inv, (60, 10, 10, "f4"), 3),
    ("solve_f4", lambda a: L.solve(a, a[..., :4]), (60, 10, 10, "f4"), 4),
    ("svd_c8", L.svd, (60, 10, 8, "c8"), 5),
]


@pytest.mark.parametrize("call,shape,ops", [c[1:] for c in ASYNC], ids=[c[0] for c in ASYNC])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_async(call, shape, ops, dtype, det):
    if isinstance(shape[-1], str):
        if dtype is np.complex128:
            pytest.skip("the dtype is the case's")
        shape, dtype = shape[:-1], shape[-1]
    a = matrices(shape, dtype)
    v = pending(viproc.asarray(a))
    got = assert_async(call, v, ops=ops)
    same(got, call(a))


def test_async_singular(det):
    """A singular matrix fails the task on a worker: no fallback, no sync at
    the call; AsyncLinAlgError (a LinAlgError) at the next check or the sync
    point, whichever comes first."""
    v = pending(viproc.asarray(np.ones((60, 10, 10))))
    before = viproc.offload_stats()["tasks"], viproc._ndarray.fallback_calls
    after = caught = None
    try:
        r = L.inv(v)
        after = viproc.offload_stats()["tasks"], viproc._ndarray.fallback_calls
        r.numpy()
    except viproc.AsyncLinAlgError as e:
        caught = e
    assert isinstance(caught, L.LinAlgError) and "Singular matrix" in str(caught)
    assert after is None or (after[0] - before[0] == 1 and after[1] == before[1])
