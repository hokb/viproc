"""numpy.linalg on NumPy's _umath_linalg gufuncs (hokb/viproc#25, step 6b):
det, slogdet, inv, solve, cholesky, eig, eigh, eigvals, eigvalsh, svd,
svdvals.

The code follows NumPy's numpy/linalg/_linalg.py (BSD-3-Clause): the same
argument checks (LinAlgError for shapes, raised at the call), the same
computation dtype (_commonType: double or cdouble, the result cast back to
single precision where the inputs were), the same gufunc and signature.
The gufunc runs as a task (_ufuncs.gufunc, vp_gufunc); NumPy's wrappers
turn the loops' FP invalid flag into LinAlgError with
errstate(invalid='call'); viproc's kernel checks the flag after each loop
call and fails the op with the same message, raised as AsyncLinAlgError (a
LinAlgError) at the next call into viproc and at the result's sync points
(error-handling.md). eig's and eigvals' _assert_finite runs in the kernel
too (VP_GUFUNC_CHECK_FINITE), before the loop. A handler returns _NOPE for
arguments it does not cover: NumPy then runs the call in the eager
fallback.
"""

import math

import numpy as np
from numpy.linalg import LinAlgError, _umath_linalg
from numpy.linalg._linalg import EighResult, EigResult, SlogdetResult, SVDResult

from . import _ndarray as _nd
from . import _ufuncs

_NOPE = _nd._NOPE
ndarray = _nd.ndarray

_D = np.dtype(np.float64)
_Z = np.dtype(np.complex128)
_REAL = {np.dtype(np.float32): np.dtype(np.float32), _D: _D,
         np.dtype(np.complex64): np.dtype(np.float32), _Z: _D}
_COMPLEX = {np.dtype(np.float32): np.dtype(np.complex64), _D: _Z,
            np.dtype(np.complex64): np.dtype(np.complex64), _Z: _Z}

_SINGULAR = "LinAlgError: Singular matrix"
_NONPOSDEF = "LinAlgError: Matrix is not positive definite"
_EIG = "LinAlgError: Eigenvalues did not converge"
_SVD = "LinAlgError: SVD did not converge"


def _arrays(*xs):
    """The arguments as viproc arrays (NumPy arrays and sequences imported as
    asarray imports them), or None if one is not covered."""
    out = []
    for x in xs:
        if type(x) is not ndarray:
            x = _nd.asarray(x)
            if type(x) is not ndarray:
                return None
        out.append(x)
    return out


def _common_type(*arrays):
    """NumPy's _commonType: (computation dtype, result dtype)."""
    single, is_complex = True, False
    for a in arrays:
        k = a.dtype.kind
        if k in "fc":
            is_complex = is_complex or k == "c"
            single = single and a.dtype.itemsize == (4 if k == "f" else 8)
        else:
            single = False
    result = np.dtype(np.float32) if single else _D
    if is_complex:
        return _Z, _COMPLEX[result]
    return _D, result


def _assert_stacked_2d(a):
    if a.ndim < 2:
        raise LinAlgError(f"{a.ndim}-dimensional array given. Array must be "
                          "at least two-dimensional")


def _assert_stacked_square(a):
    _assert_stacked_2d(a)
    m, n = a.shape[-2:]
    if m != n:
        raise LinAlgError("Last 2 dimensions of the array must be square")


_CHECK_FINITE = 2  # VP_GUFUNC_CHECK_FINITE
_SERIAL = 4  # VP_GUFUNC_SERIAL
# Matrices of at least this many elements run one LAPACK call at a time:
# OpenBLAS threads them itself, and several threaded calls at once
# oversubscribe its threads (vp_gufunc, VP_GUFUNC_SERIAL).
SERIAL_FROM = 64 * 64


def _call(gufunc, inputs, signature, invalid=None, label_sizes=None, check_finite=False):
    flags = _CHECK_FINITE if check_finite else 0
    if max(math.prod(x.shape[-2:]) for x in inputs) >= SERIAL_FROM:
        flags |= _SERIAL
    return _ufuncs.gufunc(gufunc, inputs, {"signature": signature}, label_sizes, invalid,
                          flags, _work(inputs[0]))


def _work(a):
    """A LAPACK factorization costs about m*n*min(m, n) per matrix."""
    m, n = a.shape[-2:] if a.ndim >= 2 else (1, a.shape[-1])
    batch = a.size // max(m * n, 1)
    return max(batch * m * n * min(m, n), 1)


def _cast(r, dtype):
    return r.astype(dtype, copy=False)


def _det(a):
    a = _arrays(a)
    if a is None:
        return _NOPE
    a, = a
    _assert_stacked_square(a)
    t, result_t = _common_type(a)
    r = _call(_umath_linalg.det, (a,), (t, t))
    return r if r is _NOPE else _cast(r, result_t)


def _slogdet(a):
    a = _arrays(a)
    if a is None:
        return _NOPE
    a, = a
    _assert_stacked_square(a)
    t, result_t = _common_type(a)
    r = _call(_umath_linalg.slogdet, (a,), (t, t, _D))
    if r is _NOPE:
        return _NOPE
    sign, logdet = r
    return SlogdetResult(_cast(sign, result_t), _cast(logdet, _REAL[result_t]))


def _inv(a):
    a = _arrays(a)
    if a is None:
        return _NOPE
    a, = a
    _assert_stacked_square(a)
    t, result_t = _common_type(a)
    r = _call(_umath_linalg.inv, (a,), (t, t), _SINGULAR)
    return r if r is _NOPE else _cast(r, result_t)


def _solve(a, b):
    ab = _arrays(a, b)
    if ab is None:
        return _NOPE
    a, b = ab
    _assert_stacked_square(a)
    t, result_t = _common_type(a, b)
    gufunc = _umath_linalg.solve1 if b.ndim == 1 else _umath_linalg.solve
    r = _call(gufunc, (a, b), (t, t, t), _SINGULAR)
    return r if r is _NOPE else _cast(r, result_t)


def _cholesky(a, /, *, upper=False):
    a = _arrays(a)
    if a is None:
        return _NOPE
    a, = a
    gufunc = _umath_linalg.cholesky_up if upper else _umath_linalg.cholesky_lo
    _assert_stacked_square(a)
    t, result_t = _common_type(a)
    r = _call(gufunc, (a,), (t, t), _NONPOSDEF)
    return r if r is _NOPE else _cast(r, result_t)


def _eigvals(a):
    a = _arrays(a)
    if a is None:
        return _NOPE
    a, = a
    _assert_stacked_square(a)
    t, result_t = _common_type(a)
    r = _call(_umath_linalg.eigvals, (a,), (t, _Z), _EIG, check_finite=True)
    return r if r is _NOPE else _cast(r, _COMPLEX[result_t])


def _eig(a):
    a = _arrays(a)
    if a is None:
        return _NOPE
    a, = a
    _assert_stacked_square(a)
    t, result_t = _common_type(a)
    r = _call(_umath_linalg.eig, (a,), (t, _Z, _Z), _EIG, check_finite=True)
    if r is _NOPE:
        return _NOPE
    w, vt = r
    c = _COMPLEX[result_t]
    return EigResult(_cast(w, c), _cast(vt, c))


def _uplo(UPLO):
    UPLO = UPLO.upper()
    if UPLO not in ("L", "U"):
        raise ValueError("UPLO argument must be 'L' or 'U'")
    return UPLO


def _eigvalsh(a, UPLO="L"):
    gufunc = _umath_linalg.eigvalsh_lo if _uplo(UPLO) == "L" else _umath_linalg.eigvalsh_up
    a = _arrays(a)
    if a is None:
        return _NOPE
    a, = a
    _assert_stacked_square(a)
    t, result_t = _common_type(a)
    r = _call(gufunc, (a,), (t, _D), _EIG)
    return r if r is _NOPE else _cast(r, _REAL[result_t])


def _eigh(a, UPLO="L"):
    UPLO = _uplo(UPLO)
    a = _arrays(a)
    if a is None:
        return _NOPE
    a, = a
    _assert_stacked_square(a)
    t, result_t = _common_type(a)
    gufunc = _umath_linalg.eigh_lo if UPLO == "L" else _umath_linalg.eigh_up
    r = _call(gufunc, (a,), (t, _D, t), _EIG)
    if r is _NOPE:
        return _NOPE
    w, vt = r
    return EighResult(_cast(w, _REAL[result_t]), _cast(vt, result_t))


def _svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    if hermitian:
        return _NOPE  # eigh, then argsort (step 7)
    a = _arrays(a)
    if a is None:
        return _NOPE
    a, = a
    _assert_stacked_2d(a)
    t, result_t = _common_type(a)
    m, n = a.shape[-2:]
    sizes = {"p": min(m, n)}
    if compute_uv:
        gufunc = _umath_linalg.svd_f if full_matrices else _umath_linalg.svd_s
        r = _call(gufunc, (a,), (t, t, _D, t), _SVD, sizes)
        if r is _NOPE:
            return _NOPE
        u, s, vh = r
        return SVDResult(_cast(u, result_t), _cast(s, _REAL[result_t]), _cast(vh, result_t))
    r = _call(_umath_linalg.svd, (a,), (t, _D), _SVD, sizes)
    return r if r is _NOPE else _cast(r, _REAL[result_t])


def _svdvals(x, /):
    return _svd(x, compute_uv=False, hermitian=False)


LINALG = {
    np.linalg.det: _det, np.linalg.slogdet: _slogdet, np.linalg.inv: _inv,
    np.linalg.solve: _solve, np.linalg.cholesky: _cholesky, np.linalg.eigvals: _eigvals,
    np.linalg.eig: _eig, np.linalg.eigvalsh: _eigvalsh, np.linalg.eigh: _eigh,
    np.linalg.svd: _svd, np.linalg.svdvals: _svdvals,
}
