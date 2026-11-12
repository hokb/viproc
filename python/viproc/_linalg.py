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
import operator

import numpy as np
from numpy.linalg import LinAlgError, _umath_linalg
from numpy.linalg import _linalg as _np_linalg
from numpy.linalg._linalg import EighResult, EigResult, SlogdetResult, SVDResult

from . import _ndarray as _nd
from . import _blas, _ufuncs, _viproc

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
# Without control over the BLAS threads (no OpenBLAS found, `_blas.single()`
# false), matrices of at least this many elements run one LAPACK call at a
# time: OpenBLAS threads them itself, and several threaded calls at once
# oversubscribe its threads (vp_gufunc, VP_GUFUNC_SERIAL). With one BLAS
# thread (the normal case, `_blas.apply`) the workers parallelize, at every
# optimization level.
SERIAL_FROM = 64 * 64


def _call(gufunc, inputs, signature, invalid=None, label_sizes=None, check_finite=False):
    flags = _CHECK_FINITE if check_finite else 0
    if not _blas.single() and max(math.prod(x.shape[-2:]) for x in inputs) >= SERIAL_FROM:
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
    a = _arrays(a)
    if a is None:
        return _NOPE
    a, = a
    if hermitian:  # NumPy's code: eigh, the values sorted by magnitude
        if compute_uv:
            r = _eigh(a)
            if r is _NOPE:
                return _NOPE
            s, u = r
            sgn = np.copysign(1.0, s)
            s = abs(s)
            sidx = np.argsort(s)[..., ::-1]
            sgn = np.take_along_axis(sgn, sidx, axis=-1)
            s = np.take_along_axis(s, sidx, axis=-1)
            u = np.take_along_axis(u, sidx[..., None, :], axis=-1)
            vt = np.swapaxes(u * sgn[..., None, :], -1, -2).conjugate()
            return SVDResult(u, s, vt)
        s = _eigvalsh(a)
        if s is _NOPE:
            return _NOPE
        return np.sort(abs(s))[..., ::-1]
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


# --- numpy.linalg functions NumPy composes of others ---------------------------
# NumPy's code (numpy/linalg/_linalg.py, numpy/_core/numeric.py), ported
# without the asarray/asanyarray that would sync: the matrix products,
# inverses and solves they make are viproc's ops (on the OpenCL device at
# O4, docs/design/opencl.md). Errors are NumPy's: the same checks, in the
# same order, with the same messages.

def _one(a):
    a = _arrays(a)
    return None if a is None else a[0]


def _linalg_matmul(x1, x2, /):
    return np.matmul(x1, x2)


def _linalg_vecdot(x1, x2, /, *, axis=-1):
    return np.vecdot(x1, x2, axis=axis)


def _linalg_outer(x1, x2, /):
    xs = _arrays(x1, x2)
    if xs is None:
        return _NOPE
    x1, x2 = xs
    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError(
            "Input arrays must be one-dimensional, but they are "
            f"{x1.ndim=} and {x2.ndim=}."
        )
    return np.outer(x1, x2, out=None)


def _linalg_tensordot(x1, x2, /, *, axes=2):
    return np.tensordot(x1, x2, axes=axes)


def _linalg_diagonal(x, /, *, offset=0):
    return np.diagonal(x, offset, axis1=-2, axis2=-1)


def _linalg_trace(x, /, *, offset=0, dtype=None):
    return np.trace(x, offset, axis1=-2, axis2=-1, dtype=dtype)


def _linalg_matrix_transpose(x, /):
    return np.matrix_transpose(x)


def _tensordot(a, b, axes=2):
    """np.tensordot (numpy/_core/numeric.py): transposes and reshapes into
    one matrix product (np.dot)."""
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    if len(set(axes_a)) != len(axes_a):
        raise ValueError("duplicate axes are not allowed in tensordot")
    if len(set(axes_b)) != len(axes_b):
        raise ValueError("duplicate axes are not allowed in tensordot")

    ab = _arrays(a, b)
    if ab is None:
        return _NOPE
    a, b = ab
    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = math.prod(as_[axis] for axis in axes_a)
    newshape_a = (math.prod(as_[ax] for ax in notin), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = math.prod(bs[axis] for axis in axes_b)
    newshape_b = (N2, math.prod(bs[ax] for ax in notin))
    oldb = [bs[axis] for axis in notin]

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    res = np.dot(at, bt)
    return res.reshape(olda + oldb)


def _matrix_power(a, n):
    a = _one(a)
    if a is None or a.dtype == object:
        return _NOPE
    _assert_stacked_square(a)
    try:
        n = operator.index(n)
    except TypeError as e:
        raise TypeError("exponent must be an integer") from e
    fmatmul = np.matmul
    if n == 0:
        a = np.empty_like(a)
        a[...] = np.eye(a.shape[-2], dtype=a.dtype)
        return a
    elif n < 0:
        a = np.linalg.inv(a)
        n = abs(n)
    if n == 1:
        return a
    elif n == 2:
        return fmatmul(a, a)
    elif n == 3:
        return fmatmul(fmatmul(a, a), a)
    z = result = None
    while n > 0:
        z = a if z is None else fmatmul(z, z)
        n, bit = divmod(n, 2)
        if bit:
            result = z if result is None else fmatmul(result, z)
    return result


def _multi_dot(arrays, *, out=None):
    n = len(arrays)
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    elif n == 2:
        return np.dot(arrays[0], arrays[1], out=out)
    arrays = _arrays(*arrays)
    if arrays is None:
        return _NOPE
    ndim_first, ndim_last = arrays[0].ndim, arrays[-1].ndim
    if arrays[0].ndim == 1:
        arrays[0] = np.atleast_2d(arrays[0])
    if arrays[-1].ndim == 1:
        arrays[-1] = np.atleast_2d(arrays[-1]).T
    _np_linalg._assert_2d(*arrays)
    # NumPy's helpers only read shapes and call np.dot: viproc's products
    if n == 3:
        result = _np_linalg._multi_dot_three(arrays[0], arrays[1], arrays[2], out=out)
    else:
        order = _np_linalg._multi_dot_matrix_chain_order(arrays)
        result = _np_linalg._multi_dot(arrays, order, 0, n - 1, out=out)
    if ndim_first == 1 and ndim_last == 1:
        return result[0, 0]
    elif ndim_first == 1 or ndim_last == 1:
        return result.ravel()
    else:
        return result


def _tensorsolve(a, b, axes=None):
    ab = _arrays(a, b)
    if ab is None:
        return _NOPE
    a, b = ab
    an = a.ndim
    if axes is not None:
        allaxes = list(range(an))
        for k in axes:
            allaxes.remove(k)
            allaxes.insert(an, k)
        a = a.transpose(allaxes)
    oldshape = a.shape[-(an - b.ndim):]
    prod = 1
    for k in oldshape:
        prod *= k
    if a.size != prod ** 2:
        raise LinAlgError(
            "Input arrays must satisfy the requirement "
            "            prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])"
        )
    a = a.reshape(prod, prod)
    b = b.ravel()
    res = np.linalg.solve(a, b)
    return res.reshape(oldshape)


def _tensorinv(a, ind=2):
    a = _one(a)
    if a is None:
        return _NOPE
    oldshape = a.shape
    prod = 1
    if ind > 0:
        invshape = oldshape[ind:] + oldshape[:ind]
        for k in oldshape[ind:]:
            prod *= k
    else:
        raise ValueError("Invalid ind argument.")
    a = a.reshape(prod, -1)
    ia = np.linalg.inv(a)
    return ia.reshape(*invshape)


def _matrix_norm(x, /, *, keepdims=False, ord="fro"):
    x = _one(x)
    if x is None:
        return _NOPE
    return np.linalg.norm(x, axis=(-2, -1), keepdims=keepdims, ord=ord)


def _small_ready(func, handler):
    """Small ready inputs (at most sync_below elements together): nothing to
    overlap, NumPy's own function runs right away on read-only exports (its
    results are fresh memory, adopted); else the handler's task. Not with a
    viproc array beyond the first two arguments (out=)."""
    def h(*args, **kwargs):
        arrays = [x for x in args[:2] if type(x) is ndarray]
        if (arrays and sum(x.size for x in arrays) <= _nd._small_ops[0]
                and not any(type(x) is ndarray for x in args[2:])
                and not any(type(x) is ndarray for x in kwargs.values())
                and all(_viproc.ready(x) for x in arrays)):
            r = func(*[np.asarray(x) if type(x) is ndarray else x for x in args], **kwargs)
            if isinstance(r, tuple):
                return type(r)(*[_nd.fresh(x) for x in r])
            return _nd.fresh(r)
        return handler(*args, **kwargs)

    return h


def _svd_entry(svd_small):
    """svd: hermitian=True always as NumPy's composition (its results are
    views, a reversed one too, which NumPy's result adopted would lose)."""
    def h(a, full_matrices=True, compute_uv=True, hermitian=False):
        if hermitian:
            return _svd(a, full_matrices, compute_uv, True)
        return svd_small(a, full_matrices, compute_uv, hermitian)

    return h


LINALG = {f: _small_ready(f, h) for f, h in {
    np.linalg.det: _det, np.linalg.slogdet: _slogdet, np.linalg.inv: _inv,
    np.linalg.solve: _solve, np.linalg.cholesky: _cholesky, np.linalg.eigvals: _eigvals,
    np.linalg.eig: _eig, np.linalg.eigvalsh: _eigvalsh, np.linalg.eigh: _eigh,
    np.linalg.svd: _svd, np.linalg.svdvals: _svdvals,
}.items()}
LINALG[np.linalg.svd] = _svd_entry(LINALG[np.linalg.svd])
LINALG.update({
    np.linalg.matmul: _linalg_matmul, np.linalg.vecdot: _linalg_vecdot,
    np.linalg.outer: _linalg_outer, np.linalg.tensordot: _linalg_tensordot,
    np.linalg.diagonal: _linalg_diagonal, np.linalg.trace: _linalg_trace,
    np.linalg.matrix_transpose: _linalg_matrix_transpose, np.tensordot: _tensordot,
    np.linalg.matrix_power: _matrix_power, np.linalg.multi_dot: _multi_dot,
    np.linalg.tensorsolve: _tensorsolve, np.linalg.tensorinv: _tensorinv,
    np.linalg.matrix_norm: _matrix_norm,
})
