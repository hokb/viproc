"""NumPy functions composed of viproc's asynchronous ops (hokb/viproc#25,
step 2).

Each handler issues the ufunc calls, reductions, assignments and views that
NumPy's own implementation makes, in the same order and with the same dtypes
and layouts, so results are NumPy's bit for bit. The code follows NumPy's
(numpy/_core/_methods.py, fromnumeric.py, numeric.py, calculation.c,
lib/_function_base_impl.py, _nanfunctions_impl.py, _twodim_base_impl.py,
linalg/_linalg.py; BSD-3-Clause) without the asanyarray() calls, which would
sync a viproc argument. A handler returns _NOPE for arguments it does not
cover: NumPy then runs the call in the eager fallback, with its own errors.

Where NumPy decides something from the data (a warning for an all-NaN slice,
allclose's bool), the handler syncs on that small result only, after the
work was issued as tasks.
"""

import builtins
import operator
import warnings

import numpy as np
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple

from . import _ndarray as _nd
from . import _viproc

_NV = np._NoValue
_NOPE = _nd._NOPE
ndarray = _nd.ndarray


def _vp(x):
    return type(x) is ndarray


def _arr(x):
    """asanyarray() that keeps viproc arrays as they are (no sync)."""
    return x if isinstance(x, ndarray) else np.asanyarray(x)


def _layout_flags(a):
    """NumPy's flags for a's layout, from a probe over one element (no sync)."""
    return np.lib.stride_tricks.as_strided(np.empty(1, a.dtype), a.shape, a.strides).flags


def _is_fortran(a):  # PyArray_ISFORTRAN
    f = _layout_flags(a)
    return f.f_contiguous and not f.c_contiguous


# --- element-wise ------------------------------------------------------------------


def _clip(a, a_min=_NV, a_max=_NV, out=None, *, min=_NV, max=_NV, **kwargs):
    # fromnumeric.clip, then _methods._clip
    if not _vp(a) or kwargs:
        return _NOPE
    if a_min is _NV and a_max is _NV:
        a_min = None if min is _NV else min
        a_max = None if max is _NV else max
    elif a_min is _NV or a_max is _NV or min is not _NV or max is not _NV:
        return _NOPE  # NumPy's errors
    if a.dtype.kind in "iu":
        if type(a_min) is int and a_min <= np.iinfo(a.dtype).min:
            a_min = None
        if type(a_max) is int and a_max >= np.iinfo(a.dtype).max:
            a_max = None
    out_kw = {} if out is None else {"out": out}
    if a_min is None and a_max is None:
        return np.positive(a, **out_kw)
    if a_min is None:
        return np.minimum(a, a_max, **out_kw)
    if a_max is None:
        return np.maximum(a, a_min, **out_kw)
    return np._core.umath.clip(a, a_min, a_max, **out_kw)


def _power_of_ten(n):  # calculation.c
    if n < 9:
        return (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)[n]
    r = 1e9
    for _ in range(n - 9):
        r *= 10.0
    return r


def _round(a, decimals=0, out=None):
    # PyArray_Round (calculation.c)
    if not _vp(a) or out is not None or a.dtype.kind == "b":
        return _NOPE
    decimals = operator.index(decimals)
    if a.dtype.kind == "c":
        arr = _copy(a, "K")
        arr.real[...] = _round(a.real, decimals)
        arr.imag[...] = _round(a.imag, decimals)
        return arr
    integer = a.dtype.kind in "iu"
    if decimals >= 0:
        if integer:
            return _copy(a, "K")
        if decimals == 0:
            return np.rint(a)
        op1, op2 = np.multiply, np.true_divide
    else:
        op1, op2 = np.true_divide, np.multiply
        decimals = 2**31 - 1 if decimals == -(2**31) else -decimals
    f = _power_of_ten(decimals)
    fortran = _is_fortran(a)
    dtype = np.dtype(np.float64) if integer else a.dtype
    flags = _layout_flags(a)
    if flags.c_contiguous or flags.f_contiguous:
        ret = op1(a, f)  # order K: the layout of NumPy's C or F result
    else:
        ret = _nd._adopt(np.empty(a.shape, dtype, order="F" if fortran else "C"))
        op1(a, f, out=ret)
    np.rint(ret, out=ret)
    op2(ret, f, out=ret)
    if integer:
        return ret.astype(a.dtype, order="F" if fortran else "C")
    return ret


def _angle(z, deg=False):
    if not _vp(z):
        return _NOPE
    if z.dtype.kind == "c":
        zimag, zreal = z.imag, z.real
    else:
        zimag, zreal = 0, z
    a = np.arctan2(zimag, zreal)
    if deg:
        a *= 180 / np.pi
    return a


def _sinc(x):
    if not _vp(x):
        return _NOPE
    x = np.pi * x
    eps = np.finfo(x.dtype).eps if x.dtype.kind == "f" else 1e-20
    y = np.where(x, x, eps)
    return np.sin(y) / y


def _isposinf(x):  # lib/_ufunclike_impl.py, without out=
    return np.logical_and(np.isinf(x), ~np.signbit(x))


def _isneginf(x):
    return np.logical_and(np.isinf(x), np.signbit(x))


def _nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    if not _vp(x):
        return _NOPE
    if copy:
        x = _copy(x, "K")
    if x.dtype.kind not in "fc":
        return x[()] if x.ndim == 0 else x
    dest = (x.real, x.imag) if x.dtype.kind == "c" else (x,)
    info = np.finfo(x.real.dtype)
    maxf = info.max if posinf is None else posinf
    minf = info.min if neginf is None else neginf
    for d in dest:
        idx_nan = np.isnan(d)
        idx_posinf = _isposinf(d)
        idx_neginf = _isneginf(d)
        d[idx_nan] = nan  # np.copyto(d, nan, where=idx_nan)
        d[idx_posinf] = maxf
        d[idx_neginf] = minf
    return x[()] if x.ndim == 0 else x


def _isclose_impl(a, b, rtol, atol, equal_nan):
    x, y = (v if isinstance(v, (int, float, complex)) else _arr(v) for v in (a, b))
    if (dtype := getattr(y, "dtype", None)) is not None and dtype.kind != "m":
        dt = np.result_type(y.dtype if _vp(y) else y, 1.0)
        if y.dtype != dt:
            y = y.astype(dt) if _vp(y) else np.asanyarray(y, dtype=dt)
    elif isinstance(y, int):
        y = float(y)
    result = np.less_equal(abs(x - y), atol + rtol * abs(y)) & np.isfinite(y) | (x == y)
    if equal_nan:
        result |= np.isnan(x) & np.isnan(y)
    return result[()]


def _scalar_tolerances(rtol, atol):
    ok = all(isinstance(t, (int, float, np.integer, np.floating)) for t in (rtol, atol))
    return ok and np.isfinite(rtol) and np.isfinite(atol)


def _isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    if not (_vp(a) or _vp(b)) or not _scalar_tolerances(rtol, atol):
        return _NOPE  # array tolerances, NumPy's warning for invalid ones
    return _isclose_impl(a, b, rtol, atol, equal_nan)


def _allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    if not (_vp(a) or _vp(b)) or not _scalar_tolerances(rtol, atol):
        return _NOPE
    res = np.all(_isclose_impl(a, b, rtol, atol, equal_nan))
    return builtins.bool(res)  # a sync point, as in NumPy


def _array_equal(a1, a2, equal_nan=False):
    a1, a2 = (x if isinstance(x, ndarray) else np.asarray(x) for x in (a1, a2))
    if a1.shape != a2.shape:
        return False
    if not equal_nan:
        return builtins.bool((a1 == a2).all())
    if a1 is a2:
        return True
    if a1.dtype.kind in "biu" and a2.dtype.kind in "biu":
        return builtins.bool((a1 == a2).all())
    a1nan = np.isnan(a1)
    if a1nan.all():
        return builtins.bool(np.isnan(a2).all())
    equal_or_both_nan = (a1 == a2) | (a1nan & np.isnan(a2))
    return builtins.bool(equal_or_both_nan.all())


def _polyval(p, x):
    if _vp(p) or not _vp(x):
        return _NOPE
    p = np.asarray(p)
    y = _zeros_like(x)
    for pv in p:
        y = y * x + pv
    return y


# --- reductions -------------------------------------------------------------------


def _ptp(a, axis=None, out=None, keepdims=_NV):
    if not _vp(a) or out is not None:
        return _NOPE
    return np.ptp._implementation(a, axis, None, keepdims)


def _trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    # PyArray_Trace: add.reduce over the last axis of the diagonal
    if not _vp(a) or out is not None:
        return _NOPE
    return np.add.reduce(_nd._diagonal(a, offset, axis1, axis2), axis=-1, dtype=dtype)


def _average(a, axis=None, weights=None, returned=False, *, keepdims=_NV):
    if not _vp(a):
        return _NOPE
    if axis is not None:
        axis = normalize_axis_tuple(axis, a.ndim, argname="axis")
    keepdims_kw = {} if keepdims is _NV else {"keepdims": keepdims}
    if weights is None:
        avg = a.mean(axis, **keepdims_kw)
        size = getattr(avg, "size", 1)
        scl = avg.dtype.type(a.size / size)
    else:
        wgt = _arr(weights)
        if wgt.shape != a.shape:
            return _NOPE  # NumPy's broadcasting along axis, and its errors
        if issubclass(a.dtype.type, (np.integer, np.bool_)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, "f8")
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)
        if result_dtype not in _nd._CODES:
            return _NOPE
        scl = wgt.sum(axis=axis, dtype=result_dtype, **keepdims_kw)
        if np.any(scl == 0.0):  # a sync point, as in NumPy
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")
        # np.multiply(a, wgt, dtype=result_dtype): both cast first, as NumPy does
        ac = a if a.dtype == result_dtype else a.astype(result_dtype)
        wc = wgt if wgt.dtype == result_dtype else wgt.astype(result_dtype)
        avg = np.multiply(ac, wc).sum(axis, **keepdims_kw) / scl
    if returned:
        if np.shape(scl) != np.shape(avg):
            scl = np.broadcast_to(scl, np.shape(avg)).copy()
        return avg, scl
    return avg


def _diff(a, n=1, axis=-1, prepend=_NV, append=_NV):
    if not _vp(a):
        return _NOPE
    if n == 0:
        return a
    if n < 0 or a.ndim == 0:
        return _NOPE
    axis = normalize_axis_index(axis, a.ndim)
    def side(x):  # a 0-d prepend/append is broadcast to one slice along axis
        x = _arr(x)
        if x.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            x = np.broadcast_to(x, tuple(shape))
        return x

    combined = [side(prepend)] if prepend is not _NV else []
    combined.append(a)
    if append is not _NV:
        combined.append(side(append))
    if len(combined) > 1:
        a = np.concatenate(combined, axis)
    slice1 = [slice(None)] * a.ndim
    slice2 = [slice(None)] * a.ndim
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1, slice2 = tuple(slice1), tuple(slice2)
    op = np.not_equal if a.dtype == np.bool_ else np.subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])
    return a


# NaN functions (lib/_nanfunctions_impl.py)


def _replace_nan(a, val):
    if a.dtype.kind not in "fc":
        return a, None
    mask = np.isnan(a)
    a = _copy(a, "K")
    a[mask] = val  # np.copyto(a, val, where=mask)
    return a, mask


def _copyto(a, val, mask):
    if isinstance(a, ndarray):
        a[mask] = val
    else:
        a = a.dtype.type(val)
    return a


def _divide_by_count(a, b):
    r = np.divide(a, b)  # NumPy: out=a, casting='unsafe'
    return r if r.dtype == a.dtype else r.astype(a.dtype)


def _nansum_like(func):
    def handler(a, axis=None, dtype=None, out=None, keepdims=_NV, initial=_NV, where=_NV):
        if not _vp(a) or out is not None or where is not _NV:
            return _NOPE
        a, _ = _replace_nan(a, 0 if func is np.sum else 1)
        return func(a, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial)

    return handler


def _nanminmax(reduce):
    def handler(a, axis=None, out=None, keepdims=_NV, initial=_NV, where=_NV):
        if not _vp(a) or out is not None or where is not _NV:
            return _NOPE
        kwargs = {}
        if keepdims is not _NV:
            kwargs["keepdims"] = keepdims
        if initial is not _NV:
            kwargs["initial"] = initial
        # NumPy's branch for ndarrays: fmax/fmin.reduce
        res = reduce(a, axis=axis, **kwargs)
        if a.dtype.kind in "fc" and np.isnan(res).any():  # a sync point, as in NumPy
            warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
        return res

    return handler


def _nanmean(a, axis=None, dtype=None, out=None, keepdims=_NV, *, where=_NV):
    if not _vp(a) or out is not None or where is not _NV:
        return _NOPE
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return np.mean(arr, axis=axis, dtype=dtype, keepdims=keepdims)
    if dtype is not None:
        dtype = np.dtype(dtype)
        if not issubclass(dtype.type, np.inexact):
            return _NOPE  # NumPy's TypeError
    cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=keepdims)
    tot = np.sum(arr, axis=axis, dtype=dtype, keepdims=keepdims)
    avg = _divide_by_count(tot, cnt)
    if (cnt == 0).any():  # a sync point, as in NumPy
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
    return avg


def _nanvar_impl(a, axis, dtype, ddof, keepdims, sqrt):
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        f = np.std if sqrt else np.var
        return f(arr, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    if dtype is not None:
        dtype = np.dtype(dtype)
        if not issubclass(dtype.type, np.inexact):
            return _NOPE
    cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=True)
    avg = np.sum(arr, axis=axis, dtype=dtype, keepdims=True)
    avg = _divide_by_count(avg, cnt)
    if avg.dtype != arr.dtype:
        return _NOPE  # NumPy subtracts with casting='unsafe'
    np.subtract(arr, avg, out=arr)
    arr = _copyto(arr, 0, mask)
    if arr.dtype.kind == "c":
        sqr = np.multiply(arr, arr.conj(), out=arr).real
    else:
        sqr = np.multiply(arr, arr, out=arr)
    var = np.sum(sqr, axis=axis, dtype=dtype, keepdims=keepdims)
    var_ndim = np.ndim(var) if not isinstance(var, ndarray) else var.ndim
    if var_ndim < cnt.ndim:
        cnt = cnt.squeeze(axis)
    dof = cnt - ddof
    var = _divide_by_count(var, dof)
    isbad = dof <= 0
    if np.any(isbad):  # a sync point, as in NumPy
        warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning, stacklevel=3)
        var = _copyto(var, np.nan, isbad)
    if sqrt:
        if isinstance(var, ndarray):
            return np.sqrt(var, out=var)
        return var.dtype.type(np.sqrt(var))
    return var


def _nanvar_like(sqrt):
    def handler(a, axis=None, dtype=None, out=None, ddof=0, keepdims=_NV, *, where=_NV,
                mean=_NV, correction=_NV):
        if (not _vp(a) or out is not None or where is not _NV or mean is not _NV
                or correction is not _NV):
            return _NOPE
        return _nanvar_impl(a, axis, dtype, ddof, keepdims, sqrt)

    return handler


def _nanarg(argfunc):
    def handler(a, axis=None, out=None, *, keepdims=_NV):
        if not _vp(a) or out is not None:
            return _NOPE
        a, mask = _replace_nan(a, -np.inf if argfunc is np.argmax else np.inf)
        if mask is not None and mask.size:
            if np.any(np.all(mask, axis=axis)):  # a sync point, as in NumPy
                raise ValueError("All-NaN slice encountered")
        return argfunc(a, axis=axis, keepdims=keepdims)

    return handler


def _norm(x, ord=None, axis=None, keepdims=False):
    # linalg/_linalg.py: the vector norms and the Frobenius norm
    if not _vp(x):
        return _NOPE
    if x.dtype.kind not in "fc":
        x = x.astype(float)
    if axis is None:
        ndim = x.ndim
        if ord is None or (ord in ("f", "fro") and ndim == 2) or (ord == 2 and ndim == 1):
            x = x.ravel(order="K")
            if x.dtype.kind == "c":
                x_real, x_imag = x.real, x.imag
                sqnorm = x_real.dot(x_real) + x_imag.dot(x_imag)
            else:
                sqnorm = x.dot(x)
            ret = np.sqrt(sqnorm)
            if keepdims:
                ret = ret.reshape(ndim * [1])
            return ret
        return _NOPE  # matrix norms
    if not isinstance(axis, tuple):
        try:
            axis = (int(axis),)
        except Exception:
            return _NOPE
    if len(axis) != 1:
        return _NOPE
    if ord == np.inf:
        return abs(x).max(axis=axis, keepdims=keepdims, initial=0)
    if ord == -np.inf:
        return abs(x).min(axis=axis, keepdims=keepdims)
    if ord == 0:
        return (x != 0).astype(x.real.dtype).sum(axis=axis, keepdims=keepdims)
    if ord == 1:
        return np.add.reduce(abs(x), axis=axis, keepdims=keepdims)
    if ord is None or ord == 2:
        s = (x.conj() * x).real
        return np.sqrt(np.add.reduce(s, axis=axis, keepdims=keepdims))
    if isinstance(ord, str):
        return _NOPE
    absx = abs(x)
    absx **= ord
    ret = np.add.reduce(absx, axis=axis, keepdims=keepdims)
    ret **= np.reciprocal(ord, dtype=ret.dtype)
    return ret


# --- products ---------------------------------------------------------------------


def _outer(a, b, out=None):
    if not (_vp(a) or _vp(b)):
        return _NOPE
    a, b = (x if isinstance(x, ndarray) else np.asarray(x) for x in (a, b))
    out_kw = {} if out is None else {"out": out}
    return np.multiply(a.ravel()[:, np.newaxis], b.ravel()[np.newaxis, :], **out_kw)


def _inner(a, b):
    # inner of two vectors makes dot's BLAS call (checked bit for bit)
    if not (_vp(a) and _vp(b)) or a.ndim != 1 or b.ndim != 1:
        return _NOPE
    return np.dot(a, b)


# --- copies, views, new arrays -------------------------------------------------------


def _copy(a, order="K", subok=False):
    if not _vp(a) or order not in ("K", "A", "C", "F"):
        return _NOPE
    if order == "A":
        order = "F" if _is_fortran(a) else "C"
    return a.astype(a.dtype, order=order)


def _imag(val):
    if not _vp(val) or val.dtype.kind != "c":
        return _NOPE
    return val.imag


def _probe(a):
    return np.lib.stride_tricks.as_strided(np.empty(1, a.dtype), a.shape, a.strides)


def _like(create):
    """zeros_like & co.: NumPy's function on a probe with a's layout (no
    sync, no task: fresh memory, adopted, as viproc.zeros)."""
    def handler(prototype, *args, **kwargs):
        if not _vp(prototype) or any(isinstance(v, ndarray) for v in args + tuple(kwargs.values())):
            return _NOPE
        return _nd._adopt(create(_probe(prototype), *args, **kwargs))

    return handler


def _zeros_like(a):
    return _like(np.zeros_like)(a)


def _triu_tril(upper):
    def handler(m, k=0):
        if not _vp(m) or m.ndim < 2:
            return _NOPE  # NumPy's code for 1-d input broadcasts the mask
        # fresh NumPy memory, adopted without a copy (asarray would copy)
        mask = _nd._adopt(np.tri(*m.shape[-2:], k=k - 1 if upper else k, dtype=bool))
        zero = _nd._adopt(np.zeros(1, m.dtype))
        return np.where(mask, zero, m) if upper else np.where(mask, m, zero)

    return handler


def _diag(v, k=0):
    if not _vp(v):
        return _NOPE
    if v.ndim == 2:
        return _nd._diagonal(v, k)
    if v.ndim != 1:
        return _NOPE
    n = v.shape[0] + abs(k)
    res = _nd._adopt(np.zeros((n, n), v.dtype))
    if v.shape[0]:
        i = k if k >= 0 else (-k) * n
        item = v.dtype.itemsize
        # res[:n - k].flat[i::n + 1] = v
        _viproc.view(res, (v.shape[0],), ((n + 1) * item,), i * item)[...] = v
    return res


def _broadcast_arrays(*args, subok=False):
    if not any(_vp(a) for a in args):
        return _NOPE
    args = [a if isinstance(a, ndarray) else np.array(a, copy=None, subok=subok) for a in args]
    shape = np.broadcast_shapes(*(a.shape for a in args))
    return tuple(a if a.shape == shape else np.broadcast_to(a, shape) for a in args)


def _meshgrid(*xi, copy=True, sparse=False, indexing="xy"):
    if not any(_vp(x) for x in xi) or indexing not in ("xy", "ij"):
        return _NOPE
    ndim = len(xi)
    s0 = (1,) * ndim
    output = [_arr(x).reshape(s0[:i] + (-1,) + s0[i + 1:]) for i, x in enumerate(xi)]
    if indexing == "xy" and ndim > 1:
        output[0] = output[0].reshape((1, -1) + s0[2:])
        output[1] = output[1].reshape((-1, 1) + s0[2:])
    if not sparse:
        output = _broadcast_arrays(*output)
    if copy:
        output = tuple(x.copy() for x in output)
    if sparse and not copy:
        return tuple(output)
    return output


def _numpy_code(func):
    """NumPy's own Python implementation: it only makes views (no asanyarray)."""
    def handler(ary, *args, **kwargs):
        return func._implementation(ary, *args, **kwargs) if _vp(ary) else _NOPE

    return handler


COMPOSITIONS = {
    np.clip: _clip, np.round: _round, np.around: _round, np.angle: _angle, np.sinc: _sinc,
    np.nan_to_num: _nan_to_num, np.isclose: _isclose, np.allclose: _allclose,
    np.array_equal: _array_equal, np.polyval: _polyval,
    np.ptp: _ptp, np.trace: _trace, np.average: _average, np.diff: _diff,
    np.nansum: _nansum_like(np.sum), np.nanprod: _nansum_like(np.prod),
    np.nanmax: _nanminmax(np.fmax.reduce), np.nanmin: _nanminmax(np.fmin.reduce),
    np.nanmean: _nanmean, np.nanvar: _nanvar_like(False), np.nanstd: _nanvar_like(True),
    np.nanargmax: _nanarg(np.argmax), np.nanargmin: _nanarg(np.argmin),
    np.linalg.norm: _norm, np.outer: _outer, np.inner: _inner,
    np.copy: _copy, np.imag: _imag,
    np.zeros_like: _like(np.zeros_like), np.ones_like: _like(np.ones_like),
    np.full_like: _like(np.full_like), np.empty_like: _like(np.empty_like),
    np.triu: _triu_tril(True), np.tril: _triu_tril(False), np.diag: _diag,
    np.broadcast_arrays: _broadcast_arrays, np.meshgrid: _meshgrid,
    np.array_split: _numpy_code(np.array_split), np.split: _numpy_code(np.split),
}
