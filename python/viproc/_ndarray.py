"""The viproc array type and its NumPy integration.

viproc arrays intercept NumPy through the array protocols (NEP 13
``__array_ufunc__``, NEP 18 ``__array_function__``). Supported ops are issued
asynchronously to the runtime; everything else syncs, runs eagerly in NumPy and
wraps the result again (fallback).

The per-op hot path (``__array_ufunc__`` for element-wise ufuncs and all
operators) is implemented in C by the base class ``_viproc._Array``; this
module adds everything else and the fallbacks the C code calls.
"""

import builtins
import operator
import os
import sys
import warnings
import weakref

import numpy as np
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple

from . import _viproc

AsyncError = _viproc.AsyncError
AsyncIndexError = _viproc.AsyncIndexError
AsyncValueError = _viproc.AsyncValueError

# dtype <-> runtime dtype code (vp_dtype in viproc.h)
_CODES = {
    np.dtype(np.bool_): 0,
    np.dtype(np.int32): 1,
    np.dtype(np.int64): 2,
    np.dtype(np.float32): 3,
    np.dtype(np.float64): 4,
    np.dtype(np.complex64): 5,
    np.dtype(np.complex128): 6,
}
_DTYPES = {code: dt for dt, code in _CODES.items()}

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_NUMPY_DIR = os.path.dirname(os.path.abspath(np.__file__))


def _env_policy():
    sync_below = int(os.environ.get("VIPROC_SYNC_BELOW", "-1"))
    adaptive = os.environ.get("VIPROC_DETERMINISTIC", "0") in ("", "0")
    return sync_below, adaptive


def _ensure_runtime():
    if not _viproc.initialized():
        workers = int(os.environ.get("VIPROC_WORKERS", "0"))
        sync_below, adaptive = _env_policy()
        _viproc.init(workers, 0, sync_below, adaptive)
        _small_ops[0] = 2048 if sync_below < 0 else sync_below


def _wrap_numpy(arr):
    """viproc array over a NumPy array's memory. The caller gives up `arr`."""
    _ensure_runtime()
    if arr.dtype not in _CODES or not arr.flags.aligned or not arr.dtype.isnative:
        return None
    return _viproc.from_ndarray(arr)


# viproc on/off for code that creates arrays (viproc.enable, viproc.disabled):
# off, the creation functions return NumPy's own arrays, so the same program
# runs as plain NumPy (benchmarking, debugging).
_enabled = [True]


def array(obj, dtype=None, *, copy=True, order="K", subok=False, ndmin=0, like=None):
    """Like numpy.array, returning a viproc array.

    NumPy data is always copied (the caller keeps its ndarray and may write
    it later; design doc D7), also with copy=False. Dtypes viproc does not
    support (object, str, float16, ...) give NumPy's array, as do all calls
    while viproc is disabled."""
    if not _enabled[0] or like is not None:
        return np.array(obj, dtype, copy=copy, order=order, subok=subok, ndmin=ndmin, like=like)
    if isinstance(obj, ndarray) and order in ("K", "A"):
        r = obj
        if dtype is not None and np.dtype(dtype) != obj.dtype:
            r = obj.astype(dtype)
        if copy and r is obj:
            r = obj.copy(order=order)  # NumPy: order K keeps the layout
        if ndmin > r.ndim:
            r = r.reshape((1,) * (ndmin - r.ndim) + r.shape)
        return r
    arr = np.array(obj, dtype, copy=True, order=order, ndmin=ndmin)
    return _adopt(arr)


def asarray(obj, dtype=None, order=None, *, copy=None, like=None, **kwargs):
    """Like numpy.asarray, returning a viproc array (viproc arrays of the
    requested dtype pass through; NumPy data is copied, see array())."""
    if not _enabled[0] or like is not None or kwargs:
        return np.asarray(obj, dtype, order, copy=copy, like=like, **kwargs)
    if isinstance(obj, ndarray) and (dtype is None or obj.dtype == np.dtype(dtype)) and not copy:
        return obj
    return array(obj, dtype, copy=True, order=order or "K")


def fresh(r, args=(), kwargs=None):
    """Result `r` of a NumPy function as viproc array: fresh memory is adopted
    without a copy, a view of other memory is copied first. Returned as is:
    anything but a plain ndarray, an argument of the call (out=), dtypes
    viproc does not support, and everything while viproc is disabled."""
    if not _enabled[0] or type(r) is not np.ndarray:
        return r
    if any(r is a for a in args) or (kwargs and any(r is v for v in kwargs.values())):
        return r
    if r.dtype not in _CODES:
        return r
    if not r.flags.owndata:
        r = r.copy(order="K")  # keeps NumPy's layout (a transposed result)
    return _adopt(r)


def _to_numpy(x):
    """NumPy input from a viproc array: a read-only view, no copy."""
    return np.asarray(x) if isinstance(x, ndarray) else x


def _rebuild(x, items):
    """Container of the same type as `x` (list, tuple, named tuple)."""
    if hasattr(x, "_fields"):  # named tuple
        return type(x)(*items)
    return type(x)(items)


def _tree(x, fn):
    if isinstance(x, (list, tuple)):
        return _rebuild(x, [_tree(i, fn) for i in x])
    if isinstance(x, dict):
        return {k: _tree(v, fn) for k, v in x.items()}
    return fn(x)


def _adopt(arr):
    """viproc array over fresh NumPy memory (no copy); unsupported: as is."""
    w = _wrap_numpy(arr)
    return w if w is not None else arr


class _Host:
    """Runs NumPy code eagerly on viproc arrays with NumPy's semantics
    (design doc D7, "Eager fallbacks").

    Each viproc argument becomes a *writable* NumPy view of its buffer, made
    exclusive first (export(a, True): waits for pending writers and readers,
    copies-on-write if read-only exports of it are alive). NumPy code then
    reads and writes viproc memory as it would an ndarray's: in-place
    functions, ``out=`` and overlap handling are NumPy's own. Results are
    mapped back: an argument returned as is -> that viproc array; a view of
    an argument -> a viproc view of the same storage (writable as NumPy made
    it); a view of a NumPy argument -> returned unchanged; fresh memory ->
    adopted; anything else -> copied.
    """

    def __init__(self):
        self.exports = {}  # id(export) -> (export, viproc array)
        self.plain = {}  # id(x) -> NumPy arguments

    def numpy(self, tree):
        return _tree(tree, self._convert)

    def _convert(self, x):
        if isinstance(x, ndarray):
            e = _viproc.export(x, True)  # read-only for read-only viproc arrays
            self.exports[id(e)] = (e, x)
            return e
        if isinstance(x, np.ndarray):
            self.plain[id(x)] = x
        return x

    def done(self, result):
        r = self._map(result)
        for e, _ in self.exports.values():
            e.flags.writeable = False  # in case NumPy code kept a reference
        return r

    def _map(self, r):
        if isinstance(r, (list, tuple)):
            return _rebuild(r, [self._map(i) for i in r])
        if not isinstance(r, np.ndarray) or isinstance(r, ndarray):
            return r
        for e, v in self.exports.values():
            if r is e:
                return v  # identity, as NumPy (np.atleast_1d(x) is x)
        # Views are recognized by memory, not by the base chain: NumPy code
        # may build them through other objects (as_strided, DummyArray).
        for e, v in self.exports.values():
            if np.may_share_memory(r, e):
                code = _CODES.get(r.dtype)
                if code is None or not r.flags.aligned or not r.dtype.isnative:
                    return _adopt(r.copy())  # a view viproc cannot express
                offset = r.__array_interface__["data"][0] - e.__array_interface__["data"][0]
                w = _viproc.view(v, r.shape, r.strides, offset, code)
                with warnings.catch_warnings():  # broadcast_arrays warns on reading it
                    warnings.simplefilter("ignore", FutureWarning)
                    w._readonly = not r.flags.writeable
                return w
        for p in self.plain.values():
            if r is p or np.may_share_memory(r, p):
                return r  # aliases a NumPy argument: unchanged, as NumPy returns it
        if r.flags.owndata and r.base is None:
            return _adopt(r)  # fresh memory nobody else holds
        return _adopt(r.copy())


# Eager fallbacks run so far (each one is a sync point); read by benchmarks.
fallback_calls = 0


def _host_call(fn, args, kwargs):
    global fallback_calls
    fallback_calls += 1
    host = _Host()
    np_args = host.numpy(args)
    np_kwargs = host.numpy(kwargs)
    return host.done(fn(*np_args, **np_kwargs))


# --- reductions -----------------------------------------------------------------
# Asynchronous with NumPy's own reduction loops (vp_reduce): viproc calls the
# loop in the blocks NumPy's reduction would, so results are bit-identical.
# That calling pattern is the one of NumPy >= 2.3 (earlier versions split
# loops at the buffer size); older NumPy keeps reductions eager.
_REDUCE_ASYNC = np.lib.NumpyVersion(np.__version__) >= "2.3.0"

# Reorderable binary ufuncs; reduce() starts at ufunc.identity, or at the
# first element where it is None (minimum, maximum, fmin, fmax).
_REDUCE_UFUNCS = frozenset((
    np.add, np.multiply, np.minimum, np.maximum, np.fmin, np.fmax, np.logical_and,
    np.logical_or, np.logical_xor, np.bitwise_and, np.bitwise_or, np.bitwise_xor,
))
_MISS = object()
_NV = np._NoValue
# (ufunc, dtype, ndim, axis, keepdims, requested dtype) -> (axis flags,
# keepdims, identity), or None if viproc cannot run that reduction. Per-call
# work is then one lookup (normalizing axes and probing NumPy's result dtype
# cost more than the reduction of 1,000 elements).
_reduce_plans = {}


def _plan_reduce(ufunc, dtype, ndim, axis, keepdims, requested):
    try:  # NumPy's result dtype (int32 sums accumulate in int64); a viproc dtype?
        with warnings.catch_warnings():  # the probe only: the call itself warns
            warnings.simplefilter("ignore")
            if ufunc.reduce(np.zeros(1, dtype), dtype=requested).dtype not in _CODES:
                return None
    except (TypeError, ValueError):
        return None  # NumPy raises the error
    if axis is None:
        flags = (True,) * ndim
    else:
        try:
            axes = normalize_axis_tuple(axis, ndim)
        except (TypeError, ValueError, np.exceptions.AxisError):
            return None
        flags = tuple(i in axes for i in range(ndim))
    return flags, bool(keepdims), None if requested is None else np.dtype(requested)


def _reduce(ufunc, a, axis=None, keepdims=False, dtype=None, initial=_NV, out=None):
    """ufunc.reduce(a, axis, dtype=..., keepdims=..., initial=..., out=...)
    issued asynchronously, or None if viproc cannot run it (the caller falls
    back to NumPy; empty inputs too, vp_reduce refuses them). Casting
    reductions and strided ones NumPy buffers run on NumPy's own iterator
    (vp_reduce). out=: a viproc array with the dtype and layout of the
    result NumPy would allocate (vp_reduce_out), returned as NumPy returns
    it."""
    if not _REDUCE_ASYNC or type(a) is not ndarray or ufunc not in _REDUCE_UFUNCS:
        return None
    if out is not None and (type(out) is not ndarray or out._readonly):
        return None
    try:
        key = (ufunc, a.dtype, a.ndim, axis, keepdims, dtype)
        plan = _reduce_plans.get(key, _MISS)
    except TypeError:  # unhashable axis (a list): plan without the cache
        key, plan = None, _MISS
    if plan is _MISS:
        plan = _plan_reduce(ufunc, a.dtype, a.ndim, axis, keepdims, dtype)
        if key is not None:
            if len(_reduce_plans) > 4096:
                _reduce_plans.clear()
            _reduce_plans[key] = plan
    if plan is None:
        return None
    flags, keep, requested = plan
    start = ufunc.identity if initial is _NV else initial
    if out is not None:
        return _viproc.reduce(ufunc, a, flags, keep, requested, start, out)
    r = _viproc.reduce(ufunc, a, flags, keep, requested, start)
    return _scalar_result(r) if r is not None and r.ndim == 0 else r


def _scalar_result(r):
    """A 0-d result NumPy returns as a scalar: NumPy's scalar if the data is
    there already (it ran directly), else the pending 0-d array, marked so
    that in-place operators rebind it as they rebind NumPy's scalars."""
    s = _viproc.scalar(r)
    if s is not None:
        return s
    r._scalar = True
    return r


def _reduced_count(a, axis):
    """Elements per result element (NumPy's _count_reduce_items)."""
    if axis is None:
        return a.size
    n = 1
    for ax in normalize_axis_tuple(axis, a.ndim):
        n *= a.shape[ax]
    return n


_NOPE = object()  # handler result: not supported here, fall back


def _sum_or_numpy(x, axis, keepdims):
    r = _reduce(np.add, x, axis, keepdims)
    return r if r is not None else np.add.reduce(x, axis=axis, keepdims=keepdims)


def _r_sum(ufunc):
    def handler(a, axis=None, dtype=None, out=None, keepdims=_NV, initial=_NV, where=_NV):
        if where is not _NV:
            return _NOPE
        r = _reduce(ufunc, a, axis, keepdims is True, dtype, initial, out)
        return _NOPE if r is None else r

    return handler


def _r_minmax(ufunc):
    def handler(a, axis=None, out=None, keepdims=_NV, initial=_NV, where=_NV):
        if where is not _NV:
            return _NOPE
        r = _reduce(ufunc, a, axis, keepdims is True, None, initial, out)
        return _NOPE if r is None else r

    return handler


def _r_anyall(ufunc):
    # np.any / np.all reduce with dtype=bool: each element as truth value,
    # which is `a != 0` (NaN counts as true, complex if either part is set).
    def handler(a, axis=None, out=None, keepdims=_NV, *, where=_NV):
        if out is not None or where is not _NV or not isinstance(a, ndarray):
            return _NOPE
        b = a if a.dtype == np.bool_ else (a != 0)
        if not isinstance(b, ndarray):
            return _NOPE
        r = _reduce(ufunc, b, axis, keepdims is True)
        return _NOPE if r is None else r

    return handler


_INTP = np.intp


def _divide_by_count(r, n):
    """NumPy's true_divide(r, intp(n), out=r, casting='unsafe') (and, for
    0-d results, r.dtype.type(r / n)): divided in the promoted dtype of r
    and intp (float64, complex128), cast back to r's dtype."""
    q = np.true_divide(r, _INTP(n))
    return q if q.dtype == r.dtype else q.astype(r.dtype)


def _moment_dtype(a, dtype):
    if dtype is None and a.dtype.kind in "biu":
        return np.dtype(np.float64)  # NumPy: bool and integers in float64
    return dtype


def _r_mean(a, axis=None, dtype=None, out=None, keepdims=_NV, *, where=_NV):
    # NumPy's _methods._mean: sum (in float64 for bool and integers), then
    # divide by the count.
    if out is not None or where is not _NV or type(a) is not ndarray or not a.size:
        return _NOPE
    s = _reduce(np.add, a, axis, keepdims is True, _moment_dtype(a, dtype))
    if s is None or s.dtype.kind not in "fcbiu":
        return _NOPE
    return _divide_by_count(s, _reduced_count(a, axis))


_COMPLEX_TO_FLOAT = {np.dtype(np.complex64): np.dtype(np.float32),
                     np.dtype(np.complex128): np.dtype(np.float64)}


def _r_var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=_NV, *, where=_NV,
           mean=_NV, correction=_NV, sqrt=False):
    # NumPy's _methods._var (and _std), step by step.
    if (out is not None or where is not _NV or mean is not _NV or correction is not _NV
            or type(a) is not ndarray or not a.size or a.dtype.kind not in "fcbiu"):
        return _NOPE
    count = _reduced_count(a, axis)
    if count - ddof <= 0:
        return _NOPE  # NumPy warns and returns inf/nan
    dtype = _moment_dtype(a, dtype)
    m = _reduce(np.add, a, axis, True, dtype)
    if m is None:
        return _NOPE
    m = _divide_by_count(m, count)
    x = np.subtract(a, m)
    if a.dtype.kind in "fiu":
        x = np.square(x, out=x)
    elif x.dtype in _COMPLEX_TO_FLOAT:
        xv = x.view((_COMPLEX_TO_FLOAT[x.dtype], (2,)))
        np.square(xv, out=xv)
        x = np.add(xv[..., 0], xv[..., 1], out=x.real)
    else:
        x = np.multiply(x, np.conjugate(x), out=x).real
    v = _reduce(np.add, x, axis, keepdims is True, dtype)
    if v is None:
        v = np.add.reduce(x, axis=axis, dtype=dtype, keepdims=keepdims is True)
    v = _divide_by_count(v, max(count - ddof, 0))
    if not sqrt:
        return v
    return np.sqrt(v, out=v) if isinstance(v, ndarray) else np.sqrt(v)


def _r_arg(argmax):
    # PyArray_ArgMax / ArgMin: the axis moved last (axis=None: raveled), a
    # C-contiguous copy if needed, NumPy's argmax function once per row.
    def handler(a, axis=None, out=None, *, keepdims=_NV):
        if out is not None or type(a) is not ndarray or not a.size:
            return _NOPE
        keep = keepdims is True
        if axis is None:
            src = a if a.ndim == 1 else a.reshape(-1)
            shape = (1,) * a.ndim if keep else ()
        else:
            try:
                ax = normalize_axis_tuple(axis, a.ndim)
            except (TypeError, ValueError, np.exceptions.AxisError):
                return _NOPE  # NumPy raises
            if len(ax) != 1 or not isinstance(axis, (int, np.integer)):
                return _NOPE
            ax = ax[0]
            src = np.moveaxis(a, ax, -1)
            shape = src.shape[:-1]
            if keep:
                shape = shape[:ax] + (1,) + shape[ax:]
        if not src._c_contiguous:
            src = src.astype(src.dtype, order="C")
        r = _viproc.arg_reduce(argmax, src, shape)
        if r is None:
            return _NOPE
        return _scalar_result(r) if r.ndim == 0 else r

    return handler


def _r_std(*args, **kwargs):
    return _r_var(*args, sqrt=True, **kwargs)


_REDUCTIONS = {
    np.sum: _r_sum(np.add),
    np.prod: _r_sum(np.multiply),
    np.max: _r_minmax(np.maximum),
    np.min: _r_minmax(np.minimum),
    np.amax: _r_minmax(np.maximum),
    np.amin: _r_minmax(np.minimum),
    np.any: _r_anyall(np.logical_or),
    np.all: _r_anyall(np.logical_and),
    np.argmax: _r_arg(True),
    np.argmin: _r_arg(False),
    np.mean: _r_mean,
    np.var: _r_var,
    np.std: _r_std,
}


# --- views (metadata only, no sync) --------------------------------------------
# Views are computed here and in C (_viproc.basic_view, transpose, reshape,
# broadcast_to) with NumPy's own rules (get_view_from_index, PyArray_Transpose,
# PyArray_Newshape / _attempt_nocopy_reshape, PyArray_Diagonal, ...); no
# ndarray is created for them. Invalid arguments: NumPy raises its own error
# on a zero-stride shape probe (np.broadcast_to of one element: no element
# memory), so the messages stay NumPy's.


def _numpy_error(fn, a, *args, **kwargs):
    """Raise the error NumPy raises for fn(a, ...) (a's shape and dtype
    matter only); _NOPE if NumPy does not raise (the caller falls back)."""
    probe = np.broadcast_to(np.empty((), a.dtype), a.shape)
    fn(probe, *args, **kwargs)
    return _NOPE


def _transpose(a, axes=None):
    if type(a) is not ndarray:
        return _NOPE
    r = _viproc.transpose(a, axes)
    return _numpy_error(np.transpose, a, axes) if r is None else r


def _swapaxes(a, axis1, axis2):
    if type(a) is not ndarray:
        return _NOPE
    i = normalize_axis_index(axis1, a.ndim, "axis1")
    j = normalize_axis_index(axis2, a.ndim, "axis2")
    perm = list(range(a.ndim))
    perm[i], perm[j] = perm[j], perm[i]
    return _viproc.transpose(a, perm)


def _moveaxis(a, source, destination):
    # numpy/_core/numeric.py: moveaxis
    if type(a) is not ndarray:
        return _NOPE
    source = normalize_axis_tuple(source, a.ndim, "source")
    destination = normalize_axis_tuple(destination, a.ndim, "destination")
    if len(source) != len(destination):
        raise ValueError("`source` and `destination` arguments must have "
                         "the same number of elements")
    order = [n for n in range(a.ndim) if n not in source]
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)
    return _viproc.transpose(a, order)


def _matrix_transpose(x):
    if type(x) is not ndarray:
        return _NOPE
    if x.ndim < 2:
        return _numpy_error(np.matrix_transpose, x)
    perm = list(range(x.ndim))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return _viproc.transpose(x, perm)


def _basic(a, key):
    """a[key] for a basic key as a view (also for an all-int key: 0-d)."""
    r = _viproc.basic_view(a, key)
    if r is None:
        probe = np.broadcast_to(np.empty((), a.dtype), a.shape)
        probe[key]  # NumPy's IndexError
        raise IndexError(f"invalid index {key!r}")
    return r


def _squeeze(a, axis=None):
    if type(a) is not ndarray:
        return _NOPE
    if axis is None:
        axes = [i for i, n in enumerate(a.shape) if n == 1]
    else:
        axes = normalize_axis_tuple(axis, a.ndim)
        if any(a.shape[i] != 1 for i in axes):
            raise ValueError("cannot select an axis to squeeze out which has size not "
                             "equal to one")
    if not axes:
        return _viproc.basic_view(a, ())
    return _viproc.basic_view(a, tuple(0 if i in axes else slice(None)
                                       for i in range(a.ndim)))


def _expand_dims(a, axis):
    # numpy/lib/_shape_base_impl.py: expand_dims, a.reshape(shape)
    if type(a) is not ndarray:
        return _NOPE
    if type(axis) not in (tuple, list):
        axis = (axis,)
    out_ndim = len(axis) + a.ndim
    axis = normalize_axis_tuple(axis, out_ndim)
    it = iter(a.shape)
    shape = [1 if ax in axis else next(it) for ax in range(out_ndim)]
    return _reshape(a, shape)


def _flip(m, axis=None):
    if type(m) is not ndarray:
        return _NOPE
    if axis is None:
        key = (slice(None, None, -1),) * m.ndim
    else:
        axes = normalize_axis_tuple(axis, m.ndim)
        key = tuple(slice(None, None, -1) if i in axes else slice(None)
                    for i in range(m.ndim))
    return _viproc.basic_view(m, key)


def _fliplr(m):
    if type(m) is not ndarray:
        return _NOPE
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return _viproc.basic_view(m, (slice(None), slice(None, None, -1)))


def _flipud(m):
    if type(m) is not ndarray:
        return _NOPE
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    return _viproc.basic_view(m, (slice(None, None, -1),))


def _rot90(m, k=1, axes=(0, 1)):
    # numpy/lib/_function_base_impl.py: rot90
    if type(m) is not ndarray:
        return _NOPE
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")
    if axes[0] == axes[1] or abs(axes[0] - axes[1]) == m.ndim:
        raise ValueError("Axes must be different.")
    if (axes[0] >= m.ndim or axes[0] < -m.ndim or axes[1] >= m.ndim or axes[1] < -m.ndim):
        raise ValueError(f"Axes={axes} out of range for array of ndim={m.ndim}.")
    k %= 4
    if k == 0:
        return _viproc.basic_view(m, ())
    if k == 2:
        return _flip(_flip(m, axes[0]), axes[1])
    axes_list = list(range(m.ndim))
    (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]], axes_list[axes[0]])
    if k == 1:
        return _viproc.transpose(_flip(m, axes[1]), axes_list)
    return _flip(_viproc.transpose(m, axes_list), axes[1])


def _atleast(n):
    # numpy/_core/shape_base.py: atleast_1d / 2d / 3d
    def one(a):
        if a.ndim >= n:
            return a
        if n == 1 or a.ndim == 0:
            return _reshape(a, (1,) * n)
        if n == 2:
            return _viproc.basic_view(a, (None, slice(None)))
        if a.ndim == 1:
            return _viproc.basic_view(a, (None, slice(None), None))
        return _viproc.basic_view(a, (slice(None), slice(None), None))

    def handler(*arys):
        if not arys or any(type(a) is not ndarray for a in arys):
            return _NOPE
        res = tuple(one(a) for a in arys)
        return res[0] if len(res) == 1 else res

    return handler


def _broadcast_to(array, shape, subok=False):
    if type(array) is not ndarray:
        return _NOPE
    shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
    r = _viproc.broadcast_to(array, shape)
    return _numpy_error(np.broadcast_to, array, shape) if r is None else r


def _diagonal(a, offset=0, axis1=0, axis2=1):
    # PyArray_Diagonal: a read-only view
    if type(a) is not ndarray:
        return _NOPE
    if a.ndim < 2:
        raise ValueError("diag requires an array of at least two dimensions")
    axis1 = normalize_axis_index(axis1, a.ndim, "axis1")
    axis2 = normalize_axis_index(axis2, a.ndim, "axis2")
    if axis1 == axis2:
        raise ValueError("axis1 and axis2 cannot be the same")
    shape, strides = a.shape, a.strides
    dim1, dim2 = shape[axis1], shape[axis2]
    offset = operator.index(offset)
    if offset >= 0:
        step, dim2 = strides[axis2], dim2 - offset
    else:
        offset, step, dim1 = -offset, strides[axis1], dim1 + offset
    size = min(dim1, dim2)
    start = 0
    if size < 0:
        size = 0
    else:
        start = offset * step
    keep = [i for i in range(a.ndim) if i not in (axis1, axis2)]
    v = _viproc.view(a, [shape[i] for i in keep] + [size],
                     [strides[i] for i in keep] + [strides[axis1] + strides[axis2]], start)
    v._readonly = True
    return v


def _real(val):
    if type(val) is not ndarray:
        return _NOPE
    if val.dtype.kind != "c":
        return val  # NumPy: the array itself
    return _viproc.view(val, val.shape, val.strides, 0, _CODES[_REAL_OF[val.dtype]])


_REAL_OF = {np.dtype(np.complex64): np.dtype(np.float32),
            np.dtype(np.complex128): np.dtype(np.float64)}


def _dtype_view(a, dtype):
    """a.view(dtype) as NumPy computes it (array_descr_set): the same layout
    for the same item size; else the last axis, which must be contiguous,
    rescaled. _NOPE for anything else (NumPy decides, and raises)."""
    code = _CODES.get(dtype)
    if code is None and not (dtype.subdtype and dtype.subdtype[0] in _CODES):
        return _NOPE
    if dtype.subdtype:  # (base, shape): a.view((float64, (2,))) appends axes
        base, sub = dtype.subdtype
        if base.itemsize * int(np.prod(sub)) != a.itemsize or not sub:
            return _NOPE
        strides = list(a.strides)
        inner = []
        step = base.itemsize
        for n in reversed(sub):
            inner.insert(0, step)
            step *= n
        return _viproc.view(a, a.shape + tuple(sub), tuple(strides) + tuple(inner), 0,
                            _CODES[base])
    if dtype.itemsize == a.itemsize:
        return _viproc.view(a, a.shape, a.strides, 0, code)
    if a.ndim == 0 or a.strides[-1] != a.itemsize or (a.shape[-1] * a.itemsize) % dtype.itemsize:
        return _NOPE
    shape = a.shape[:-1] + (a.shape[-1] * a.itemsize // dtype.itemsize,)
    return _viproc.view(a, shape, a.strides[:-1] + (dtype.itemsize,), 0, code)


def _reshape(a, shape=None, order="C", *, newshape=None, copy=None):
    """np.reshape: a view where NumPy makes one, else NumPy's copy (in the
    requested order) reshaped; copy=False raises as NumPy."""
    if type(a) is not ndarray:
        return _NOPE
    if newshape is not None:
        if shape is not None:
            return _NOPE  # NumPy raises
        shape = newshape
    if shape is None or order not in ("C", "F", "A"):
        return _NOPE
    shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
    if copy:
        src = a.astype(a.dtype, order="F" if order == "F" else "C")
        return _viproc.reshape(src, shape, order)
    r = _viproc.reshape(a, shape, order)
    if r is not None:
        return r
    if copy is False:
        raise ValueError("Unable to avoid creating a copy while reshaping.")
    if order == "A":
        order = "C"  # A on an array that is not F-contiguous
    r = _viproc.reshape(a.astype(a.dtype, order=order), shape, order)
    return _NOPE if r is None else r


def _ravel(a, order="C"):
    """PyArray_Ravel: a view where a is contiguous in the order (K: in C or
    F order), else a copy in that order (NumPy's PyArray_Flatten), also for
    a strided 1-d array that a reshape would only view."""
    if type(a) is not ndarray or order not in ("C", "F", "A", "K"):
        return _NOPE
    if a._c_contiguous and order != "F":
        return _reshape(a, -1, "C")
    f = np.lib.stride_tricks.as_strided(np.empty(1, a.dtype), a.shape,
                                        a.strides).flags.f_contiguous
    if order == "A":
        order = "F" if f else "C"
    elif order == "K":
        if not f and a.ndim > 1:
            return _NOPE  # a view in memory order, or a K-order copy
        order = "F" if f else "C"  # 1-d strided: the K copy is the C copy
    # copies own their data (PyArray_Flatten), as NumPy's: base None
    if order == "C":
        return _viproc.own(a.astype(a.dtype, order="C").reshape(-1))
    if f:
        return _reshape(a, -1, "F")
    return _viproc.own(_reshape(a.astype(a.dtype, order="F"), -1, "F"))


_VIEW_HANDLERS = {
    np.transpose: _transpose, np.permute_dims: _transpose, np.swapaxes: _swapaxes,
    np.moveaxis: _moveaxis, np.matrix_transpose: _matrix_transpose, np.squeeze: _squeeze,
    np.expand_dims: _expand_dims, np.flip: _flip, np.fliplr: _fliplr, np.flipud: _flipud,
    np.rot90: _rot90, np.atleast_1d: _atleast(1), np.atleast_2d: _atleast(2),
    np.atleast_3d: _atleast(3), np.broadcast_to: _broadcast_to, np.diagonal: _diagonal,
    np.real: _real, np.reshape: _reshape, np.ravel: _ravel,
}


def _where(condition, *xy):
    """np.where(cond, x, y) as a select task: the result dtype as NumPy's
    (np.result_type, Python scalars weak), x and y cast to it, cond as bool."""
    if len(xy) != 2:
        return _NOPE  # np.where(cond): nonzero, data-dependent shape
    try:
        dtype = np.result_type(*(a.dtype if isinstance(a, ndarray) else a for a in xy))
    except TypeError:
        return _NOPE
    if dtype not in _CODES:
        return _NOPE
    ops = []
    for a, want in ((condition, np.dtype(np.bool_)), (xy[0], dtype), (xy[1], dtype)):
        if isinstance(a, ndarray):
            if type(a) is not ndarray:
                return _NOPE
            ops.append(a if a.dtype == want else a.astype(want))
        else:
            ops.append(asarray(np.asarray(a).astype(want)))
    r = _viproc.select(*ops)
    return _scalar_result(r) if r.ndim == 0 else r


def _readonly_call(fn):
    """A NumPy function with a data-dependent result shape that only reads
    its argument (nonzero, ...): it syncs on the argument and runs in NumPy,
    on a read-only export instead of the general fallback, which would also
    wait for the argument's readers and make it writable first."""
    def handler(a, *args, **kwargs):
        global fallback_calls
        if type(a) is not ndarray:
            return _NOPE
        fallback_calls += 1
        r = fn(a.numpy(), *args, **kwargs)
        if isinstance(r, tuple):
            return tuple(_adopt(x) if isinstance(x, np.ndarray) and x.flags.owndata else x
                         for x in r)
        return _adopt(r) if isinstance(r, np.ndarray) and r.flags.owndata else r

    return handler


_VIEW_HANDLERS.update({np.where: _where,
                  np.nonzero: _readonly_call(np.nonzero),
                  np.flatnonzero: _readonly_call(np.flatnonzero),
                  np.argwhere: _readonly_call(np.argwhere),
                  np.count_nonzero: _readonly_call(np.count_nonzero)})


# --- advanced indexing ---------------------------------------------------------
# Integer array indices (and boolean masks, as their nonzero() indices) run
# as gather / scatter tasks: element copies, so any element order gives
# NumPy's values. NumPy's rules decide the result: the basic part of the key
# is applied first as a view; the index arrays broadcast together, and their
# dimensions replace the indexed axes if those are adjacent, else go first.
# NumPy's result layout comes from NumPy itself, indexing a tiny proxy with
# the same axis order (extents 1 or 2) once per key pattern (cached).
_layout_cache = {}


def _parse_index(a, key):
    """(view, index arrays, their view axes, result axes, broadcast shape,
    key pattern) for an advanced index, or None to leave the key to NumPy.
    Boolean masks sync (their nonzero() indices give the result's shape)."""
    keys = key if isinstance(key, tuple) else (key,)
    items = []  # ("b", basic item) / ("a", index) / ("m", mask as NumPy array)
    for k in keys:
        if k is None or k is Ellipsis or isinstance(k, slice):
            items.append(("b", k))
        elif isinstance(k, (bool, np.bool_)):
            return None
        elif isinstance(k, (int, np.integer)):
            items.append(("i", int(k)))
        elif isinstance(k, ndarray):
            if k.dtype == np.bool_:
                items.append(("m", k.numpy()))  # sync on the mask
            elif k.dtype in (np.int32, np.int64):
                items.append(("a", k))
            else:
                return None
        else:
            try:
                x = np.asarray(k)
            except AsyncError:
                raise  # a failed op's error: never swallowed (error-handling.md)
            except Exception:
                return None
            if x.dtype == np.bool_:
                items.append(("m", x))
            elif x.dtype.kind in "iu" and x.size:
                items.append(("a", x))
            else:
                return None
    if not any(t in ("a", "m") for t, _ in items):
        return None
    if sum(1 for t, k in items if k is Ellipsis) > 1:
        return None
    consumed = 0
    for t, k in items:
        if t == "m":
            if k.ndim == 0:
                return None
            consumed += k.ndim
        elif not (t == "b" and (k is None or k is Ellipsis)):
            consumed += 1
    if consumed > a.ndim:
        return None
    # The basic part as a view; `pattern` is the key for that view (slices
    # for its axes, the Ellipsis kept: NumPy's placement depends on it).
    basic, idx, axes, pos, pattern, masks = [], [], [], [], [], []
    vax = n = 0
    for t, k in items:
        if t == "b":
            basic.append(k)
            pattern.append(Ellipsis if k is Ellipsis else slice(None))
            vax += a.ndim - consumed if k is Ellipsis else 1
            n += 1
            continue
        group = np.nonzero(k) if t == "m" else (k,)  # a mask: one array per axis
        if t == "m":
            masks.append((vax, k.shape))
        for x in group:
            basic.append(slice(None))
            pattern.append(len(idx))
            idx.append(x)
            axes.append(vax)
            pos.append(n)
            vax += 1
            n += 1
    v = a._view(tuple(basic))
    for ax, shape in masks:
        if v.shape[ax:ax + len(shape)] != shape:
            return None  # NumPy raises
    arrays = []
    for k in idx:
        if isinstance(k, ndarray):
            arrays.append(k)
        else:
            x = np.asarray(k)
            if x.dtype not in (np.int32, np.int64):
                x = x.astype(np.intp)
            arrays.append(asarray(x))
    try:
        bshape = np.broadcast_shapes(*(x.shape for x in arrays))
    except ValueError:
        return None
    if 0 in v.shape or 0 in bshape:
        return None
    rest = [k for k in range(v.ndim) if k not in axes]
    bdims = [-1 - d for d in range(len(bshape))]
    # Adjacent in the key (NumPy: any other item between them, even an
    # Ellipsis standing for no axes, puts the broadcast dimensions first).
    if pos == list(range(pos[0], pos[0] + len(pos))):
        res = [k for k in rest if k < axes[0]] + bdims + [k for k in rest if k > axes[-1]]
    else:
        res = bdims + rest
    return v, arrays, axes, res, bshape, tuple(pattern)


def _result_strides(v, arrays, axes, res, bshape, pattern):
    """NumPy's strides for the indexing result (see above)."""
    order = sorted(range(v.ndim), key=lambda k: -abs(v.strides[k]))
    sig = (v.ndim, tuple(order), tuple(st < 0 for st in v.strides),
           tuple(st == 0 for st in v.strides), tuple(n == 1 for n in v.shape), pattern,
           tuple(tuple(n == 1 for n in x.shape) for x in arrays), v.itemsize)
    perm = _layout_cache.get(sig)
    if perm is None:
        small = [1 if n == 1 else 2 for n in v.shape]
        pstr = [0] * v.ndim
        step = v.itemsize
        for k in reversed(order):
            if v.strides[k] != 0:
                pstr[k] = step
                step *= small[k]
        buf = np.zeros(max(step // v.itemsize, 1), v.dtype)
        p = np.lib.stride_tricks.as_strided(buf, shape=small, strides=pstr)
        p = p[tuple(slice(None, None, -1) if st < 0 else slice(None) for st in v.strides)]
        key = tuple(np.zeros([1 if n == 1 else 2 for n in arrays[k].shape], np.intp)
                    if isinstance(k, int) else k for k in pattern)
        r = p[key]
        # Axes of extent 2 in memory order (innermost first); an extent-1
        # axis has the stride of the extent-2 axes inside it: how many, its
        # stride tells.
        inner = sorted((k for k in range(r.ndim) if r.shape[k] != 1), key=lambda k: r.strides[k])
        ones = {k: sum(1 for j in inner if r.strides[j] < r.strides[k])
                for k in range(r.ndim) if r.shape[k] == 1}
        perm = (inner, ones)
        if len(_layout_cache) > 1024:
            _layout_cache.clear()
        _layout_cache[sig] = perm
    inner, ones = perm
    shape = [v.shape[k] if k >= 0 else bshape[-1 - k] for k in res]
    strides = [0] * len(shape)
    steps = [v.itemsize]
    for k in inner:
        strides[k] = steps[-1]
        steps.append(steps[-1] * shape[k])
    for k, c in ones.items():
        strides[k] = steps[c]
    return strides


_INDEX_DTYPES = (np.dtype(np.int64), np.dtype(np.int32))


_FULL = slice(None)


def _gather_fast(a, key):
    """a[idx] or a[:, ..., :, idx] for one integer index array on
    C-contiguous data, the common cases, without the general parsing. NumPy's
    result layout there: the index dimensions outermost in memory, the other
    axes inside them in C order (for a[idx] that is C order). None if the key
    is not like that."""
    if type(key) is tuple:
        p = len(key) - 1
        if p < 0 or p >= a.ndim or any(type(k) is not slice or k != _FULL for k in key[:p]):
            return None
        idx = key[p]
    else:
        p, idx = 0, key
    if type(idx) is not ndarray:
        if type(idx) is not np.ndarray or idx.dtype.kind not in "iu" or not idx.ndim:
            return None
        idx = asarray(idx if idx.dtype in _INDEX_DTYPES else idx.astype(np.intp))
    if (idx.dtype not in _INDEX_DTYPES or not a.ndim or not a.size or not idx.size
            or not a._c_contiguous):
        return None
    nb = idx.ndim
    res = list(range(p)) + [-1 - d for d in range(nb)] + list(range(p + 1, a.ndim))
    if p == 0:
        return _viproc.gather(a, (idx,), (0,), res, None)
    shape = a.shape[:p] + idx.shape + a.shape[p + 1:]
    strides = [0] * len(shape)
    step = a.itemsize
    for k in reversed([k for k in range(len(shape)) if not p <= k < p + nb]):
        strides[k] = step
        step *= shape[k]
    for k in reversed(range(p, p + nb)):
        strides[k] = step
        step *= shape[k]
    return _viproc.gather(a, (idx,), (p,), res, strides)


def _gather(a, key):
    parsed = _parse_index(a, key)
    if parsed is None:
        return _NOPE
    v, arrays, axes, res, bshape, pattern = parsed
    strides = _result_strides(v, arrays, axes, res, bshape, pattern)
    return _viproc.gather(v, arrays, axes, res, strides)


def _as_values(value, dtype):
    """`value` as a viproc array of `dtype` (NumPy's setitem conversion)."""
    if isinstance(value, ndarray):
        return value if value.dtype == dtype else value.astype(dtype)
    if np.ndim(value) == 0 and not isinstance(value, np.ndarray):
        t = np.empty((), dtype)
        t[()] = value  # NumPy's conversion of a[i] = value (errors included)
        return _adopt(t)
    return _adopt(np.array(value, dtype=dtype))  # as NumPy's setitem: a cast copy


def _scatter(a, key, value):
    vdim = value.ndim if isinstance(value, ndarray) else np.ndim(value)  # np.ndim would export
    if isinstance(key, (ndarray, np.ndarray)) and key.dtype == np.bool_ and (
            vdim == 0 and key.ndim and key.shape == a.shape[:key.ndim]):
        # a[mask] = scalar: no count needed (where-assignment, mask broadcast
        # over the remaining axes)
        m = key if isinstance(key, ndarray) else asarray(key)
        m = m[(Ellipsis,) + (None,) * (a.ndim - key.ndim)]
        _viproc.mask_assign(a, m, _as_values(value, a.dtype))
        return True
    parsed = _parse_index(a, key)
    if parsed is None:
        return False
    v, arrays, axes, res, bshape, _ = parsed
    vals = _as_values(value, a.dtype)
    shape = tuple(v.shape[k] if k >= 0 else bshape[-1 - k] for k in res)
    try:
        if np.broadcast_shapes(vals.shape, shape) != shape:
            return False
    except ValueError:
        return False  # NumPy raises
    _viproc.scatter(v, arrays, axes, res, vals)
    return True


# Ops on at most this many elements whose data is ready run right away
# (Runtime::issue's direct path; small advanced indexing in NumPy itself).
_small_ops = [2048]


def _ready_key(key):
    keys = key if isinstance(key, tuple) else (key,)
    return all(not isinstance(k, ndarray) or _viproc.ready(k) for k in keys)


def _numpy_key(key):
    if isinstance(key, tuple):
        return tuple(k.numpy() if isinstance(k, ndarray) else k for k in key)
    return key.numpy() if isinstance(key, ndarray) else key


def _is_basic_index(key):
    key = key if isinstance(key, tuple) else (key,)
    for k in key:
        if k is None or k is Ellipsis or isinstance(k, slice):
            continue
        if isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)):
            continue
        return False
    return True



def _numpy_error(fn):
    try:
        fn()
    except Exception as e:
        return str(e)
    raise AssertionError("NumPy raised no error")


# ndarray.resize: NumPy's messages, and its strides (_array_fill_strides).
_MAX_INTP = np.iinfo(np.intp).max
_RESIZE_SEGMENT = _numpy_error(lambda: np.empty(3)[::2].resize(1))
_RESIZE_NEGATIVE = _numpy_error(lambda: np.empty(1).resize(-1))


def _fill_strides(shape, itemsize, fortran):
    strides = [0] * len(shape)
    for i in range(len(shape)) if fortran else reversed(range(len(shape))):
        strides[i] = itemsize
        if shape[i]:
            itemsize *= shape[i]
    return tuple(strides)


class ndarray(_viproc._Array):
    """Array whose ops run asynchronously in the viproc runtime.

    Supported ops return immediately with a pending result; reading data
    (printing, ``np.asarray``, ``float()``, unsupported functions) is a sync
    point. ``shape``, ``strides``, ``dtype`` and ``ndim`` come from the C base
    class and never sync.
    """

    __slots__ = ()
    __array_priority__ = 100

    # --- metadata (no sync) -------------------------------------------------
    @property
    def itemsize(self):
        return self.dtype.itemsize

    @property
    def nbytes(self):
        return self.size * self.itemsize

    def __len__(self):
        if not self.shape:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    # --- sync points ----------------------------------------------------------
    def numpy(self):
        """Sync point: the data as a read-only NumPy array, without a copy.

        The NumPy array keeps the values it shows: later writes to this
        viproc array go to a new buffer (copy-on-write). Use ``np.array(x)``
        or ``x.numpy().copy()`` for a writable copy."""
        return _viproc.export(self)

    def __array__(self, dtype=None, copy=None):
        arr = self.numpy()
        if dtype is not None and np.dtype(dtype) != arr.dtype:
            if copy is False:
                raise ValueError("viproc: converting the dtype needs a copy")
            return arr.astype(dtype)
        return arr.copy() if copy else arr

    def wait(self):
        """Sync point: wait until the data is computed (raises AsyncError)."""
        _viproc.wait(self)
        return self

    def ready(self):
        return _viproc.ready(self)

    def __repr__(self):
        return "viproc." + repr(self.numpy())

    def __str__(self):
        return str(self.numpy())

    def __bool__(self):
        return bool(self.numpy())

    def __float__(self):
        return float(self.numpy())

    def __int__(self):
        return int(self.numpy())

    def __complex__(self):
        return complex(self.numpy())

    def __format__(self, spec):  # sync point; 0-d results format like NumPy scalars
        return format(self.numpy()[()] if self.ndim == 0 else self.numpy(), spec)

    def __round__(self, ndigits=None):  # sync point
        v = self.numpy()[()] if self.ndim == 0 else self.numpy()
        return round(v) if ndigits is None else round(v, ndigits)

    def __index__(self):
        return self.numpy().__index__()

    def __iter__(self):
        if self.ndim > 1:
            return (self[i] for i in range(self.shape[0]))  # row views, as NumPy
        return iter(self.numpy())

    def item(self, *args):
        return self.numpy().item(*args)

    def tolist(self):
        return self.numpy().tolist()

    # --- indexing ------------------------------------------------------------
    def _view(self, key):
        """Basic indexing as a runtime view (metadata only, no sync; an
        all-int key gives a 0-d view)."""
        return _basic(self, key)

    # --- NumPy protocols -------------------------------------------------------
    # __array_function__ is _Array's (C): reductions there, everything else
    # here.
    def _array_function(self, func, types, args, kwargs):
        if func is np.copyto and len(args) >= 2 and isinstance(args[0], ndarray) and not kwargs:
            args[0][...] = args[1]
            return None
        handler = (_REDUCTIONS.get(func) or _VIEW_HANDLERS.get(func) or _PRODUCTS.get(func)
                   or _COMPOSITIONS.get(func) or _JOINING.get(func) or _INDEXING.get(func)
                   or _ACCUMULATING.get(func))
        if handler is not None:
            try:
                r = handler(*args, **kwargs)
            except TypeError:  # unexpected arguments: NumPy reports them
                r = _NOPE
            if r is not _NOPE:
                return r
        return _host_call(func, args, kwargs)

    # --- common methods (NumPy semantics via the protocols) -------------------
    # sum, prod, max, min, any, all, mean, var, std, argmax, argmin, copy and
    # astype are _Array's C methods (setup_reductions below); these are the
    # Python paths behind copy and astype.
    def _astype_py(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        dtype = np.dtype(dtype)
        if dtype == self.dtype and not copy:
            return self
        code = _CODES.get(dtype)
        if (code is not None and order in ("K", "C", "F") and casting == "unsafe"
                and self.size and type(self) is ndarray):
            r = _viproc.astype(self, code, order)  # NumPy's casts, asynchronously
            if r is not None:
                return r
        return _host_call(lambda a: a.astype(dtype, order, casting, subok, copy), (self,), {})

    def _copy_py(self, order="C"):
        if order != "C":
            return _host_call(lambda a: a.copy(order), (self,), {})
        out = _wrap_numpy(np.empty(self.shape, self.dtype))
        _viproc.assign(out, self)
        return out

    def _reshape_py(self, *shape, **kwargs):  # behind _Array.reshape (C)
        shape = shape[0] if len(shape) == 1 else shape
        r = _reshape(self, shape, **kwargs) if not kwargs else _NOPE
        return np.reshape(self, shape, **kwargs) if r is _NOPE else r

    def ravel(self, order="C"):
        return np.ravel(self, order)

    def flatten(self, order="C"):
        if order == "C" and type(self) is ndarray:
            return _viproc.own(self.astype(self.dtype, order="C").reshape(-1))  # a copy
        return _host_call(lambda a: a.flatten(order), (self,), {})

    @property
    def T(self):
        return _viproc.transpose(self, None)

    @property
    def mT(self):
        return _matrix_transpose(self)

    def transpose(self, *axes):
        if len(axes) == 1 and (axes[0] is None or isinstance(axes[0], (tuple, list))):
            axes = axes[0]
        return _transpose(self, axes if axes != () else None)

    def _swapaxes_py(self, axis1, axis2):  # behind _Array.swapaxes (C)
        return _swapaxes(self, axis1, axis2)

    def squeeze(self, axis=None):
        return _squeeze(self, axis)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return _diagonal(self, offset, axis1, axis2)

    @property
    def real(self):
        return _real(self)

    @property
    def imag(self):
        if self.dtype.kind == "c":  # the second half of each element
            return _viproc.view(self, self.shape, self.strides, self.itemsize // 2,
                                _CODES[_REAL_OF[self.dtype]])
        return _host_call(lambda a: a.imag, (self,), {})  # NumPy: read-only zeros

    def view(self, dtype=None, type=None):
        if type is None and dtype is not None and not (
                isinstance(dtype, builtins.type) and issubclass(dtype, np.ndarray)):
            r = _dtype_view(self, np.dtype(dtype))
            if r is not _NOPE:
                return r
        return _host_call(lambda a: a.view(dtype) if type is None else a.view(dtype, type),
                          (self,), {})

    # sum, prod, min, max, mean, std, var, any, all, argmin, argmax: _Array's
    # (C), with _reduction_fallback below for what C does not handle.

    def dot(self, b, out=None):
        return np.dot(self, b, out=out)

    # --- the other ndarray methods ---------------------------------------------
    # Delegates to NumPy's function forms, so they take whatever path the
    # function takes (asynchronous where viproc supports it, else the eager
    # fallback) and get faster with it. In-place methods without a function
    # form run NumPy's method in the fallback, on the writable view it makes
    # after waiting for readers (vp_array_prepare_write).
    def conj(self):
        # PyArray_Conjugate: a real array is returned as is (no copy).
        return np.conjugate(self) if self.dtype.kind == "c" else self

    def conjugate(self):
        return np.conjugate(self) if self.dtype.kind == "c" else self

    def round(self, decimals=0, out=None):
        return np.round(self, decimals, out)

    def clip(self, min=None, max=None, out=None, **kwargs):
        return np.clip(self, min, max, out, **kwargs)

    def cumsum(self, axis=None, dtype=None, out=None):
        return np.cumsum(self, axis, dtype, out)

    def cumprod(self, axis=None, dtype=None, out=None):
        return np.cumprod(self, axis, dtype, out)

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        return np.trace(self, offset, axis1, axis2, dtype, out)

    def nonzero(self):
        return np.nonzero(self)

    def argsort(self, *args, **kwargs):
        return np.argsort(self, *args, **kwargs)

    def argpartition(self, *args, **kwargs):
        return np.argpartition(self, *args, **kwargs)

    def searchsorted(self, v, side="left", sorter=None):
        return np.searchsorted(self, v, side, sorter)

    def take(self, indices, axis=None, out=None, mode="raise"):
        return np.take(self, indices, axis, out, mode)

    def put(self, indices, values, mode="raise"):
        np.put(self, indices, values, mode)

    def choose(self, choices, out=None, mode="raise"):
        return np.choose(self, choices, out, mode)

    def compress(self, condition, axis=None, out=None):
        return np.compress(condition, self, axis, out)

    def repeat(self, repeats, axis=None):
        return np.repeat(self, repeats, axis)

    def sort(self, *args, **kwargs):
        _host_call(lambda a, *ar, **kw: a.sort(*ar, **kw), (self,) + args, kwargs)

    def partition(self, *args, **kwargs):
        _host_call(lambda a, *ar, **kw: a.partition(*ar, **kw), (self,) + args, kwargs)

    def _resize_py(self, args, kwargs, refs):  # behind _Array.resize (C)
        # NumPy's array_resize and PyArray_Resize, on viproc data. NumPy
        # parses the arguments and raises its errors on tiny NumPy probes in
        # the same state (contiguity, sizes, OWNDATA, weak references,
        # reference count), so they are NumPy's in every version; the data
        # step is asynchronous. `refs`: the object's reference count, taken
        # where NumPy takes it (in the C method).
        dt = self.dtype
        kwargs = kwargs or {}
        try:  # parses as NumPy does, then stops at the single-segment check
            np.empty(3, dt)[::2].resize(*args, **kwargs)
            return None  # no shape (a.resize(), a.resize(None)): nothing to do
        except ValueError as e:
            if str(e) != _RESIZE_SEGMENT:
                raise
            segment = e
        refcheck = operator.index(kwargs.get("refcheck", 1))  # NumPy accepted it
        shape = args[0] if len(args) == 1 else args
        try:
            shape = (operator.index(shape),)
        except TypeError:
            shape = tuple(operator.index(n) for n in shape)
        c, f = self._c_contiguous, self.T._c_contiguous
        if not (c or f):
            raise segment
        size = 1
        for n in shape:
            if n == 0:
                size = 0
                break
            if n < 0 or size * n > _MAX_INTP:
                np.empty(1, dt).resize(shape)  # NumPy's error: fails before allocating
            size *= n
        itemsize = dt.itemsize
        if size * itemsize > _MAX_INTP:
            np.empty(1, dt).resize(shape)  # MemoryError
        if any(n < 0 for n in shape):
            # NumPy accepts a negative extent after a zero one (its size loop
            # stops at the zero); viproc's layouts cannot hold it.
            raise ValueError(_RESIZE_NEGATIVE)
        strides = _fill_strides(shape, itemsize, f and not c)
        if size == self.size:  # same bytes: only shape and strides change
            return _viproc.view(self, shape, strides, 0)
        other = 2 if size == 1 else 1  # a probe size whose bytes differ, too
        if self.base is not None:  # not OWNDATA
            np.empty(other, dt)[:].resize(shape)
        if weakref.getweakrefcount(self):
            probe = np.empty(other, dt)
            ref = weakref.ref(probe)  # noqa: F841 (alive during the call)
            probe.resize(shape)
        if refcheck and (refs != 1 if sys.version_info >= (3, 14) else refs > 2):
            _viproc.resize_refcheck(lambda: np.empty(other, dt), refs, shape)
        # PyDataMem_UserRENEW and zeros: the old bytes in memory order first.
        m = min(size, self.size)
        flat = _viproc.view(self, (self.size,), (itemsize,), 0)
        out = _wrap_numpy(np.empty(size, dt))
        if m:
            out[:m] = flat[:m]
        if size > m:
            out[m:] = 0
        return _viproc.view(out, shape, strides, 0)

    def byteswap(self, inplace=False):
        return _host_call(lambda a: a.byteswap(inplace), (self,), {})

    def getfield(self, dtype, offset=0):
        return _host_call(lambda a: a.getfield(dtype, offset), (self,), {})

    def setfield(self, val, dtype, offset=0):
        _host_call(lambda a, v: a.setfield(v, dtype, offset), (self, val), {})

    def fill(self, value):
        if (value.ndim if isinstance(value, ndarray) else np.ndim(value)) != 0:  # NumPy's error
            _host_call(lambda a: a.fill(value), (self,), {})
            return
        self[...] = value  # an assignment task (vp_assign), as NumPy casts

    # Sync points: they read the data.
    @property
    def flat(self):
        return _FlatIter(self)

    @property
    def data(self):
        return self.numpy().data  # read-only, as every export

    def tobytes(self, order="C"):
        return self.numpy().tobytes(order)

    def tofile(self, fid, sep="", format="%s"):
        self.numpy().tofile(fid, sep, format)

    def dumps(self):
        return self.numpy().dumps()

    def dump(self, file):
        self.numpy().dump(file)

    # Metadata (no sync).
    @property
    def flags(self):
        return _Flags(self)

    def setflags(self, write=None, align=None, uic=None):
        if uic:
            raise ValueError("cannot set WRITEBACKIFCOPY flag to True")
        # align: viproc's memory is always aligned; NumPy accepts both values.
        if write is not None:
            if write and self._readonly:
                raise ValueError("cannot set WRITEABLE flag to True of this array")
            self._readonly = not write

    @property
    def device(self):
        return "cpu"

    def to_device(self, device, /, *, stream=None):
        if stream is not None:
            raise ValueError("The stream argument in to_device() is not supported")
        if device != "cpu":
            raise ValueError(f"Unsupported device {device!r}. Only 'cpu' is accepted.")
        return self


class _Flags:
    """ndarray.flags of a viproc array, from its layout (no sync): NumPy's
    flags of a probe with the same shape and strides over one element, with
    WRITEABLE from the viproc array (settable, as NumPy's) and OWNDATA as
    NumPy has it: True unless the array is a view (`base` is set)."""

    __slots__ = ("_a", "_f")

    def __init__(self, a):
        probe = np.lib.stride_tricks.as_strided(np.empty(1, a.dtype), a.shape, a.strides)
        object.__setattr__(self, "_a", a)
        object.__setattr__(self, "_f", probe.flags)

    @property
    def writeable(self):
        return not self._a._readonly

    @writeable.setter
    def writeable(self, value):
        self._a.setflags(write=value)

    @property
    def owndata(self):
        return self._a.base is None

    def __getattr__(self, name):
        return getattr(self._f, name)

    def __setattr__(self, name, value):
        if name != "writeable":
            raise AttributeError(f"viproc: flag {name!r} cannot be set")
        type(self).writeable.fset(self, value)

    def __getitem__(self, key):
        if key in ("W", "WRITEABLE"):
            return self.writeable
        if key in ("O", "OWNDATA"):
            return self.owndata
        return self._f[key]

    def __setitem__(self, key, value):
        if key not in ("W", "WRITEABLE"):
            raise KeyError(f"viproc: flag {key!r} cannot be set")
        self.writeable = value

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __repr__(self):
        return repr(self._f).replace(
            f"OWNDATA : {self._f.owndata}", f"OWNDATA : {self.owndata}").replace(
            f"WRITEABLE : {self._f.writeable}", f"WRITEABLE : {self.writeable}")


class _FlatIter:
    """a.flat of a viproc array: reading syncs (NumPy's flatiter on the
    read-only export), assigning runs NumPy's in the eager fallback."""

    __slots__ = ("base",)

    def __init__(self, a):
        self.base = a

    def __len__(self):
        return self.base.size

    def __iter__(self):
        return iter(self.base.numpy().flat)

    def __getitem__(self, key):
        r = self.base.numpy().flat[_numpy_key(key)]
        return _adopt(r) if isinstance(r, np.ndarray) else r  # NumPy copies

    def __setitem__(self, key, value):
        def assign(a, k, v):
            a.flat[k] = v

        _host_call(assign, (self.base, key, value), {})

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.base.numpy().flat.copy(), dtype=dtype)

    def copy(self):
        return self.base.flatten()


def _ufunc_reduce(ufunc, a, axis=0, dtype=None, out=None, keepdims=False, initial=_NV,
                  where=True):
    if isinstance(out, tuple):  # NumPy's normalized out=
        if len(out) != 1:
            return None
        out = out[0]
    if where is not True:
        return None
    return _reduce(ufunc, a, axis, keepdims, dtype, initial, out)


def _product_operands(inputs):
    """Both operands of a matrix product as viproc arrays (NumPy arrays are
    imported), or None if NumPy has to run it."""
    if len(inputs) != 2 or not all(isinstance(x, (ndarray, np.ndarray)) for x in inputs):
        return None
    ops = tuple(x if isinstance(x, ndarray) else asarray(x) for x in inputs)
    if not all(isinstance(x, ndarray) for x in ops) or ops[0].dtype != ops[1].dtype:
        return None  # unsupported dtype, or a cast (NumPy's iterator copies)
    return ops


def _matmul(a, b):
    ops = _product_operands((a, b))
    return _viproc.matmul(*ops) if ops is not None else None


def _dot_operand(x):
    """x as NumPy's dot passes it to BLAS (cblas_matrixproduct): copied to C
    order if its strides are not BLAS-able (negative, zero, not a multiple
    of the item size) or, for a matrix, if it is neither C- nor
    F-contiguous."""
    st, n = x.strides, x.itemsize
    bad = any(s <= 0 or s % n for s in st)
    if x.ndim == 2 and not bad:
        c = st[1] == n and st[0] == x.shape[1] * n
        f = st[0] == n and st[1] == x.shape[0] * n
        bad = not (c or f)
    return x.astype(x.dtype, order="C") if bad else x


def _dot(a, b, out=None):
    """np.dot of vectors and matrices as matmul where both make the same loop
    calls: NumPy's dot gives integers and bool the values of matmul (exact
    sums), and for floating-point matrices and vectors with every extent >= 2
    makes the BLAS call matmul makes (gemm/syrk, gemv, dot), once its
    operands are prepared as dot prepares them. Everything else (0-d and
    n-d operands, extents of 1, out=) runs in NumPy."""
    ops = _product_operands((a, b)) if out is None else None
    if ops is None or not all(1 <= x.ndim <= 2 for x in ops):
        return _NOPE
    if ops[0].dtype.kind in "fc":
        if any(e < 2 for x in ops for e in x.shape):
            return _NOPE  # dot's scalar/row/column special cases
        ops = tuple(_dot_operand(x) for x in ops)
    r = _viproc.matmul(*ops)
    return _NOPE if r is None else r


_PRODUCTS = {np.dot: _dot}


def _fallback_ufunc(ufunc, method, inputs, kwargs):
    """ufunc calls the C fast path leaves out (_ufuncs.call), reduce,
    accumulate, outer and matmul asynchronously where possible; anything
    else syncs and runs eagerly in NumPy (in place for out= and ufunc.at)."""
    if method == "__call__" and ufunc is np.matmul and not kwargs:
        r = _matmul(*inputs) if len(inputs) == 2 else None
        if r is not None:
            return r
    if method == "reduce" and len(inputs) == 1:
        try:
            r = _ufunc_reduce(ufunc, *inputs, **kwargs)
        except TypeError:
            r = None
        if r is not None:
            return r
    if method in ("__call__", "accumulate", "outer", "reduceat", "at"):
        r = getattr(_ufuncs, "call" if method == "__call__" else method)(ufunc, inputs, kwargs)
        if r is not _NOPE:
            return r
    return _host_call(getattr(ufunc, method), inputs, kwargs)


# Hooks for the C fast path (see _viproc.c).
_OPS = {
    name: getattr(np, name)
    for name in (
        "add", "subtract", "multiply", "true_divide", "floor_divide", "remainder", "divmod",
        "power", "left_shift", "right_shift", "bitwise_and", "bitwise_or", "bitwise_xor",
        "matmul", "negative", "positive", "absolute", "invert", "less", "less_equal",
        "equal", "not_equal", "greater", "greater_equal",
    )
}
def _setitem(self, key, value):
    """a[key] = value beyond _Array's C slot (which hands small work on ready
    data to NumPy and issues basic-key assignments of scalars and viproc
    arrays): advanced keys, sequences as values, NumPy's errors."""
    if not _is_basic_index(key):  # NumPy assigns in place, overlap included
        if not self._readonly and type(self) is ndarray:
            try:
                done = _scatter(self, key, value)
            except AsyncError:
                raise
            except (TypeError, ValueError, IndexError, OverflowError):
                done = False  # NumPy raises its own error
            if done:
                return

        def assign(a, k, v):
            a[k] = v

        _host_call(assign, (self, key, value), {})
        return
    if self._readonly:
        raise ValueError("assignment destination is read-only")
    target = self._view(key)
    if isinstance(value, ndarray):
        src = value  # another dtype: cast in the assignment (vp_assign)
    else:
        src = asarray(np.asarray(value).astype(self.dtype, copy=False))
    try:
        _viproc.assign(target, src)
    except NotImplementedError:  # a cast NumPy refused (warning turned into an error)
        _viproc.assign(target, asarray(np.asarray(src).astype(self.dtype, copy=False)))


def _getitem(self, key):
    """a[key] beyond the C fast path (_Array.__getitem__ handles integer
    keys of ready arrays and basic keys itself): advanced indexing, NumPy's
    scalar of a pending a[i, j], errors."""
    if _is_basic_index(key):
        v = _basic(self, key)  # raises NumPy's error for an invalid key
        if v.ndim == 0:
            # NumPy returns a scalar: free if the data is there, else a
            # 0-d copy (a snapshot, as NumPy's scalar) instead of a sync
            return _scalar_result(v.copy()) if not _viproc.ready(v) else _viproc.scalar(v)
        return v
    if type(self) is ndarray:
        if type(key) in (ndarray, tuple, np.ndarray):  # a[idx], a[:, idx]: fast path
            r = _gather_fast(self, key)
            if r is not None:
                return r
        if self.size <= _small_ops[0] and _viproc.ready(self) and _ready_key(key):
            # Small and ready: nothing to overlap, NumPy indexes right away
            # (advanced indexing always copies: the result is fresh).
            return _adopt(np.asarray(self.numpy()[_numpy_key(key)]))
        try:
            r = _gather(self, key)
        except AsyncError:
            raise
        except (TypeError, ValueError, IndexError):
            r = _NOPE  # NumPy raises its own error
        if r is not _NOPE:
            return r
    return _host_call(lambda a, k: a[k], (self, key), {})


def _reduction_fallback(func, args, kwargs):
    """np.<reduction>(*args, **kwargs) beyond the C path (_Array's methods and
    __array_function__): the asynchronous handlers, else NumPy."""
    handler = _REDUCTIONS[func]
    try:
        r = handler(*args, **kwargs)
    except TypeError:  # unexpected arguments: NumPy reports them
        r = _NOPE
    return _host_call(func, args, kwargs) if r is _NOPE else r


_viproc.setup_reductions(
    tuple(
        (f, m, k, u, lambda args, kwargs, f=f: _reduction_fallback(f, args, kwargs))
        for f, m, k, u in (
            (np.sum, "sum", 0, np.add),
            (np.prod, "prod", 0, np.multiply),
            (np.max, "max", 1, np.maximum),
            (np.min, "min", 1, np.minimum),
            (np.amax, None, 1, np.maximum),
            (np.amin, None, 1, np.minimum),
            (np.any, "any", 2, np.logical_or),
            (np.all, "all", 2, np.logical_and),
            (np.mean, "mean", 3, None),
            (np.var, "var", 3, None),
            (np.std, "std", 3, None),
            (np.argmax, "argmax", 3, None),
            (np.argmin, "argmin", 3, None),
        )
    )
    + (
        (None, "copy", 3, None, lambda args, kwargs: ndarray._copy_py(*args, **kwargs)),
        (None, "astype", 3, None, lambda args, kwargs: ndarray._astype_py(*args, **kwargs)),
    ),
    ndarray._array_function,
    np._NoValue,
    _REDUCE_ASYNC,
)

_viproc.setup(
    ndarray,
    tuple(_DTYPES[c] for c in range(7)),
    _fallback_ufunc,
    asarray,
    (_PKG_DIR, _NUMPY_DIR),
    _OPS,
    np.ndarray,
    np.generic,
    _getitem,
    _setitem,
)


def wait_all():
    """Sync point: wait until all issued ops have completed."""
    if _viproc.initialized():
        _viproc.wait_all()


def init(workers=0, max_active_tasks=0, sync_below=None, adaptive=None):
    """(Re)starts the runtime.

    workers: worker threads (0: one per physical core).
    sync_below: ops whose inputs are ready and that produce fewer elements run
        inline on the calling thread (default 2048, env VIPROC_SYNC_BELOW).
    adaptive: let each issue site switch to inline execution while its results
        are usually needed right after issuing (default True; False, or env
        VIPROC_DETERMINISTIC=1, gives a deterministic schedule). Results are
        identical either way; only where ops run changes.
    """
    env_sync, env_adaptive = _env_policy()
    sync_below = env_sync if sync_below is None else sync_below
    try:
        shutdown()  # raises an error of the old runtime nothing reported yet
    finally:
        _viproc.init(workers, max_active_tasks, sync_below,
                     env_adaptive if adaptive is None else adaptive)
        _small_ops[0] = 2048 if sync_below < 0 else sync_below


def offload_stats():
    """{'inline': n, 'offloaded': m, 'direct': d, 'tasks': t, 'syncs': s}:
    ready ops run inline as a task / handed to workers / run directly without
    a task (small, inputs ready); t counts every task issued, whether its
    inputs were ready or pending (inline and offloaded count only the ready
    ones); s counts sync points that found their array pending (the main
    thread blocked; wait_all() not counted)."""
    _ensure_runtime()
    return _viproc.offload_counts()


def pool_stats():
    """Element memory pool: {'hits', 'misses', 'cached', 'dropped', 'limit'}.

    hits/misses count requests served from the pool / from the system,
    cached is the number of bytes the pool holds, dropped the releases that
    went back to the system because of the bound (limit, bytes)."""
    return _viproc.pool_stats()


def trim():
    """Return all element memory cached by the pool to the system."""
    _viproc.pool_trim()


def set_pool_limit(nbytes):
    """Bound for the bytes the element pool may cache (0: no caching)."""
    _viproc.pool_set_limit(int(nbytes))


def shutdown():
    """Waits for all ops and stops the runtime. Raises the error of a failed op
    that was not reported yet (at exit Python prints it)."""
    if _viproc.initialized():
        _viproc.shutdown()


def workers():
    _ensure_runtime()
    return _viproc.workers()


from ._compose import COMPOSITIONS as _COMPOSITIONS  # noqa: E402 (uses this module)
from ._join import JOINING as _JOINING  # noqa: E402
from ._indexing import INDEXING as _INDEXING  # noqa: E402
from . import _ufuncs  # noqa: E402
from ._ufuncs import ACCUMULATING as _ACCUMULATING  # noqa: E402
