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
import os
import warnings

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

from . import _viproc

AsyncError = _viproc.AsyncError

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


def asarray(obj, dtype=None):
    """viproc array from `obj` (copies NumPy data; viproc arrays pass through)."""
    if isinstance(obj, ndarray) and (dtype is None or obj.dtype == np.dtype(dtype)):
        return obj
    arr = np.array(obj, dtype=dtype, copy=True)
    wrapped = _wrap_numpy(arr)
    if wrapped is None:
        raise TypeError(f"viproc: unsupported dtype {arr.dtype}")
    return wrapped


array = asarray


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


def _reduce(ufunc, a, axis=None, keepdims=False, dtype=None, initial=_NV):
    """ufunc.reduce(a, axis, dtype=..., keepdims=..., initial=...) issued
    asynchronously, or None if viproc cannot run it (the caller falls back to
    NumPy; empty inputs too, vp_reduce refuses them). Casting reductions and
    strided ones NumPy buffers run on NumPy's own iterator (vp_reduce)."""
    if not _REDUCE_ASYNC or type(a) is not ndarray or ufunc not in _REDUCE_UFUNCS:
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
        if out is not None or where is not _NV:
            return _NOPE
        r = _reduce(ufunc, a, axis, keepdims is True, dtype, initial)
        return _NOPE if r is None else r

    return handler


def _r_minmax(ufunc):
    def handler(a, axis=None, out=None, keepdims=_NV, initial=_NV, where=_NV):
        if out is not None or where is not _NV:
            return _NOPE
        r = _reduce(ufunc, a, axis, keepdims is True, None, initial)
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
            src = a.reshape(-1)
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
# NumPy functions that only rearrange metadata (shape, strides, offset, dtype)
# run on a stand-in: an ndarray with the viproc array's shape and strides over
# a one-element dummy. Its data is never read (these functions do not touch
# it); the view NumPy returns is mapped onto a runtime view of the same
# storage. NumPy validates the arguments and raises its own errors.
_VIEW_FUNCS = frozenset((
    np.transpose, np.swapaxes, np.moveaxis, np.rollaxis, np.squeeze, np.expand_dims,
    np.flip, np.fliplr, np.flipud, np.rot90, np.atleast_1d, np.atleast_2d, np.atleast_3d,
    np.broadcast_to, np.diagonal, np.real, np.matrix_transpose, np.permute_dims,
))


def _meta_view(fn, a, args=(), kwargs=None):
    """fn(a, *args, **kwargs) as a runtime view of `a` if NumPy's result is a
    view of its argument, else _NOPE. `fn` must not read data."""
    if type(a) is not ndarray:
        return _NOPE
    fake = _viproc.stand_in(a)
    r = fn(fake, *args, **(kwargs or {})) if kwargs else fn(fake, *args)
    if r is fake:
        return a  # NumPy returns its argument itself (np.real of real data, ...)
    w = _viproc.view_of(a, fake, r)
    return _NOPE if w is None else w


def _reshape(a, *args, **kwargs):
    """np.reshape: a view where NumPy makes one, else NumPy's copy (C or F
    order, as requested) as an async copy, reshaped."""
    if type(a) is not ndarray or kwargs.get("copy") is not None:
        return _NOPE
    fake = _viproc.stand_in(a)
    try:
        r = np.reshape(fake, *args, copy=False, **kwargs)
    except ValueError:  # NumPy would copy: copy (async), then view
        order = kwargs.get("order", "C")
        if order not in ("C", "F"):
            return _NOPE
        c = a.astype(a.dtype, order="C") if order == "C" else _NOPE
        if c is _NOPE:
            return _NOPE
        return _reshape(c, *args, **kwargs)
    if r is fake:
        return a
    w = _viproc.view_of(a, fake, r)
    return _NOPE if w is None else w


def _ravel(a, order="C"):
    if order not in ("C",):
        return _NOPE
    return _reshape(a, -1)


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


_VIEW_HANDLERS = {np.reshape: _reshape, np.ravel: _ravel, np.where: _where}


# --- advanced indexing ---------------------------------------------------------
# Integer array indices (and boolean masks, as their nonzero() indices) run
# as gather / scatter tasks: element copies, so any element order gives
# NumPy's values. NumPy's rules decide the result: the basic part of the key
# is applied first as a view; the index arrays broadcast together, and their
# dimensions replace the indexed axes if those are adjacent, else go first.
# NumPy's result layout comes from NumPy itself, indexing a tiny stand-in
# with the same axis order (extents 1 or 2, so it costs nothing).
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


def _gather_fast(a, key):
    """a[idx] for one integer viproc array on C-contiguous data, the common
    case: NumPy's result is in C order (index dimensions, then the remaining
    axes). None if the key is not like that."""
    if key.dtype in _INDEX_DTYPES and a.ndim and a.size and key.size and a._c_contiguous:
        res = [-1 - d for d in range(key.ndim)] + list(range(1, a.ndim))
        return _viproc.gather(a, (key,), (0,), res, None)
    return None


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
    return asarray(np.asarray(value).astype(dtype, copy=False))


def _scatter(a, key, value):
    if isinstance(key, (ndarray, np.ndarray)) and key.dtype == np.bool_ and (
            np.ndim(value) == 0 and key.ndim and key.shape == a.shape[:key.ndim]):
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
        """Basic indexing as a runtime view (metadata only, no sync)."""
        fake = _viproc.stand_in(self)
        keys = key if isinstance(key, tuple) else (key,)
        if not any(k is Ellipsis for k in keys):
            keys = keys + (Ellipsis,)  # always a view, never a NumPy scalar
        return _viproc.view_of(self, fake, fake[keys])

    def __getitem__(self, key):
        r = _viproc.item(self, key)  # a[i, j] of ready data: NumPy's scalar, fast
        if r is not None:
            return r
        if _is_basic_index(key):
            v = self._view(key)
            keys = key if isinstance(key, tuple) else (key,)
            if v.ndim == 0 and not any(k is Ellipsis for k in keys):
                # NumPy returns a scalar: free if the data is there, else a
                # 0-d copy (a snapshot, as NumPy's scalar) instead of a sync
                return _scalar_result(v.copy()) if not _viproc.ready(v) else _viproc.scalar(v)
            return v
        if type(self) is ndarray:
            if type(key) is ndarray:  # a[idx]: the gather's own fast path first
                r = _gather_fast(self, key)
                if r is not None:
                    return r
            if self.size <= _small_ops[0] and _viproc.ready(self) and _ready_key(key):
                # Small and ready: nothing to overlap, NumPy indexes right away
                # (advanced indexing always copies: the result is fresh).
                return _adopt(np.asarray(self.numpy()[_numpy_key(key)]))
            try:
                r = _gather(self, key)
            except (TypeError, ValueError, IndexError):
                r = _NOPE  # NumPy raises its own error
            if r is not _NOPE:
                return r
        return _host_call(lambda a, k: a[k], (self, key), {})

    def __setitem__(self, key, value):
        if not _is_basic_index(key):  # NumPy assigns in place, overlap included
            if not self._readonly and type(self) is ndarray:
                try:
                    done = _scatter(self, key, value)
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
            src = value if value.dtype == self.dtype else value.astype(self.dtype)
        else:
            src = asarray(np.asarray(value).astype(self.dtype, copy=False))
        _viproc.assign(target, src)

    # --- NumPy protocols -------------------------------------------------------
    def __array_function__(self, func, types, args, kwargs):
        if func is np.copyto and len(args) >= 2 and isinstance(args[0], ndarray) and not kwargs:
            args[0][...] = args[1]
            return None
        if func in _VIEW_FUNCS and args:
            r = _meta_view(func, args[0], args[1:], kwargs)
            if r is not _NOPE:
                return r
        handler = _REDUCTIONS.get(func) or _VIEW_HANDLERS.get(func)
        if handler is not None:
            try:
                r = handler(*args, **kwargs)
            except TypeError:  # unexpected arguments: NumPy reports them
                r = _NOPE
            if r is not _NOPE:
                return r
        return _host_call(func, args, kwargs)

    # --- common methods (NumPy semantics via the protocols) -------------------
    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        dtype = np.dtype(dtype)
        if dtype == self.dtype and not copy:
            return self
        code = _CODES.get(dtype)
        if (code is not None and order in ("K", "C") and casting == "unsafe"
                and self.size and type(self) is ndarray):
            r = _viproc.astype(self, code, order)  # NumPy's casts, asynchronously
            if r is not None:
                return r
        return _host_call(lambda a: a.astype(dtype, order, casting, subok, copy), (self,), {})

    def copy(self):
        out = _wrap_numpy(np.empty(self.shape, self.dtype))
        _viproc.assign(out, self)
        return out

    def reshape(self, *shape, **kwargs):
        shape = shape[0] if len(shape) == 1 else shape
        r = _reshape(self, shape, **kwargs) if not kwargs else _NOPE
        return np.reshape(self, shape, **kwargs) if r is _NOPE else r

    def ravel(self, order="C"):
        return np.ravel(self, order)

    def flatten(self, order="C"):
        if order == "C" and type(self) is ndarray:
            return self.astype(self.dtype, order="C").reshape(-1)  # always a copy
        return _host_call(lambda a: a.flatten(order), (self,), {})

    @property
    def T(self):
        r = _meta_view(np.transpose, self)  # no __array_function__ dispatch
        return np.transpose(self) if r is _NOPE else r

    @property
    def mT(self):
        return np.matrix_transpose(self)

    def transpose(self, *axes):
        if len(axes) == 1 and (axes[0] is None or isinstance(axes[0], (tuple, list))):
            axes = axes[0]
        return np.transpose(self, axes if axes != () else None)

    def swapaxes(self, axis1, axis2):
        return np.swapaxes(self, axis1, axis2)

    def squeeze(self, axis=None):
        return np.squeeze(self, axis)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return np.diagonal(self, offset, axis1, axis2)

    @property
    def real(self):
        return np.real(self)

    @property
    def imag(self):
        if self.dtype.kind == "c":
            r = _meta_view(lambda a: a.imag, self)
            if r is not _NOPE:
                return r
        return _host_call(lambda a: a.imag, (self,), {})

    def view(self, dtype=None, type=None):
        if type is None and dtype is not None and not (
                isinstance(dtype, builtins.type) and issubclass(dtype, np.ndarray)):
            r = _meta_view(lambda a: a.view(dtype), self)
            if r is not _NOPE:
                return r
        return _host_call(lambda a: a.view(dtype) if type is None else a.view(dtype, type),
                          (self,), {})

    def sum(self, *a, **k):
        return np.sum(self, *a, **k)

    def prod(self, *a, **k):
        return np.prod(self, *a, **k)

    def min(self, *a, **k):
        return np.min(self, *a, **k)

    def max(self, *a, **k):
        return np.max(self, *a, **k)

    def mean(self, *a, **k):
        return np.mean(self, *a, **k)

    def std(self, *a, **k):
        return np.std(self, *a, **k)

    def var(self, *a, **k):
        return np.var(self, *a, **k)

    def any(self, *a, **k):
        return np.any(self, *a, **k)

    def all(self, *a, **k):
        return np.all(self, *a, **k)

    def argmin(self, *a, **k):
        return np.argmin(self, *a, **k)

    def argmax(self, *a, **k):
        return np.argmax(self, *a, **k)


def _ufunc_reduce(ufunc, a, axis=0, dtype=None, out=None, keepdims=False, initial=_NV,
                  where=True):
    if out is not None or where is not True:
        return None
    return _reduce(ufunc, a, axis, keepdims, dtype, initial)


def _fallback_ufunc(ufunc, method, inputs, kwargs):
    """ufunc.reduce asynchronously where possible; anything else syncs and
    runs eagerly in NumPy (in place for out= and ufunc.at)."""
    if method == "reduce" and len(inputs) == 1:
        try:
            r = _ufunc_reduce(ufunc, *inputs, **kwargs)
        except TypeError:
            r = None
        if r is not None:
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
_viproc.setup(
    ndarray,
    tuple(_DTYPES[c] for c in range(7)),
    _fallback_ufunc,
    asarray,
    (_PKG_DIR, _NUMPY_DIR),
    _OPS,
    np.ndarray,
    np.generic,
)


def wait_all():
    """Sync point: wait until all issued ops have completed."""
    if _viproc.initialized():
        _viproc.wait_all()


def init(workers=0, max_active_tasks=0, sync_below=None, adaptive=None):
    """(Re)starts the runtime.

    workers: worker threads (0: one per CPU).
    sync_below: ops whose inputs are ready and that produce fewer elements run
        inline on the calling thread (default 2048, env VIPROC_SYNC_BELOW).
    adaptive: let each issue site switch to inline execution while its results
        are usually needed right after issuing (default True; False, or env
        VIPROC_DETERMINISTIC=1, gives a deterministic schedule). Results are
        identical either way; only where ops run changes.
    """
    env_sync, env_adaptive = _env_policy()
    shutdown()
    sync_below = env_sync if sync_below is None else sync_below
    _viproc.init(workers, max_active_tasks, sync_below,
                 env_adaptive if adaptive is None else adaptive)
    _small_ops[0] = 2048 if sync_below < 0 else sync_below


def offload_stats():
    """{'inline': n, 'offloaded': m, 'direct': d}: ready ops run inline as a task /
    handed to workers / run directly without a task (small, inputs ready)."""
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
    if _viproc.initialized():
        _viproc.shutdown()


def workers():
    _ensure_runtime()
    return _viproc.workers()
