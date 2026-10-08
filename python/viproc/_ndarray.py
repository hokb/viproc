"""The viproc array type and its NumPy integration.

viproc arrays intercept NumPy through the array protocols (NEP 13
``__array_ufunc__``, NEP 18 ``__array_function__``). Supported ops are issued
asynchronously to the runtime; everything else syncs, runs eagerly in NumPy and
wraps the result again (fallback).
"""

import os
import sys

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.lib.stride_tricks import as_strided

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
_PY_SCALARS = (bool, int, float, complex)


def _site():
    """Source location of the user code issuing the current op."""
    f = sys._getframe(1)
    while f is not None:
        fn = f.f_code.co_filename
        if not (fn.startswith(_PKG_DIR) or fn.startswith(_NUMPY_DIR)):
            return f"{fn}:{f.f_lineno} in {f.f_code.co_name}"
        f = f.f_back
    return "<unknown>"


def _ensure_runtime():
    if not _viproc.initialized():
        workers = int(os.environ.get("VIPROC_WORKERS", "0"))
        _viproc.init(workers)


def _wrap_numpy(arr):
    """viproc array over a NumPy array's memory. The caller gives up `arr`."""
    _ensure_runtime()
    if arr.dtype not in _CODES or not arr.flags.aligned or not arr.dtype.isnative:
        return None
    return ndarray._from_handle(_viproc.from_ndarray(arr))


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
    return np.asarray(x) if isinstance(x, ndarray) else x


def _tree(x, fn):
    if isinstance(x, (list, tuple)):
        return type(x)(_tree(i, fn) for i in x)
    if isinstance(x, dict):
        return {k: _tree(v, fn) for k, v in x.items()}
    return fn(x)


def _wrap_result(x):
    """Fallback results: NumPy arrays become viproc arrays again."""
    if isinstance(x, np.ndarray) and not isinstance(x, ndarray):
        if x.ndim == 0:
            return x[()]
        # Fresh result of a fallback: nobody else holds it, wrap without copy.
        w = _wrap_numpy(x if x.flags.writeable else x.copy())
        return w if w is not None else x
    if isinstance(x, (list, tuple)):
        return type(x)(_wrap_result(i) for i in x)
    return x


def _is_basic_index(key):
    key = key if isinstance(key, tuple) else (key,)
    for k in key:
        if k is None or k is Ellipsis or isinstance(k, slice):
            continue
        if isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)):
            continue
        return False
    return True


# Functions that modify an argument in place: the fallback would modify a copy.
_MUTATING = {np.copyto, np.put, np.place, np.putmask, np.fill_diagonal}


class ndarray(NDArrayOperatorsMixin):
    """Array whose ops run asynchronously in the viproc runtime.

    Supported ops return immediately with a pending result; reading data
    (printing, ``np.asarray``, ``float()``, unsupported functions) is a sync
    point.
    """

    __slots__ = ("_h", "_meta")
    __array_priority__ = 100

    @classmethod
    def _from_handle(cls, h):
        self = object.__new__(cls)
        self._h = h
        shape, strides, code = _viproc.info(h)
        self._meta = (shape, strides, _DTYPES[code])
        return self

    # --- metadata (no sync) -------------------------------------------------
    @property
    def shape(self):
        return self._meta[0]

    @property
    def strides(self):
        return self._meta[1]

    @property
    def dtype(self):
        return self._meta[2]

    @property
    def ndim(self):
        return len(self._meta[0])

    @property
    def size(self):
        return int(np.prod(self._meta[0], dtype=np.int64))

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
        """Sync point: a NumPy copy of the data."""
        return _viproc.to_ndarray(self._h)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("viproc: a NumPy array of a viproc array is always a copy")
        arr = self.numpy()
        return arr if dtype is None else arr.astype(dtype, copy=False)

    def wait(self):
        """Sync point: wait until the data is computed (raises AsyncError)."""
        _viproc.wait(self._h)
        return self

    def ready(self):
        return _viproc.ready(self._h)

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

    def __index__(self):
        return self.numpy().__index__()

    def __iter__(self):
        return iter(_wrap_result(self.numpy()) if self.ndim > 1 else self.numpy())

    def item(self, *args):
        return self.numpy().item(*args)

    def tolist(self):
        return self.numpy().tolist()

    # --- indexing ------------------------------------------------------------
    def _view(self, key):
        """Basic indexing as a runtime view (metadata only, no sync)."""
        dummy = np.empty(1, dtype=self.dtype)
        fake = as_strided(dummy, shape=self.shape, strides=self.strides, writeable=False)
        keys = key if isinstance(key, tuple) else (key,)
        if not any(k is Ellipsis for k in keys):
            keys = keys + (Ellipsis,)  # always a view, never a NumPy scalar
        v = fake[keys]
        offset = v.__array_interface__["data"][0] - dummy.__array_interface__["data"][0]
        return ndarray._from_handle(_viproc.view(self._h, v.shape, v.strides, offset))

    def __getitem__(self, key):
        if _is_basic_index(key):
            v = self._view(key)
            keys = key if isinstance(key, tuple) else (key,)
            if v.ndim == 0 and not any(k is Ellipsis for k in keys):
                return v.numpy()[()]  # NumPy returns a scalar here: sync point
            return v
        return _wrap_result(self.numpy()[_tree(key, _to_numpy)])

    def __setitem__(self, key, value):
        if _is_basic_index(key):
            target = self._view(key)
        else:
            arr = self.numpy()
            arr[_tree(key, _to_numpy)] = _to_numpy(value)
            target, value = self._view(Ellipsis), arr
        src = value if isinstance(value, ndarray) else None
        if src is None or src.dtype != self.dtype:
            src = asarray(np.asarray(_to_numpy(value)).astype(self.dtype, copy=False))
        _viproc.assign(target._h, src._h, _site())

    # --- NumPy protocols -------------------------------------------------------
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if (method == "__call__" and ufunc.nout == 1 and ufunc.signature is None
                and set(kwargs) <= {"out"}):
            result = _issue_ufunc(ufunc, inputs, kwargs.get("out"))
            if result is not NotImplemented:
                return result
        return _fallback_ufunc(ufunc, method, inputs, kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func is np.copyto and len(args) >= 2 and isinstance(args[0], ndarray) and not kwargs:
            args[0][...] = args[1]
            return None
        if func in _MUTATING:
            raise NotImplementedError(f"viproc: in-place function {func.__name__} is not supported")
        result = func(*_tree(args, _to_numpy), **_tree(kwargs, _to_numpy))
        return _wrap_result(result)

    # --- common methods (NumPy semantics via the protocols) -------------------
    def astype(self, dtype, copy=True):
        dtype = np.dtype(dtype)
        if dtype == self.dtype and not copy:
            return self
        return asarray(self.numpy().astype(dtype))

    def copy(self):
        out = _wrap_numpy(np.empty(self.shape, self.dtype))
        _viproc.assign(out._h, self._h, _site())
        return out

    def reshape(self, *shape, **kwargs):
        return np.reshape(self, shape[0] if len(shape) == 1 else shape, **kwargs)

    @property
    def T(self):
        return np.transpose(self)

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


def _issue_ufunc(ufunc, inputs, out):
    """Issues ufunc(*inputs, out=out) asynchronously, or NotImplemented."""
    if out is not None:
        if len(out) != 1 or not isinstance(out[0], ndarray):
            return NotImplemented
        out = out[0]
    spec = []
    for x in inputs:
        if isinstance(x, ndarray):
            spec.append(x.dtype)
        elif isinstance(x, _PY_SCALARS):
            spec.append(type(x))  # weak Python scalar (NEP 50)
        elif isinstance(x, (np.ndarray, np.generic)):
            spec.append(x.dtype)
        else:
            return NotImplemented
    try:
        dtypes = ufunc.resolve_dtypes(tuple(spec) + (None,))
    except (TypeError, ValueError, np.exceptions.DTypePromotionError):
        return NotImplemented
    in_dtypes, out_dtype = dtypes[: ufunc.nin], dtypes[ufunc.nin]
    if out_dtype not in _CODES:
        return NotImplemented

    handles = []
    shapes = []
    for x, dt in zip(inputs, in_dtypes):
        if isinstance(x, ndarray):
            if x.dtype != dt:
                return NotImplemented  # needs casting
            v = x
        elif isinstance(x, _PY_SCALARS):
            v = asarray(np.asarray(x, dtype=dt))  # may raise like NumPy (overflow)
        else:
            if x.dtype != dt or dt not in _CODES:
                return NotImplemented
            v = asarray(x)
        handles.append(v._h)
        shapes.append(v.shape)
    try:
        shape = np.broadcast_shapes(*shapes)
    except ValueError:
        return NotImplemented  # the fallback raises NumPy's own error
    if out is not None and (out.shape != shape or out.dtype != out_dtype):
        return NotImplemented
    try:
        h = _viproc.ufunc(ufunc, tuple(handles), _CODES[out_dtype], shape,
                          None if out is None else out._h, _site())
    except NotImplementedError:
        return NotImplemented
    return out if out is not None else ndarray._from_handle(h)


def _fallback_ufunc(ufunc, method, inputs, kwargs):
    """Sync, run eagerly in NumPy, wrap the result."""
    out = kwargs.pop("out", None)
    args = _tree(inputs, _to_numpy)
    kwargs = _tree(kwargs, _to_numpy)
    if out is None:
        return _wrap_result(getattr(ufunc, method)(*args, **kwargs))
    out = out if isinstance(out, tuple) else (out,)
    np_out = tuple(_to_numpy(o) for o in out)
    getattr(ufunc, method)(*args, out=np_out if len(np_out) > 1 else np_out[0], **kwargs)
    for o, r in zip(out, np_out):
        if isinstance(o, ndarray):
            o[...] = r
    return out[0] if len(out) == 1 else out


def wait_all():
    """Sync point: wait until all issued ops have completed."""
    if _viproc.initialized():
        _viproc.wait_all()


def init(workers=0, max_active_tasks=0):
    """(Re)starts the runtime with `workers` threads (0: one per CPU)."""
    shutdown()
    _viproc.init(workers, max_active_tasks)


def shutdown():
    if _viproc.initialized():
        _viproc.shutdown()


def workers():
    _ensure_runtime()
    return _viproc.workers()
