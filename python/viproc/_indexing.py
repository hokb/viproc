"""Indexing functions on the gather / scatter / select tasks (hokb/viproc#25,
step 4): take, take_along_axis, put, put_along_axis, compress, extract,
place, select, copyto(where=).

The code follows NumPy's (item_selection.c: PyArray_TakeFrom, PyArray_PutTo,
PyArray_Compress; lib/_shape_base_impl.py, lib/_function_base_impl.py,
multiarraymodule.c: copyto; BSD-3-Clause) without the asanyarray() calls,
which would sync a viproc argument. Where NumPy needs the data to know the
result's shape (compress, extract, place: the positions of a condition's
True values), the handler syncs on the condition only, as boolean indexing
does. A handler returns _NOPE for arguments it does not cover: NumPy then
runs the call in the eager fallback, with its own errors.

Index errors of indices viproc does not see on the main thread (a pending
index array) surface asynchronously as AsyncIndexError (an IndexError), as
for advanced indexing (error-handling.md).
"""

import numpy as np
from numpy.lib.array_utils import normalize_axis_index

from . import _ndarray as _nd
from . import _viproc

_NOPE = _nd._NOPE
ndarray = _nd.ndarray
_INDEX_DTYPES = _nd._INDEX_DTYPES


def _vp(x):
    return type(x) is ndarray


def _c_strides(shape, itemsize):
    strides = [0] * len(shape)
    s = itemsize
    for k in range(len(shape) - 1, -1, -1):
        strides[k] = s
        s *= shape[k]
    return strides


_MODES = {"raise": 0, "wrap": 1, "clip": 2}  # VP_INDEX_RAISE, _WRAP, _CLIP


def _indices(indices):
    """The index array as an int32/int64 viproc array; None if not covered.
    NumPy's mode is the gather's / scatter's (an index out of bounds in mode
    "raise" fails the task: AsyncIndexError, no check pass)."""
    if isinstance(indices, ndarray):
        return indices if indices.dtype in _INDEX_DTYPES else None
    x = np.asarray(indices)
    if x.dtype.kind not in "iu" or x.dtype == np.uint64:
        return None
    # NumPy's take/put convert to intp: that copy is fresh memory, adopted
    # without another copy
    return _nd._adopt(np.array(x, dtype=np.intp, copy=True))


def _take(a, indices, axis=None, out=None, mode="raise"):
    if not _vp(a) or out is not None or mode not in _MODES:
        return _NOPE
    if axis is None:
        a = a.ravel()  # PyArray_CheckAxis: C order
        axis = 0
    elif a.ndim == 0:
        return _NOPE
    else:
        axis = normalize_axis_index(axis, a.ndim)
    n = a.shape[axis]
    if n == 0 or a.size == 0:
        return _NOPE
    idx = _indices(indices)
    if idx is None or idx.size == 0:
        return _NOPE
    if idx.ndim == 0:  # one index: NumPy's take of it (a scalar for a 1-d a)
        r = _take(a, idx.reshape(1), axis, mode=mode)
        return r[(slice(None),) * axis + (0,)]
    # NumPy's result: a new C-ordered array of a.shape[:axis] + idx.shape +
    # a.shape[axis + 1:]
    shape = a.shape[:axis] + idx.shape + a.shape[axis + 1:]
    res = list(range(axis)) + [-1 - d for d in range(idx.ndim)] + list(range(axis + 1, a.ndim))
    return _viproc.gather(a, (idx,), (axis,), res, _c_strides(shape, a.itemsize), _MODES[mode])


def _flat_target(a):
    """a as a writable 1-d view in C order, or None (not C-contiguous)."""
    if a._readonly or not a._c_contiguous:
        return None
    return a.reshape(-1)


def _put(a, ind, v, mode="raise"):
    if not _vp(a) or mode not in _MODES or a.size == 0:
        return _NOPE
    flat = _flat_target(a)
    if flat is None:
        return _NOPE  # NumPy writes a C-ordered copy back (not yet)
    idx = _indices(ind)
    if idx is None:
        return _NOPE
    idx = idx.reshape(-1)
    ni = idx.size
    if ni == 0:
        return None
    vals = _nd._as_values(v, a.dtype)  # NumPy: a forced (unsafe) cast
    nv = vals.size
    if nv == 0:
        return _NOPE
    vals = vals.reshape(-1)
    if nv != ni and nv != 1:  # values repeat: dest[idx[i]] = values[i % nv]
        vals = vals[_nd.asarray(np.arange(ni, dtype=np.intp) % nv)]
    _viproc.scatter(flat, (idx,), (0,), (-1,), vals, _MODES[mode])
    return None


def _along_axis(arr, indices, axis):
    from numpy.lib._shape_base_impl import _make_along_axis_idx

    axis = normalize_axis_index(axis, arr.ndim)
    return _make_along_axis_idx(arr.shape, indices, axis)


def _take_along_axis(arr, indices, axis=-1):
    if not _vp(arr):
        return _NOPE
    if axis is None:
        if indices.ndim != 1:
            return _NOPE  # NumPy's ValueError
        arr = arr.ravel()  # np.array(arr.flat): C order
        axis = 0
    return arr[_along_axis(arr, indices, axis)]


def _put_along_axis(arr, indices, values, axis):
    if not _vp(arr) or axis is None:
        return _NOPE  # axis=None: NumPy writes into a copy (no effect)
    arr[_along_axis(arr, indices, axis)] = values
    return None


def _condition_indices(condition):
    """The positions of the True values of a 1-d condition (a sync on a
    viproc condition: they decide the result's shape)."""
    c = condition.numpy() if isinstance(condition, ndarray) else np.asarray(condition)
    if c.ndim != 1:
        return None
    return np.flatnonzero(c)


def _compress(condition, a, axis=None, out=None):
    if not _vp(a) or out is not None:
        return _NOPE
    idx = _condition_indices(condition)
    if idx is None or idx.size == 0:
        return _NOPE
    return _take(a, idx, axis)


def _extract(condition, arr):
    if not _vp(arr):
        return _NOPE
    c = condition.ravel() if isinstance(condition, ndarray) else np.ravel(condition)
    idx = _condition_indices(c)
    if idx is None or idx.size == 0:
        return _NOPE
    return _take(arr.ravel(), idx)


def _place(arr, mask, vals):
    if not _vp(arr) or (vals.size if _vp(vals) else np.size(vals)) == 0:
        return _NOPE  # NumPy's errors
    m = mask.ravel() if isinstance(mask, ndarray) else np.ravel(mask)
    if m.size != arr.size:
        return _NOPE
    idx = _condition_indices(m)
    if idx is None:
        return _NOPE
    if idx.size == 0:
        return None
    return _put(arr, idx, vals.ravel() if _vp(vals) else np.ravel(vals))


def _copyto(dst, src, casting="same_kind", where=True):
    if not _vp(dst) or dst._readonly:
        return _NOPE
    src_dtype = src.dtype if isinstance(src, (ndarray, np.ndarray)) else np.result_type(src)
    if not np.can_cast(src_dtype, dst.dtype, casting):
        return _NOPE  # NumPy's TypeError
    if where is True:
        dst[...] = src
        return None
    mask = where if isinstance(where, ndarray) else np.asarray(where)
    if mask.dtype != np.bool_:
        return _NOPE
    mask = mask if isinstance(mask, ndarray) else _nd.asarray(mask)
    values = _nd._as_values(src, dst.dtype)
    try:
        if np.broadcast_shapes(mask.shape, values.shape, dst.shape) != dst.shape:
            return _NOPE
    except ValueError:
        return _NOPE
    # dst[mask] = src as a select of the new and the old values: element
    # copies, so the values are NumPy's
    dst[...] = np.where(mask, values, dst)
    return None


def _select(condlist, choicelist, default=0):
    if len(condlist) != len(choicelist) or not condlist:
        return _NOPE  # NumPy's ValueError
    choices = [c if type(c) in (int, float, complex) or isinstance(c, ndarray) else np.asarray(c)
               for c in choicelist]
    dflt = default if type(default) in (int, float, complex) or isinstance(default, ndarray) \
        else np.asarray(default)
    try:
        # dtypes for arrays (np.result_type of a viproc array would export it)
        dtype = np.result_type(*(x.dtype if isinstance(x, ndarray) else x
                                 for x in choices + [dflt]))
    except TypeError:
        return _NOPE
    if dtype not in _nd._CODES:
        return _NOPE
    conds = [c if isinstance(c, ndarray) else np.asarray(c) for c in condlist]
    if any(c.dtype != np.bool_ for c in conds):
        return _NOPE  # NumPy's TypeError
    try:
        shape = np.broadcast_shapes(*(x.shape if isinstance(x, ndarray) else np.shape(x)
                                      for x in conds + choices + [dflt]))
    except ValueError:
        return _NOPE
    if 0 in shape:
        return _NOPE
    # result = full(shape, default); copyto(result, choice, where=cond) from
    # the last choice to the first: the same as nested selects, with the
    # first matching condition winning
    r = _nd._as_values(dflt, dtype)
    for c, ch in zip(conds[::-1], choices[::-1]):
        r = np.where(c if isinstance(c, ndarray) else _nd.asarray(c), _nd._as_values(ch, dtype), r)
    if r.shape != shape:
        r = np.broadcast_to(r, shape)
    if r.ndim == 0 or not r._c_contiguous:
        r = r.astype(dtype, order="C")  # NumPy's result (np.full) is C-ordered
    return r


INDEXING = {
    np.take: _take, np.put: _put, np.take_along_axis: _take_along_axis,
    np.put_along_axis: _put_along_axis, np.compress: _compress, np.extract: _extract,
    np.place: _place, np.select: _select, np.copyto: _copyto,
}
