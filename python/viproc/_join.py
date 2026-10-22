"""Joining and copying functions as one placement task each (hokb/viproc#25,
step 3): concatenate, stack, hstack, vstack, dstack, column_stack, append,
tile, repeat, pad (constant mode), roll, delete, insert.

NumPy builds these results by assigning its inputs into views of a new array
(``new[slices] = arr``). viproc records the same assignments as parts of one
new array (``_Placer``) and issues them as a single task (``vp_place``):
element copies, so the values are NumPy's; the result's layout is the one
NumPy allocates. The code follows NumPy's (numpy/_core/shape_base.py,
multiarraymodule.c, item_selection.c, lib/_function_base_impl.py,
lib/_shape_base_impl.py, lib/_arraypad_impl.py, numeric.py; BSD-3-Clause)
without the asanyarray() calls, which would sync a viproc argument. A handler
returns _NOPE for arguments it does not cover: NumPy then runs the call in
the eager fallback, with its own errors.
"""

import itertools
import operator

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


def _contiguous(a, axes):
    """NumPy's contiguity flag for the axes from the innermost (extent-1 axes
    may have any stride; empty arrays are contiguous)."""
    if 0 in a.shape:
        return True
    s = a.dtype.itemsize
    for ax in axes:
        if a.shape[ax] != 1:
            if a.strides[ax] != s:
                return False
            s *= a.shape[ax]
    return True


def _fnc(a):  # a.flags.fnc: F-contiguous and not C-contiguous
    return _contiguous(a, range(a.ndim)) and not _contiguous(a, range(a.ndim - 1, -1, -1))


def _stride_perm(arrays, ndim):
    """PyArray_CreateMultiSortedStridePerm (shape.c): the axes from the
    biggest to the smallest stride over all arrays, ambiguities resolved as
    NumPy's iterator resolves them (C order wins)."""
    perm = list(range(ndim))
    for i0 in range(1, ndim):
        ipos = i0
        ax0 = perm[i0]
        for i1 in range(i0 - 1, -1, -1):
            ambig, swap = True, False
            ax1 = perm[i1]
            for a in arrays:
                if a.shape[ax0] != 1 and a.shape[ax1] != 1:
                    if abs(a.strides[ax0]) <= abs(a.strides[ax1]):
                        swap = False
                    elif ambig:
                        swap = True
                    ambig = False
            if not ambig:
                if swap:
                    ipos = i1
                else:
                    break
        if ipos != i0:
            perm[ipos + 1:i0 + 1] = perm[ipos:i0]
            perm[ipos] = ax0
    return perm


def _dense_strides(shape, itemsize, perm):
    """Dense strides with the axes in `perm` order (outermost first)."""
    strides = [0] * len(shape)
    s = itemsize
    for ax in reversed(perm):
        strides[ax] = s
        s *= shape[ax]
    return tuple(strides)


def _order_strides(shape, itemsize, order):
    perm = range(len(shape)) if order == "C" else range(len(shape) - 1, -1, -1)
    return _dense_strides(shape, itemsize, perm)


def _as_dtype(x, dtype):
    """x as a viproc array of `dtype`, cast as NumPy's assignment casts
    (unsafe): a viproc array through an async cast, anything else on the
    host (NumPy's conversion and errors), then imported."""
    if isinstance(x, ndarray):
        return x if x.dtype == dtype else x.astype(dtype)
    return _nd._adopt(np.array(x, dtype=dtype))


class _Placer:
    """A new array of `dtype` and `shape` with dense byte `strides`,
    assembled as NumPy assembles it: ``p[key] = value`` records the
    assignment into the view ``new[key]`` (keys of slices and None only),
    result() issues all of them as one task (vp_place)."""

    def __init__(self, dtype, shape, strides):
        self.dtype = dtype
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.parts = []

    def add(self, value, shape, strides, offset):
        """The part (shape, byte strides, byte offset) gets `value` (this
        dtype, shaped or broadcastable to `shape`)."""
        if 0 not in shape:
            self.parts.append((value, shape, strides, offset))

    def __setitem__(self, key, value):
        # the view new[key], as NumPy's basic indexing computes it
        shape, strides, offset, d = [], [], 0, 0
        for k in key:
            if k is None:
                shape.append(1)
                strides.append(0)
                continue
            start, stop, step = k.indices(self.shape[d])
            shape.append(len(range(start, stop, step)))
            strides.append(step * self.strides[d])
            offset += start * self.strides[d]
            d += 1
        shape = tuple(shape) + self.shape[d:]
        if 0 in shape:
            return
        value = _as_dtype(value, self.dtype)
        if value.shape != shape and value.ndim:
            extra = value.ndim - len(shape)
            if extra > 0 and all(n == 1 for n in value.shape[:extra]):
                value = value.reshape(value.shape[extra:])  # NumPy drops leading 1s
            if value.ndim > len(shape) or any(
                    n != 1 and n != m for n, m in zip(value.shape[::-1], shape[::-1])):
                _assign_error(value, shape)
        self.parts.append((value, shape, tuple(strides) + self.strides[d:], offset))

    def result(self):
        if 0 in self.shape:
            return _NOPE  # empty: NumPy's own strides, no data to wait for
        return _viproc.place(self.dtype, self.shape, self.strides, self.parts)


def _assign_error(value, shape):
    """Raises NumPy's error for new[...] = value into a part of `shape`, on
    zero-stride probes (no element memory)."""
    dst = np.lib.stride_tricks.as_strided(np.empty(1, value.dtype), shape, (0,) * len(shape))
    dst[...] = np.broadcast_to(np.empty(1, value.dtype), value.shape)
    raise AssertionError("NumPy accepted the assignment")  # pragma: no cover


# --- concatenate and stacking (shape_base.py, multiarraymodule.c) -------------------


def _concatenate(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    if out is not None:
        return _NOPE
    if isinstance(arrays, ndarray):  # iterated over its first axis
        if arrays.ndim == 0:
            return _NOPE
        arrays = [arrays[i] for i in range(arrays.shape[0])]
    arrays = [_arr(x) for x in arrays]
    if not arrays or not any(_vp(x) for x in arrays):
        return _NOPE
    dt = arrays[0].dtype
    if dtype is not None or any(x.dtype != dt for x in arrays):
        dt = np.dtype(dtype) if dtype is not None else np.result_type(*(x.dtype for x in arrays))
        if dt not in _nd._CODES or not all(np.can_cast(x.dtype, dt, casting) for x in arrays):
            return _NOPE
        arrays_dt = [_as_dtype(x, dt) for x in arrays]
    else:
        arrays_dt = [_as_dtype(x, dt) for x in arrays] if not all(_vp(x) for x in arrays) else arrays
    item = dt.itemsize
    if axis is None:  # PyArray_ConcatenateFlattenedArrays: each in C order
        p = _Placer(dt, (sum(x.size for x in arrays),), (item,))
        at = 0
        for x in arrays_dt:  # the next x.size elements of the result, C-ordered as x
            p.add(x, x.shape, _order_strides(x.shape, item, "C"), at * item)
            at += x.size
        return p.result()
    ndim = arrays[0].ndim
    if ndim == 0 or any(x.ndim != ndim for x in arrays):
        return _NOPE
    axis = normalize_axis_index(axis, ndim)
    shape = list(arrays[0].shape)
    for x in arrays[1:]:
        if any(x.shape[d] != shape[d] for d in range(ndim) if d != axis):
            return _NOPE
        shape[axis] += x.shape[axis]
    perm = _stride_perm(arrays, ndim) if ndim > 1 else (0,)
    p = _Placer(dt, shape, _dense_strides(shape, item, perm))
    step = p.strides[axis]
    at = 0
    for x in arrays_dt:  # new[..., at:at + n, ...] = x
        p.add(x, x.shape, p.strides, at * step)
        at += x.shape[axis]
    return p.result()


def _stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    arrays = [_arr(x) for x in arrays]
    if not arrays or len({x.shape for x in arrays}) != 1:
        return _NOPE
    axis = normalize_axis_index(axis, arrays[0].ndim + 1)
    sl = (slice(None),) * axis + (np.newaxis,)
    return _concatenate([x[sl] for x in arrays], axis=axis, out=out, dtype=dtype,
                        casting=casting)


def _atleast(nd):
    def conv(x):
        x = _arr(x)
        if x.ndim >= nd:
            return x
        if nd == 3:  # atleast_3d: (N,) -> (1, N, 1), (M, N) -> (M, N, 1)
            return x.reshape((1, x.size, 1) if x.ndim < 2 else x.shape + (1,))
        return x.reshape((1,) * (nd - x.ndim) + x.shape)

    return conv


def _stacker(nd, axis_of):
    def handler(tup, *, dtype=None, casting="same_kind"):
        arrs = [_atleast(nd)(x) for x in tup]
        if not arrs:
            return _NOPE
        return _concatenate(arrs, axis_of(arrs), dtype=dtype, casting=casting)

    return handler


def _column_stack(tup):
    arrays = []
    for v in tup:
        arr = _arr(v)
        if arr.ndim < 2:
            arr = arr.reshape(1, -1).T if arr.ndim == 1 else arr.reshape(1, 1)
        arrays.append(arr)
    return _concatenate(arrays, 1)


def _append(arr, values, axis=None):
    arr = _arr(arr)
    if axis is None:
        if arr.ndim != 1:
            arr = arr.ravel()
        values = _arr(values).ravel()
        axis = arr.ndim - 1
    return _concatenate((arr, values), axis=axis)


# --- tile, repeat (lib/_shape_base_impl.py, item_selection.c) -------------------


def _tile(A, reps):
    if not _vp(A):
        return _NOPE
    try:
        tup = tuple(reps)
    except TypeError:
        tup = (reps,)
    tup = tuple(operator.index(t) for t in tup)
    d = len(tup)
    if all(t == 1 for t in tup):  # NumPy copies (A is an ndarray there)
        return _nd.array(A, copy=True, ndmin=d)  # order K
    c = A.reshape((1,) * (d - A.ndim) + A.shape) if d > A.ndim else A
    tup = (1,) * (c.ndim - d) + tup
    if c.size == 0 or any(t < 0 for t in tup):
        return _NOPE
    shape_out = tuple(s * t for s, t in zip(c.shape, tup))
    if 2 * c.ndim > 64:
        return _NOPE
    # The C-ordered result as (t0, s0, t1, s1, ...): every tile a copy of c.
    item = c.dtype.itemsize
    strides = _order_strides(shape_out, item, "C")
    p = _Placer(c.dtype, shape_out, strides)
    shape, pst, key = [], [], []
    for s, t, st in zip(c.shape, tup, strides):
        shape += [t, s]
        pst += [s * st, st]
        key += [np.newaxis, slice(None)]
    p.add(c[tuple(key)], tuple(shape), tuple(pst), 0)
    return p.result()


def _repeat(a, repeats, axis=None):
    if not _vp(a):
        return _NOPE
    r = repeats
    if not isinstance(r, (int, np.integer)) or isinstance(r, (bool, np.bool_)):
        if isinstance(r, ndarray):
            return _NOPE
        r = np.asarray(r)
        if r.size != 1 or r.dtype.kind not in "iu":
            return _NOPE  # repeats per element: a gather (not yet)
        r = r.item()
    r = int(r)
    if axis is None:
        a = a.reshape(a.shape + (1,)) if a.ndim else a.reshape(1, 1)
        axis = a.ndim - 1
        out_shape = (a.size * r,)
    else:
        if a.ndim == 0:
            return _NOPE
        axis = normalize_axis_index(axis, a.ndim)
        out_shape = a.shape[:axis] + (a.shape[axis] * r,) + a.shape[axis + 1:]
        a = a[(slice(None),) * (axis + 1) + (np.newaxis,)]
        axis += 1
    if r <= 0 or a.size == 0:
        return _NOPE
    item = a.dtype.itemsize
    p = _Placer(a.dtype, out_shape, _order_strides(out_shape, item, "C"))
    shape = a.shape[:axis] + (r,) + a.shape[axis + 1:]
    p.add(a, shape, _order_strides(shape, item, "C"), 0)
    return p.result()


# --- pad (lib/_arraypad_impl.py), constant mode ---------------------------------------


def _pad(array, pad_width, mode="constant", **kwargs):
    from numpy.lib._arraypad_impl import _as_pairs

    if not _vp(array) or mode != "constant" or set(kwargs) - {"constant_values"}:
        return _NOPE
    if isinstance(pad_width, dict) or isinstance(pad_width, ndarray):
        return _NOPE
    pad_width = np.asarray(pad_width)
    if pad_width.dtype.kind != "i":
        return _NOPE
    pad_width = [(int(left), int(right))
                 for left, right in _as_pairs(pad_width, array.ndim, as_index=True)]
    new_shape = tuple(left + size + right for size, (left, right) in zip(array.shape, pad_width))
    order = "F" if _fnc(array) else "C"
    p = _Placer(array.dtype, new_shape, _order_strides(new_shape, array.dtype.itemsize, order))
    original = tuple(slice(left, left + size) for size, (left, right) in zip(array.shape, pad_width))
    p[original] = array
    values = kwargs.get("constant_values", 0)
    if isinstance(values, ndarray):
        return _NOPE
    values = _as_pairs(values, array.ndim)
    for axis, (left, right), (vl, vr) in zip(range(array.ndim), pad_width, values):
        # _set_pad_area on _view_roi: all of the axes up to `axis`, the
        # original area of the later ones
        n = new_shape[axis]
        rest = original[axis + 1:]
        p[(slice(None),) * axis + (slice(None, left),) + rest] = _value(vl, array.dtype)
        p[(slice(None),) * axis + (slice(n - right, None),) + rest] = _value(vr, array.dtype)
    return p.result()


def _value(v, dtype):
    """A pad value as NumPy's ``padded[sl] = v`` converts it."""
    t = np.empty((), dtype)
    t[()] = v
    return _nd._adopt(t)


# --- roll (numeric.py) ----------------------------------------------------------------


def _roll(a, shift, axis=None):
    if not _vp(a):
        return _NOPE
    if axis is None:
        r = _roll(a.ravel(), shift, 0)
        return r if r is _NOPE else r.reshape(a.shape)
    axis = normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
    if isinstance(shift, ndarray):
        return _NOPE
    broadcasted = np.broadcast(shift, axis)
    if broadcasted.ndim > 1 or a.size == 0:
        return _NOPE
    shifts = dict.fromkeys(range(a.ndim), 0)
    for sh, ax in broadcasted:
        shifts[ax] += int(sh)
    rolls = [((slice(None), slice(None)),)] * a.ndim
    for ax, offset in shifts.items():
        offset %= a.shape[ax] or 1
        if offset:
            rolls[ax] = ((slice(None, -offset), slice(offset, None)),
                         (slice(-offset, None), slice(None, offset)))
    # empty_like(a): NumPy's order K
    p = _Placer(a.dtype, a.shape,
                _dense_strides(a.shape, a.dtype.itemsize, _stride_perm([a], a.ndim)))
    for indices in itertools.product(*rolls):
        arr_index, res_index = zip(*indices)
        p[res_index] = a[arr_index]
    return p.result()


# --- delete, insert (lib/_function_base_impl.py) ---------------------------------------


def _prepare(arr, axis):
    ndim = arr.ndim
    order = "F" if _fnc(arr) else "C"
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()
        axis = arr.ndim - 1
    else:
        axis = normalize_axis_index(axis, ndim)
    return arr, axis, order


def _delete(arr, obj, axis=None):
    if not _vp(arr) or isinstance(obj, ndarray) or arr.ndim == 0:
        return _NOPE
    arr, axis, order = _prepare(arr, axis)
    ndim = arr.ndim
    slobj = [slice(None)] * ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)

    def new():
        return _Placer(arr.dtype, newshape, _order_strides(newshape, arr.dtype.itemsize, order))

    if isinstance(obj, slice):
        start, stop, step = obj.indices(N)
        xr = range(start, stop, step)
        numtodel = len(xr)
        if numtodel <= 0:
            return arr.copy(order=order)
        if step < 0:
            step = -step
            start = xr[-1]
            stop = xr[0] + 1
        newshape[axis] -= numtodel
        p = new()
        if start != 0:
            slobj[axis] = slice(None, start)
            p[tuple(slobj)] = arr[tuple(slobj)]
        if stop != N:
            slobj[axis] = slice(stop - numtodel, None)
            slobj2 = [slice(None)] * ndim
            slobj2[axis] = slice(stop, None)
            p[tuple(slobj)] = arr[tuple(slobj2)]
        if step != 1:
            keep = np.ones(stop - start, dtype=bool)
            keep[:stop - start:step] = False
            slobj[axis] = slice(start, stop - numtodel)
            slobj2 = [slice(None)] * ndim
            slobj2[axis] = slice(start, stop)
            sub = arr[tuple(slobj2)]
            slobj2[axis] = np.flatnonzero(keep)  # = keep (a gather)
            p[tuple(slobj)] = sub[tuple(slobj2)]
        return p.result()

    if isinstance(obj, (int, np.integer)) and not isinstance(obj, (bool, np.bool_)):
        single = True
    else:
        single = False
        _obj = obj
        obj = np.asarray(obj)
        if obj.size == 0 and not isinstance(_obj, np.ndarray):
            obj = obj.astype(np.intp)
        elif obj.size == 1 and obj.dtype.kind in "ui":
            obj = obj.item()
            single = True
    if single:
        if obj < -N or obj >= N:
            return _NOPE  # NumPy's IndexError
        if obj < 0:
            obj += N
        newshape[axis] -= 1
        p = new()
        slobj[axis] = slice(None, obj)
        p[tuple(slobj)] = arr[tuple(slobj)]
        slobj[axis] = slice(obj, None)
        slobj2 = [slice(None)] * ndim
        slobj2[axis] = slice(obj + 1, None)
        p[tuple(slobj)] = arr[tuple(slobj2)]
        return p.result()
    if obj.dtype == bool:
        if obj.shape != (N,):
            return _NOPE
        keep = ~obj
    else:
        keep = np.ones(N, dtype=bool)
        try:
            keep[obj,] = False
        except (IndexError, ValueError, TypeError):
            return _NOPE  # NumPy's error
    slobj[axis] = keep
    return arr[tuple(slobj)]


def _insert(arr, obj, values, axis=None):
    if not _vp(arr) or isinstance(obj, ndarray) or arr.ndim == 0:
        return _NOPE
    arr, axis, order = _prepare(arr, axis)
    ndim = arr.ndim
    slobj = [slice(None)] * ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)
    if isinstance(obj, slice):
        indices = np.arange(*obj.indices(N), dtype=np.intp)
    else:
        indices = np.array(obj)
        if indices.dtype == bool or indices.ndim > 1:
            return _NOPE
    if indices.size != 1:
        return _NOPE  # several indices: a scatter (not yet)
    index = indices.item()
    if index < -N or index > N:
        return _NOPE  # NumPy's IndexError
    if index < 0:
        index += N
    if isinstance(values, ndarray):
        values = values.astype(arr.dtype, copy=False)
        if values.ndim < ndim:
            values = values.reshape((1,) * (ndim - values.ndim) + values.shape)
    else:
        values = np.array(values, copy=None, ndmin=ndim, dtype=arr.dtype)
    if indices.ndim == 0:
        values = np.moveaxis(values, 0, axis)
    numnew = values.shape[axis]
    newshape[axis] += numnew
    p = _Placer(arr.dtype, newshape, _order_strides(newshape, arr.dtype.itemsize, order))
    slobj[axis] = slice(None, index)
    p[tuple(slobj)] = arr[tuple(slobj)]
    slobj[axis] = slice(index, index + numnew)
    p[tuple(slobj)] = values
    slobj[axis] = slice(index + numnew, None)
    slobj2 = [slice(None)] * ndim
    slobj2[axis] = slice(index, None)
    p[tuple(slobj)] = arr[tuple(slobj2)]
    return p.result()


def _small_ready(func, handler):
    """Small ready inputs (at most sync_below elements together): nothing to
    overlap, NumPy's own function runs right away on read-only exports (its
    result is fresh memory, adopted); else the handler's task."""
    def h(*args, **kwargs):
        arrays = [x for a in args[:3] + tuple(kwargs.values())
                  for x in (a if isinstance(a, (list, tuple)) else (a,)) if isinstance(x, ndarray)]
        if (arrays and sum(x.size for x in arrays) <= _nd._small_ops[0]
                and all(_viproc.ready(x) for x in arrays)):
            return _nd.fresh(func(*_nd._tree(args, _nd._to_numpy),
                                  **{k: _nd._tree(v, _nd._to_numpy) for k, v in kwargs.items()}))
        return handler(*args, **kwargs)

    return h


JOINING = {f: _small_ready(f, h) for f, h in {
    np.concatenate: _concatenate, np.stack: _stack,
    np.vstack: _stacker(2, lambda arrs: 0),
    np.hstack: _stacker(1, lambda arrs: 0 if arrs[0].ndim == 1 else 1),
    np.dstack: _stacker(3, lambda arrs: 2),
    np.column_stack: _column_stack, np.append: _append,
    np.tile: _tile, np.repeat: _repeat, np.pad: _pad, np.roll: _roll,
    np.delete: _delete, np.insert: _insert,
}.items()}
