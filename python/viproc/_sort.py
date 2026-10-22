"""np.sort, np.argsort, ndarray.sort and ndarray.argsort as tasks
(hokb/viproc#25, step 7a) on NumPy's per-dtype sort functions (vp_sort).

NumPy sorts each row along the axis on its own (_new_sortlike,
_new_argsortlike); viproc's kernel calls the same functions on the same
rows, so results (also the order of equal elements, NaNs and signed zeros)
are NumPy's. Kinds: quicksort and heapsort run NumPy's introsort (NumPy 2
maps heapsort to it), mergesort and stable its timsort/radix sort. np.sort
is NumPy's `copy(order="K")` sorted in place: the result has the copy's
layout; argsort's result is C-ordered intp. axis=None sorts the C-ordered
flat copy. order= (structured dtypes) and other arguments fall back. Small
ready data takes vp_sort's direct path too (no task). np.sort(a[, axis])
of a C-contiguous array and np.argsort(a[, axis]) skip this module:
_Array.__array_function__ issues them in C (sort_call in _viproc.c).
"""

import numpy as np
from numpy.lib.array_utils import normalize_axis_index

from . import _ndarray as _nd
from . import _viproc

_NOPE = _nd._NOPE
ndarray = _nd.ndarray

_KINDS = {None: 0, "quicksort": 0, "heapsort": 0, "mergesort": 2, "stable": 2}


def _kind(kind, stable):
    """NumPy's table index for (kind, stable), or None for what NumPy
    checks itself (unknown kinds, both given)."""
    if stable is not None:
        if kind is not None or type(stable) is not bool:
            return None
        return 2 if stable else 0
    return _KINDS.get(kind) if kind is None or type(kind) is str else None


def _args(a, axis, kind, order, stable):
    """(array, axis, kind) of a call viproc runs, or None."""
    if type(a) is not ndarray:
        a = _nd.asarray(a)
        if type(a) is not ndarray:
            return None
    k = _kind(kind, stable)
    if order is not None or k is None:
        return None
    if axis is None:
        a = a.reshape(-1)  # NumPy's flatten: C order
        axis = 0
    elif a.ndim == 0 or type(axis) is not int:
        return None  # NumPy's AxisError / TypeError
    return a, normalize_axis_index(axis, a.ndim), k


def _sort(a, axis=-1, kind=None, order=None, *, stable=None):
    args = _args(a, axis, kind, order, stable)
    if args is None:
        return _NOPE
    a, axis, k = args
    r = _viproc.sort(a, axis, k, 0, False, ())  # vp_sort: copy(order="K")'s layout
    return _NOPE if r is None else r


def _argsort(a, axis=-1, kind=None, order=None, *, stable=None):
    args = _args(a, axis, kind, order, stable)
    if args is None:
        return _NOPE
    a, axis, k = args
    r = _viproc.sort(a, axis, k, 1, False, ())
    return _NOPE if r is None else r


def sort_in_place(a, axis=-1, kind=None, order=None, *, stable=None):
    """ndarray.sort: True if issued as a task, else False (the caller runs
    NumPy's method)."""
    k = _kind(kind, stable)
    if order is not None or k is None or a.ndim == 0 or type(axis) is not int:
        return False
    if not -a.ndim <= axis < a.ndim:
        return False  # NumPy's AxisError
    return _viproc.sort(a, axis % a.ndim, k, 0, True, ()) is True


SORTING = {np.sort: _sort, np.argsort: _argsort}
