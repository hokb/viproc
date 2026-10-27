"""np.sort, np.argsort, ndarray.sort and ndarray.argsort on NumPy's sort
functions (hokb/viproc#25 step 7a): bit-identical to NumPy (values, the
order of equal elements, NaNs and signed zeros, dtypes, layouts, errors) and
on the asynchronous path."""

import numpy as np
import pytest

import viproc
from async_path import assert_async, pending
from test_compositions import DTYPES, check, det, same, sample  # noqa: F401 (det: fixture)
from test_joining import path  # noqa: F401 (fixture)

KINDS = [{}, {"kind": "quicksort"}, {"kind": "heapsort"}, {"kind": "mergesort"},
         {"kind": "stable"}, {"stable": True}, {"stable": False}]


def ties(shape, dtype):
    """Many equal values (and -0.0 next to 0.0, NaNs): the order of equal
    elements shows in argsort and in the bits of a sorted float array."""
    x = sample(shape, dtype)
    if np.dtype(dtype).kind in "fc":
        x = np.round(x)
        if x.size > 10:
            flat = x.reshape(-1)
            flat[::7] = -0.0
            flat[2::11] = 0.0
            flat[4::13] = np.nan
    return x


@pytest.mark.parametrize("shape", [(40,), (6, 9), (3, 4, 5), (2, 0, 3), (1,), (5, 1)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_sort(shape, dtype, path):
    x = ties(shape, dtype)
    for kw in KINDS:
        for f in (np.sort, np.argsort):
            check(lambda a: f(a, **kw), x)
            check(lambda a: f(a, axis=None, **kw), x)
            check(lambda a: f(a, axis=0, **kw), x)
        check(lambda a: a.argsort(**kw), x)


@pytest.mark.parametrize("dtype", DTYPES)
def test_sort_in_place(dtype, path):
    x = ties((6, 9), dtype)
    for kw in KINDS:
        for axis in (-1, 0, 1):
            def f(a):
                a = a.copy()
                a.sort(axis, **kw)
                return a
            check(f, x)

            def g(a):  # through a view: the base changes
                b = a.copy()
                b[1:, ::2].sort(axis, **kw)
                return b
            check(g, x)


def test_sort_in_place_export(path):
    """An export taken before a.sort() sees the sort, as NumPy's view."""
    x = ties((60, 50), np.float64)
    v = viproc.asarray(x)
    e = np.asarray(v)
    v.sort()
    same(e, np.sort(x))
    same(v, np.sort(x))


def test_errors(path):
    x = ties((6, 9), np.float64)
    check(lambda a: np.sort(a, axis=2), x)  # AxisError
    check(lambda a: np.sort(a, kind="x"), x)  # ValueError
    check(lambda a: np.sort(a, kind="stable", stable=True), x)
    check(lambda a: np.argsort(a, axis=1.0), x)  # TypeError
    check(lambda a: np.sort(a, order="f"), x)  # ValueError: no fields
    check(lambda a: np.sort(a[0, 0]), x)  # 0-d: AxisError
    check(lambda a: np.sort(a[0, 0], axis=None), x)
    check(lambda a: a.sort(axis=3), x)

    def ro(a):
        b = np.broadcast_to(a, (2,) + a.shape)
        b.sort()
    check(ro, x)  # read-only: ValueError


ASYNC = [  # name, call, shape, dtype, tasks
    ("sort", np.sort, (100, 64), "f8", 1),
    ("sort_axis0", lambda a: np.sort(a, axis=0), (100, 64), "f8", 1),
    ("sort_stable", lambda a: np.sort(a, kind="stable"), (100, 64), "i8", 1),
    ("sort_none", lambda a: np.sort(a, axis=None), (100, 64), "f4", 1),
    ("sort_t", lambda a: np.sort(a.T), (100, 64), "c16", 1),
    ("argsort", np.argsort, (100, 64), "f8", 1),
    ("argsort_stable", lambda a: a.argsort(axis=0, stable=True), (100, 64), "i4", 1),
    ("argsort_none", lambda a: np.argsort(a, axis=None), (100, 64), "i8", 1),
    # a.sort() on the pending array: one in/out task
    ("method", lambda a: (a.sort(), a)[1], (100, 64), "f8", 1),
    ("method_view", lambda a: (a[::2].sort(axis=0), a)[1], (100, 64), "c8", 1),
]


@pytest.mark.parametrize("call,shape,dtype,ops", [c[1:] for c in ASYNC],
                         ids=[c[0] for c in ASYNC])
def test_async(call, shape, dtype, ops, det):
    x = ties(shape, dtype)
    v = pending(viproc.asarray(x))
    got = assert_async(call, v, ops=ops)
    same(got, call(x.copy()))
