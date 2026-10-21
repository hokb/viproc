"""Indexing functions on viproc's gather / scatter / select tasks
(_indexing.py, hokb/viproc#25 step 4): take, take_along_axis, put,
put_along_axis, compress, extract, place, select, copyto(where=) and
a[mask] = array, bit-identical to NumPy (values, dtype, shape, layout and the
written arrays) and on the asynchronous path."""

import warnings

import numpy as np
import pytest

import viproc
from async_path import SIZE, assert_async
from test_compositions import DTYPES, big, det, sample, same  # noqa: F401
from test_compositions import check as check_now

rng = np.random.default_rng(4)
SHAPES = [(7,), (4, 5), (3, 4, 2)]


def data(shape, dtype=np.float64):
    return sample(shape, dtype, special=False)


def synced(fn):
    """fn with its result synced: an index error of a task (AsyncIndexError,
    an IndexError) is raised in the call, as NumPy raises it."""
    def call(*args):
        r = fn(*args)
        if isinstance(r, viproc.ndarray):
            r.wait()
        return r

    return call


def check(fn, *arrays):
    check_now(synced(fn), *arrays)


def check_inplace(fn, x, *rest):
    """fn writes into its first argument: the NumPy and the viproc array (all
    layouts) end up equal, the calls return the same and raise alike."""
    views = [lambda a: a]
    if x.ndim >= 2:
        views += [lambda a: a.T, lambda a: a[::2, ::-1]]
    elif x.ndim == 1:
        views.append(lambda a: a[::-3])
    for view in views:  # a fresh base each time: fn writes into it
        xn, v = view(x.copy()), view(viproc.asarray(x))
        vr = [viproc.asarray(a) if isinstance(a, np.ndarray) else a for a in rest]
        try:
            want = fn(xn, *rest)
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                fn(v, *vr)
                viproc.wait_all()
            continue
        got = fn(v, *vr)
        assert got is None and want is None
        same(v, xn)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_take(shape, dtype):
    x = data(shape, dtype)
    n = shape[0]
    for idx in ([0, 2, 1], [-1, 0, 0], np.array([[1, 0], [2, 2]]), 2, -1, np.int32(1),
                np.array([1, 3], np.int32)):
        check((lambda a: np.take(a, idx)), x)
        check((lambda a: a.take(idx)), x)
        for axis in range(-len(shape), len(shape)):
            check((lambda a: np.take(a, idx, axis=axis)), x)
    for mode in ("wrap", "clip"):
        check((lambda a: np.take(a, [-7, 0, n, 3 * n + 1], mode=mode)), x)
        check((lambda a: np.take(a, [-7, 0, n, 3 * n + 1], axis=0, mode=mode)), x)
    check((lambda a: np.take(a, [0, x.size])), x)  # IndexError (async)
    check((lambda a: np.take(a, [0.5])), x)  # TypeError (float indices)
    # a viproc index array, also in the modes
    check((lambda a, i: np.take(a, i, axis=0)), x, np.array([n - 1, 0, -n], np.int64))
    check((lambda a, i: np.take(a, i, axis=0, mode="wrap")), x,
          np.array([5 * n, -1], np.int32))
    check((lambda a, i: np.take(a, i, mode="clip")), x, np.array([[-3, 99]], np.int64))


def test_take_out_of_bounds_is_async():
    v = viproc.asarray(data((10,)))
    i = viproc.asarray(np.array([0, 10]))
    with pytest.raises(IndexError):  # at the call (ready data, run directly) or at the sync
        np.take(v, i).numpy()
    viproc.wait_all()


@pytest.mark.parametrize("shape", SHAPES[1:])
def test_take_and_put_along_axis(shape):
    x = data(shape)
    for axis in range(-len(shape), len(shape)):
        n = shape[axis]
        ishape = list(shape)
        ishape[axis] = 3
        idx = rng.integers(-n, n, ishape)
        check(lambda a, i: np.take_along_axis(a, i, axis), x, idx)
        check(lambda a: np.take_along_axis(a, idx, axis), x)
        check_inplace(lambda a, i: np.put_along_axis(a, i, -5.0, axis), x, idx)
        vals = data(ishape)
        check_inplace(lambda a, i, w: np.put_along_axis(a, i, w, axis), x, idx, vals)
    flat = rng.integers(0, x.size, 4)
    check(lambda a: np.take_along_axis(a, flat, None), x)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_put(shape, dtype):
    x = data(shape, dtype)
    size = x.size
    v3 = data((3,), dtype)
    for ind, vals in (([0, 2, 4], v3), ([1, 1, 1], v3), ([-1, 0], v3), ([[0, 3], [1, 2]], v3[:1]),
                      ([0, 1, 2, 3, 5], v3), (2, v3[0]), ([], v3)):
        check_inplace(lambda a: np.put(a, ind, vals), x)
        check_inplace(lambda a: a.put(ind, vals), x)
    for mode in ("wrap", "clip"):
        check_inplace(lambda a: np.put(a, [-2 * size, size + 3, 1], v3, mode=mode), x)
    check_inplace(lambda a, i, w: np.put(a, i, w), x, np.array([3, 0, -1]), v3)
    check_inplace(lambda a: np.put(a, [0, 1], [7.9, -2.1]), x)  # cast as NumPy
    check_inplace(lambda a: np.put(a, [0, size], v3), x)  # IndexError (async)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_compress_extract_place(shape, dtype):
    x = data(shape, dtype)
    cond = rng.random(shape[0]) > 0.4
    for axis in (None,) + tuple(range(-len(shape), len(shape))):
        c = rng.random(x.size if axis is None else shape[axis]) > 0.4
        check(lambda a: np.compress(c, a, axis=axis), x)
        check(lambda a: a.compress(c.tolist(), axis), x)
        check(lambda a, m: np.compress(m, a, axis=axis), x, c)  # a viproc condition
        check(lambda a: np.compress(c[:2], a, axis=axis), x)  # shorter: a prefix
    check(lambda a: np.compress(np.ones(shape[0] + 1, bool), a, axis=0), x)  # IndexError
    check(lambda a: np.compress(cond, a, axis=0), x)
    mask = rng.random(shape) > 0.5
    check(lambda a: np.extract(mask, a), x)
    check(lambda a, m: np.extract(m, a), x, mask)
    check(lambda a: np.extract(a != a.flat[0], a), x)
    w = data((2,), dtype)
    check_inplace(lambda a: np.place(a, mask, w), x)
    check_inplace(lambda a, m: np.place(a, m, [1, 2, 3]), x, mask)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_select_and_copyto(shape, dtype):
    x, y = data(shape, dtype), data(shape, dtype)
    c1, c2 = x != x.flat[0], rng.random(shape) > 0.5
    check(lambda a, b: np.select([c1, c2], [a, b]), x, y)
    check(lambda a, b: np.select([c1, c2], [a, b], default=a.flat[1]), x, y)
    check(lambda a, b, m: np.select([m, c2], [a, 5], default=1), x, y, c1)
    check(lambda a: np.select([c2], [a], 2.5), x)  # promotion
    check(lambda a: np.select([c2[..., :1]], [a.T.T], 0), x)  # broadcast conditions
    check(lambda a: np.select([c1, c2], [a]), x)  # ValueError
    check(lambda a: np.select([a], [a]), x)  # TypeError unless bool
    check_inplace(lambda a, b: np.copyto(a, b, where=c2), x, y)
    check_inplace(lambda a, b, m: np.copyto(a, b, where=m), x, y, c2)
    check_inplace(lambda a: np.copyto(a, 3, where=c2), x)
    check_inplace(lambda a, b: np.copyto(a, b[..., :1], where=c2), x, y)
    check_inplace(lambda a: np.copyto(a, 1.5, where=c2), x)  # same_kind: TypeError for ints
    check_inplace(lambda a: np.copyto(a, x.astype(np.complex128), where=c2, casting="unsafe"),
                  x)


@pytest.mark.parametrize("shape", SHAPES)
def test_mask_assignment_of_arrays(shape):
    x = data(shape)
    mask = rng.random(shape) > 0.5
    k = int(mask.sum())

    def assign(a, m, w):
        a[m] = w

    check_inplace(assign, x, mask, data((k,)))
    check_inplace(lambda a, m: assign(a, m, np.arange(k, dtype=float)), x, mask)
    check_inplace(lambda a, m: assign(a, m, [1.5]), x, mask)


ASYNC = [  # name, call, inputs (shape, dtype), tasks, syncs
    ("take", lambda a: np.take(a, np.arange(0, SIZE, 2)), [((SIZE,), "f8")], 1, 0),
    ("take_axis", lambda a, i: np.take(a, i, axis=1), [((50, 100), "f8"), ((90,), "i8")], 1, 0),
    ("take_wrap", lambda a, i: np.take(a, i, mode="wrap"), [((SIZE,), "f8"), ((SIZE,), "i8")],
     1, 0),
    ("take_clip", lambda a, i: np.take(a, i * 100, mode="clip"),
     [((SIZE,), "f8"), ((SIZE,), "i8")], 2, 0),
    ("take_method", lambda a: a.take(np.arange(SIZE)), [((SIZE,), "f8")], 1, 0),
    ("take_along_axis", lambda a, i: np.take_along_axis(a, i % 100, 1),
     [((50, 100), "f8"), ((50, 100), "i8")], 2, 0),
    ("put", lambda a, i: np.put(a, i % SIZE, 1.5), [((SIZE,), "f8"), ((SIZE,), "i8")], 2, 0),
    ("put_method", lambda a, w: a.put(np.arange(SIZE), w), [((SIZE,), "f8"), ((SIZE,), "f8")],
     1, 0),
    ("put_along_axis", lambda a, i: np.put_along_axis(a, i % 100, 0.5, 1),
     [((50, 100), "f8"), ((50, 100), "i8")], 2, 0),
    # the condition decides the result's shape: one sync on it, as boolean indexing
    ("compress", lambda a, c: np.compress(c > 0, a), [((SIZE,), "f8"), ((SIZE,), "f8")], 2, 1),
    ("compress_method", lambda a: a.compress(np.arange(SIZE) % 2 == 0), [((SIZE,), "f8")], 1, 0),
    ("extract", lambda a: np.extract(np.arange(SIZE) % 2 == 0, a), [((SIZE,), "f8")], 1, 0),
    ("place", lambda a, w: np.place(a, np.arange(SIZE) % 2 == 0, w),
     [((SIZE,), "f8"), ((SIZE // 2,), "f8")], 1, 0),
    ("select", lambda a, b: np.select([a > 0, b > 0], [a, b], 0.0),
     [((SIZE,), "f8"), ((SIZE,), "f8")], 4, 0),
    ("copyto_where", lambda a, b: np.copyto(a, b, where=b > 0),
     [((SIZE,), "f8"), ((SIZE,), "f8")], 3, 0),
    # mask, gather, multiply, scatter; the mask syncs once (its count)
    ("mask_assign_array", lambda a, b: double_where(a, b > 0), [((SIZE,), "f8"), ((SIZE,), "f8")],
     4, 1),
]


def double_where(a, m):
    a[m] = a[m] * 2


@pytest.mark.parametrize("call, inputs, tasks, syncs", [c[1:] for c in ASYNC],
                         ids=[c[0] for c in ASYNC])
def test_async_path(det, call, inputs, tasks, syncs):
    pairs = [big(shape, dtype) for shape, dtype in inputs]
    want_args = [x.copy() for x, _ in pairs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            want = call(*want_args)
        except Exception:  # noqa: BLE001  (a test input NumPy refuses)
            raise
        got = assert_async(call, *(v for _, v in pairs), ops=tasks, syncs=syncs,
                           small_steps=tasks == 0, min_tasks=0)
    same(got, want) if want is not None else None
    for (_, v), w in zip(pairs, want_args):
        same(v, w)  # the arrays the call wrote
