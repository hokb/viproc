"""Joining and copying functions viproc runs as one placement task (_join.py,
hokb/viproc#25 step 3): bit-identical to NumPy (values, dtype, shape and
layout) over dtypes, layouts and arguments, and on the asynchronous path."""

import warnings

import numpy as np
import pytest

import viproc
from async_path import SIZE, assert_async, counters
from test_compositions import DTYPES, big, check, det, sample, same  # noqa: F401 (det: fixture)


def pair(shape, dtype):
    return sample(shape, dtype, special=False)


SHAPES = [(7,), (4, 5), (3, 4, 2)]


@pytest.fixture(params=["numpy", "task"])
def path(request):
    """Small ready inputs: NumPy's own function (the default), or the
    placement task (sync_below=0), so both paths meet NumPy's results."""
    if request.param == "numpy":
        yield
        return
    viproc.init(sync_below=0, adaptive=False)
    try:
        yield
    finally:
        viproc.init()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_concatenate_and_stacking(shape, dtype, path):
    x, y = pair(shape, dtype), pair(shape, dtype)
    for axis in list(range(-len(shape), len(shape))) + [None]:
        check(lambda a, b: np.concatenate([a, b, a], axis=axis), x, y)
        check(lambda a, b: np.concatenate((b, a), axis=axis), x, y)
    for axis in range(-len(shape) - 1, len(shape) + 1):
        check(lambda a, b: np.stack([a, b], axis=axis), x, y)
    check(lambda a, b: np.vstack([a, b]), x, y)
    check(lambda a, b: np.hstack([a, b]), x, y)
    check(lambda a, b: np.dstack([a, b]), x, y)
    check(lambda a, b: np.append(a, b), x, y)
    check(lambda a, b: np.append(a, b, axis=0), x, y)
    if len(shape) >= 2:  # 1-d: NumPy's dispatch iterates over it (elements, no viproc array)
        check(lambda a: np.concatenate(a), x)  # the rows of a
    if len(shape) <= 2:
        check(lambda a, b: np.column_stack([a, b]), x, y)


@pytest.mark.parametrize("shape", SHAPES)
def test_concatenate_mixed(shape, path):
    x = pair(shape, np.float64)
    i32 = pair(shape, np.int32)
    f32 = pair(shape, np.float32)
    # promotion, casts, NumPy and Python inputs next to viproc arrays
    check(lambda a, b: np.concatenate([a, b]), x, i32)
    check(lambda a, b: np.concatenate([a, b]), f32, i32)
    check(lambda a, b: np.concatenate([a, b]), i32, x)
    check(lambda a: np.concatenate([a, i32]), x)
    check(lambda a: np.concatenate([a, a], dtype=np.float32), x)
    check(lambda a: np.concatenate([a, a], dtype=np.int64, casting="unsafe"), x)
    check(lambda a: np.concatenate([a, a], dtype=np.int64), x)  # TypeError: same_kind
    check(lambda a: np.concatenate([a, a.astype(np.float32)[..., :1]], axis=0), x)  # shapes
    check(lambda a: np.concatenate([a, a.tolist()]), x)
    check(lambda a: np.stack([a, a[..., ::-1]], dtype=np.complex64), x)
    check(lambda a: np.concatenate([a]), x)
    check(lambda a: np.concatenate([a[:0], a]), x)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tile_and_repeat(shape, dtype, path):
    x = pair(shape, dtype)
    for reps in (2, (2, 3), (1, 1), (3, 1, 2), (2, 1, 1, 2), 1, (0, 2)):
        check(lambda a: np.tile(a, reps), x)
    for r in (1, 3, [2], np.int32(2), 0):
        check(lambda a: np.repeat(a, r), x)
        for axis in range(-len(shape), len(shape)):
            check(lambda a: np.repeat(a, r, axis=axis), x)
            check(lambda a: a.repeat(r, axis), x)
    check(lambda a: np.repeat(a, list(range(shape[0])), axis=0), x)  # per element (fallback)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_pad(shape, dtype, path):
    x = pair(shape, dtype)
    check(lambda a: np.pad(a, 2), x)
    check(lambda a: np.pad(a, (1, 3)), x)
    check(lambda a: np.pad(a, [(i, 2 - i) for i in range(len(shape))]), x)
    check(lambda a: np.pad(a, 1, constant_values=7), x)
    check(lambda a: np.pad(a, 2, constant_values=(1, 2)), x)
    check(lambda a: np.pad(a, 2, constant_values=[(i, 10 + i) for i in range(len(shape))]), x)
    check(lambda a: np.pad(a, ((0, 0),) * len(shape)), x)
    check(lambda a: np.pad(a, 1, constant_values=2.7), x)  # cast as NumPy's assignment
    check(lambda a: np.pad(a, 2, mode="edge"), x)  # fallback
    check(lambda a: np.pad(a, -1), x)  # ValueError


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_roll(shape, dtype, path):
    x = pair(shape, dtype)
    for shift in (0, 1, -3, 100):
        check(lambda a: np.roll(a, shift), x)
        for axis in range(-len(shape), len(shape)):
            check(lambda a: np.roll(a, shift, axis=axis), x)
    if len(shape) >= 2:
        check(lambda a: np.roll(a, (1, -2), axis=(0, 1)), x)
        check(lambda a: np.roll(a, 2, axis=(1, 0)), x)
        check(lambda a: np.roll(a, (1, 2), axis=(1, 1)), x)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_delete_and_insert(shape, dtype, path):
    x = pair(shape, dtype)
    for axis in list(range(-len(shape), len(shape))) + [None]:
        for obj in (0, -1, 1, slice(1, 3), slice(None, None, 2), slice(3, 0, -2), slice(5, 1),
                    [0, 2], np.array([True] + [False] * (shape[0] - 1)), [], 99):
            check(lambda a: np.delete(a, obj, axis=axis), x)
        for obj in (0, -1, 2, [1], slice(1, 2), [0, 2], 99):
            check(lambda a: np.insert(a, obj, 7, axis=axis), x)
            check(lambda a: np.insert(a, obj, [5, 6], axis=axis), x)
        check(lambda a, b: np.insert(a, 1, b, axis=axis), x, pair((1,), dtype))


def test_diff_prepend_append(path):
    x = pair((6, 7), np.float64)
    check(lambda a: np.diff(a, prepend=0), x)
    check(lambda a: np.diff(a, append=[[1.0]] * 6), x)
    check(lambda a: np.diff(a, 2, axis=0, prepend=-1, append=x[:1]), x)


ASYNC = [  # name, call, inputs (shape, dtype), tasks
    ("concatenate", lambda a, b: np.concatenate([a, b, a]), [((SIZE,), "f8"), ((SIZE,), "f8")], 1),
    ("concatenate_axis1", lambda a, b: np.concatenate([a, b], axis=1),
     [((100, 50), "f8"), ((100, 30), "f8")], 1),
    ("concatenate_none", lambda a, b: np.concatenate([a, b.T], axis=None),
     [((100, 50), "f8"), ((50, 100), "f8")], 1),
    ("concatenate_cast", lambda a, b: np.concatenate([a, b]), [((SIZE,), "f8"), ((SIZE,), "i4")],
     2),
    ("stack", lambda a, b: np.stack([a, b], axis=1), [((SIZE,), "f8"), ((SIZE,), "f8")], 1),
    ("vstack", lambda a, b: np.vstack([a, b]), [((SIZE,), "f8"), ((SIZE,), "f8")], 1),
    ("hstack", lambda a, b: np.hstack([a, b]), [((SIZE,), "f8"), ((SIZE,), "f8")], 1),
    ("dstack", lambda a, b: np.dstack([a, b]), [((100, 50), "f8"), ((100, 50), "f8")], 1),
    ("column_stack", lambda a, b: np.column_stack([a, b]), [((SIZE,), "f8"), ((SIZE,), "f8")],
     1),
    ("append", np.append, [((SIZE,), "f8"), ((SIZE,), "f8")], 1),
    ("tile", lambda a: np.tile(a, (3, 2)), [((100, 60), "f8")], 1),
    ("repeat", lambda a: np.repeat(a, 3, axis=0), [((100, 60), "f8")], 1),
    ("repeat_method", lambda a: a.repeat(2), [((SIZE,), "f8")], 1),
    ("pad", lambda a: np.pad(a, ((1, 2), (3, 4)), constant_values=5), [((60, 50), "f8")], 1),
    ("roll", lambda a: np.roll(a, (3, -7), axis=(0, 1)), [((60, 50), "f8")], 1),
    ("roll_flat", lambda a: np.roll(a, 7), [((SIZE,), "f8")], 1),
    ("delete", lambda a: np.delete(a, 5), [((SIZE,), "f8")], 1),
    ("delete_slice", lambda a: np.delete(a, slice(10, 90, 3), axis=1), [((100, 100), "f8")], 2),
    ("delete_list", lambda a: np.delete(a, [1, 5, 9]), [((SIZE,), "f8")], 1),
    ("insert", lambda a: np.insert(a, 7, [1.0, 2.0]), [((SIZE,), "f8")], 1),
    ("insert_array", lambda a, b: np.insert(a, 1, b, axis=1), [((SIZE, 2), "f8"), ((SIZE,), "f8")],
     1),
    ("diff_prepend", lambda a: np.diff(a, prepend=0.0), [((SIZE,), "f8")], 2),
]


def test_small_ready_inputs_run_in_numpy(det):
    """Small ready inputs: NumPy's function right away, no task, no fallback."""
    x = sample((100,), np.float64, special=False)
    v = viproc.asarray(x)
    viproc.wait_all()
    before = counters()
    r = np.concatenate([v, v])
    after = counters()
    assert type(r) is viproc.ndarray
    assert after["tasks"] == before["tasks"] and after["fallbacks"] == before["fallbacks"]
    same(r, np.concatenate([x, x]))


@pytest.mark.parametrize("call, inputs, tasks", [c[1:] for c in ASYNC], ids=[c[0] for c in ASYNC])
def test_async_path(det, call, inputs, tasks):
    pairs = [big(shape, dtype) for shape, dtype in inputs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = call(*(x for x, _ in pairs))
        got = assert_async(call, *(v for _, v in pairs), ops=tasks)
    same(got, want)


@pytest.mark.parametrize("vshape", [(1, 1, 3), (1, 1, 1, 1), (2, 1, 3), (1, 2, 3), (1, 1, 2)])
def test_insert_values_with_extra_axes(vshape):
    """NumPy's assignment drops a value's leading length-1 axes and raises its
    broadcast error otherwise (hokb/viproc#29 review)."""
    x = np.arange(3000.0).reshape(1000, 3)
    v = np.arange(np.prod(vshape), dtype=float).reshape(vshape) + 0.5

    def run(xp, a):
        try:
            return xp.insert(a, 1, v, axis=0)
        except Exception as e:
            return type(e), str(e)

    expected, got = run(np, x), run(viproc, viproc.asarray(x))
    if isinstance(expected, tuple):
        assert got == expected
    else:
        same(got, expected)
