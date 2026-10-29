"""The ops viproc runs asynchronously today take the asynchronous path:
tasks on worker threads, no fallback to NumPy, no sync point (async_path.py,
hokb/viproc#25). Each test also compares the result with NumPy."""

import operator

import numpy as np
import pytest

import viproc
from async_path import SIZE, assert_async, deterministic, pending

rng = np.random.default_rng(25)


@pytest.fixture(autouse=True)
def _deterministic():
    with deterministic():
        yield


def data(shape=(SIZE,), dtype=np.float64):
    """NumPy's array and a pending viproc array of the same values."""
    x = (rng.random(shape) + 0.5).astype(dtype)
    return x, pending(viproc.asarray(x))


def same(r, expected):
    got = np.asarray(r)
    expected = np.asarray(expected)
    assert got.dtype == expected.dtype and got.shape == expected.shape
    assert np.array_equal(got, expected, equal_nan=True)


def test_counters_count_tasks_on_pending_inputs_and_blocking_syncs():
    x, v = data()
    before = viproc.offload_stats()
    for _ in range(10):
        v = np.sin(v)  # each on the pending result of the one before
    viproc.wait_all()
    after = viproc.offload_stats()
    assert after["tasks"] - before["tasks"] == 10
    assert after["syncs"] == before["syncs"]  # wait_all() is not counted
    v.wait()  # ready: no sync that blocks
    assert viproc.offload_stats()["syncs"] == after["syncs"]
    big = viproc.asarray(rng.random(1_000_000))
    w = pending(big, steps=50)  # milliseconds of work
    w.wait()
    assert viproc.offload_stats()["syncs"] == after["syncs"] + 1


@pytest.mark.parametrize("name", ["sin", "exp", "sqrt", "negative", "isnan", "floor"])
def test_unary_ufunc(name):
    x, v = data()
    f = getattr(np, name)
    same(assert_async(f, v), f(x))


@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, operator.truediv,
                                operator.pow, operator.lt, np.maximum, np.arctan2])
def test_binary_op(op):
    (x, v), (y, w) = data(), data()
    same(assert_async(op, v, w), op(x, y))
    same(assert_async(op, v, 1.5), op(x, 1.5))


def test_ternary_ufunc_clip():
    from numpy._core import umath
    x, v = data()
    same(assert_async(umath.clip, v, 0.7, 1.2), umath.clip(x, 0.7, 1.2))


def test_scalar_operands_are_cached_per_object_and_dtype():
    # Scalar operands are converted once per (object, dtype) into a small
    # direct-mapped cache (hokb/viproc#47). Many distinct scalars, two per op,
    # replace slots while an earlier operand of the same op still uses one.
    from numpy._core import umath
    x, v = data()
    los = [float(f) for f in rng.random(300) * 0.5 + 0.4]
    for lo in los:
        hi = lo + 0.3
        same(umath.clip(v, lo, hi), umath.clip(x, lo, hi))
    # One object, several dtypes: one entry per dtype.
    two = 2
    for dtype in (np.int32, np.int64, np.float32, np.complex64):
        y, w = data(dtype=dtype)
        same(assert_async(operator.mul, w, two), y * two)
    # A failed conversion raises NumPy's error and caches nothing.
    y, w = data(dtype=np.int32)
    with pytest.raises(OverflowError) as got:
        w + 2**40
    with pytest.raises(OverflowError) as want:
        y + 2**40
    assert str(got.value) == str(want.value)
    same(w + 3, y + 3)


def test_cast_of_an_input_is_a_task_of_its_own():
    (x, v), (y, w) = data(), data(dtype=np.float32)
    same(assert_async(operator.add, v, w, ops=2), x + y)


def test_in_place_and_out():
    (x, v), (y, w), (z, u) = data(), data(), data()
    assert_async(operator.iadd, v, w)
    same(v, x + y)
    assert_async(np.multiply, v, w, out=(u,))
    same(u, (x + y) * y)


@pytest.mark.parametrize("call, ops", [
    (lambda a: a.sum(), 1), (lambda a: np.sum(a), 1), (lambda a: a.sum(axis=0), 1),
    (lambda a: a.sum(axis=1), 1), (lambda a: a.prod(axis=1), 1), (lambda a: np.max(a), 1),
    (lambda a: a.min(axis=0, keepdims=True), 1), (lambda a: np.add.reduce(a, axis=0), 1),
    (lambda a: np.maximum.reduce(a, axis=1), 1),
    (lambda a: (a > 1).any(axis=0), 2), (lambda a: (a > 0).all(), 2),  # comparison + reduction
])
def test_reduction(call, ops):
    x, v = data((100, 100))
    same(assert_async(call, v, ops=ops), call(x))


@pytest.mark.xfail(strict=True, reason="argmin/argmax decide the direct path by their output "
                   "size (vp_arg_reduce passes no work size, vp_reduce passes the input's): an "
                   "argmax of ready data with a small result runs on the main thread "
                   "(hokb/viproc#25)")
@pytest.mark.parametrize("call", [lambda a: a.argmax(), lambda a: np.argmin(a),
                                  lambda a: a.argmax(axis=1)])
def test_argmax_of_ready_data(call):
    x, v = data((100, 100))
    v.wait()
    same(assert_async(call, v), call(x))


@pytest.mark.parametrize("name, ops", [("mean", 2), ("var", 6), ("std", 7)])
def test_moments(name, ops):
    x, v = data()
    same(assert_async(getattr(v, name), ops=ops, small_steps=True), getattr(x, name)())


def test_matmul_and_dot():
    (x, v), (y, w) = data((100, 100)), data((100, 100))
    same(assert_async(operator.matmul, v, w), x @ y)
    same(assert_async(np.dot, v, w), np.dot(x, y))


def test_where():
    (c, cv), (x, v), (y, w) = data(), data(), data()
    cond, cond_v = c > 1, cv > 1
    same(assert_async(np.where, cond_v, v, w), np.where(cond, x, y))
    same(assert_async(np.where, cond_v, v, 0.0), np.where(cond, x, 0.0))


def test_gather_scatter_and_masked_assignment():
    x, v = data()
    idx = rng.integers(0, SIZE, SIZE)
    iv = pending(viproc.asarray(idx))
    same(assert_async(operator.getitem, v, iv), x[idx])
    small = rng.integers(0, SIZE, 100)
    assert_async(operator.setitem, v, viproc.asarray(small), 2.0)
    x[small] = 2.0
    same(v, x)
    mask, mask_v = x > 1, v > 1
    assert_async(operator.setitem, v, mask_v, 0.0)
    x[mask] = 0.0
    same(v, x)


def test_basic_assignment():
    x, v = data()
    y, w = data((SIZE // 2,))
    assert_async(operator.setitem, v, slice(0, None, 2), w)
    x[::2] = y
    same(v, x)
    assert_async(operator.setitem, v, slice(1, None, 2), 3.0)
    x[1::2] = 3.0
    same(v, x)


def test_astype_copy_and_reshape_copy():
    x, v = data((50, 100))
    same(assert_async(v.astype, np.float32), x.astype(np.float32))
    same(assert_async(v.copy), x.copy())
    same(assert_async(lambda: v.T.reshape(-1)), x.T.reshape(-1))


def test_views_issue_no_task():
    x, v = data((50, 100))
    same(assert_async(lambda: v.T, ops=0), x.T)
    same(assert_async(np.swapaxes, v, 0, 1, ops=0), np.swapaxes(x, 0, 1))
    same(assert_async(operator.getitem, v, (slice(None, None, 2), 3), ops=0), x[::2, 3])


def test_direct_budget_is_scaled_to_the_machine():
    """The direct-path budget stands for main-thread costs: the bridge sets
    it to 9 times the measured time of a small direct op (clamped to
    1-50 us), once per process."""
    viproc.init(sync_below=2048, adaptive=True)
    try:
        budget = viproc._viproc.direct_budget_ns()
        assert 1000 <= budget <= 50000
        viproc.init(sync_below=2048, adaptive=True)
        assert viproc._viproc.direct_budget_ns() == budget
    finally:
        viproc.shutdown()


def test_expensive_small_op_becomes_a_task():
    """Below sync_below, an op runs directly on the main thread unless its
    loop is expensive: the direct path times the loop (DirectCost) and ops
    worth more than ~10 us of NumPy time become tasks, as large ones.
    Adaptive mode only: deterministic mode keeps the fixed threshold."""
    viproc.init(sync_below=2048, adaptive=True)
    try:
        data = np.exp(1j * np.arange(1500.0))
        x = viproc.asarray(data)
        viproc.wait_all()
        before = viproc.offload_stats()
        results = [np.power(x, 1.5 + 0.5j) for _ in range(6)]  # ~100 ns per element
        viproc.wait_all()
        after = viproc.offload_stats()
        assert after["direct"] - before["direct"] == 2  # the two timed runs
        assert after["tasks"] - before["tasks"] == 4
        expected = np.power(data, 1.5 + 0.5j)
        assert all(np.array_equal(np.asarray(r), expected) for r in results)
        # Every 32nd promoted op re-tests the loop, as a task timed on its
        # worker: none runs on the main thread.
        before = viproc.offload_stats()
        results = [np.power(x, 1.5 + 0.5j) for _ in range(70)]
        viproc.wait_all()
        after = viproc.offload_stats()
        assert after["direct"] == before["direct"]
        assert after["tasks"] - before["tasks"] == 70
        assert all(np.array_equal(np.asarray(r), expected) for r in results)
        # A cheap loop stays on the direct path.
        y = viproc.asarray(np.arange(1500.0))
        before = viproc.offload_stats()
        for _ in range(6):
            np.add(y, y)
        assert viproc.offload_stats()["direct"] - before["direct"] == 6
    finally:
        viproc.init()
    with deterministic():
        x = viproc.asarray(data)
        viproc.wait_all()
        before = viproc.offload_stats()
        for _ in range(6):
            np.power(x, 1.5 + 0.5j)
        assert viproc.offload_stats()["direct"] - before["direct"] == 6
