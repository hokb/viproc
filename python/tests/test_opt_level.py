"""Optimization levels (docs/design/goals.md): O1 is the default and
bit-identical to NumPy; O2 runs every BLAS call on one thread and no longer
serializes LAPACK calls, with results that are correct but may deviate."""

import warnings

import numpy as np
import pytest

import viproc
from viproc import _blas, _linalg, _ufuncs

from async_path import SYNC_BELOW, assert_async, pending


@pytest.fixture(autouse=True, scope="module")
def default_env():
    """These tests start from the default level, whatever VIPROC_OPT the
    suite runs with; the runtime of that level is restored after them."""
    import os

    saved = os.environ.pop("VIPROC_OPT", None)
    viproc.init()
    try:
        yield
    finally:
        if saved is not None:
            os.environ["VIPROC_OPT"] = saved
        viproc.init()


@pytest.fixture
def level2():
    viproc.init(sync_below=SYNC_BELOW, adaptive=False, level=2)
    try:
        yield
    finally:
        viproc.init()


def test_default_is_o1():
    viproc.init()
    assert viproc.opt_level() == 1


@pytest.mark.parametrize("bad", [0, 6, "2", 2.0, True])
def test_invalid_level(bad):
    viproc.init()
    with pytest.raises(ValueError):
        viproc.init(level=bad)
    assert viproc.opt_level() == 1  # the running runtime is kept


@pytest.mark.parametrize("value, level", [("2", 2), ("O2", 2), ("o2", 2), ("5", 5)])
def test_env(monkeypatch, value, level):
    monkeypatch.setenv("VIPROC_OPT", value)
    try:
        viproc.init()
        assert viproc.opt_level() == level
    finally:
        monkeypatch.delenv("VIPROC_OPT")
        viproc.init()
    assert viproc.opt_level() == 1


def test_blas_threads_restored():
    viproc.init()
    before = _blas.threads()
    if before is None:
        pytest.skip("no BLAS library with a thread setting found")
    viproc.init(level=2)
    assert _blas.threads() == 1
    viproc.init()
    assert _blas.threads() == before
    viproc.init(level=2)
    viproc.shutdown()
    assert _blas.threads() == before
    viproc.init()


def _flags_seen(monkeypatch):
    seen = []
    real = _ufuncs.gufunc

    def spy(*args):
        seen.append(args[5])
        return real(*args)

    monkeypatch.setattr(_ufuncs, "gufunc", spy)
    return seen


@pytest.mark.parametrize("level", [1, 2])
def test_lapack_serialized_only_at_o1(monkeypatch, level):
    viproc.init(level=level)
    try:
        seen = _flags_seen(monkeypatch)
        a = np.random.default_rng(0).random((80, 80)) + 80 * np.eye(80)
        np.linalg.inv(viproc.asarray(a)).wait()
        assert bool(seen[-1] & _linalg._SERIAL) == (level == 1)
    finally:
        viproc.init()


def test_o2_results(level2):
    rng = np.random.default_rng(1)
    a = rng.random((300, 300)) + 300 * np.eye(300)
    b = rng.random((300, 7))
    va, vb = viproc.asarray(a), viproc.asarray(b)
    np.testing.assert_allclose(np.asarray(np.linalg.inv(va)), np.linalg.inv(a), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(np.linalg.solve(va, vb)), np.linalg.solve(a, b),
                               rtol=1e-12)
    np.testing.assert_allclose(np.asarray(va @ vb), a @ b, rtol=1e-13)
    # element-wise ops stay NumPy's, bit-identical
    assert np.array_equal(np.asarray(np.sin(va) * 2.0), np.sin(a) * 2.0)


def test_o2_async(level2):
    a = np.random.default_rng(2).random((8, 30, 30)) + 30 * np.eye(30)
    va = pending(viproc.asarray(a))
    r = assert_async(np.linalg.inv, va)
    np.testing.assert_allclose(np.asarray(r), np.linalg.inv(a), rtol=1e-12)


def _stats():
    st = viproc.offload_stats()
    return st["syncs"], st["size_waits"]


def test_o2_sizes_in_tasks(level2):
    """O2 (pipelining.md, S2): element-wise ufuncs on pending inputs are
    issued without their size; the tasks size them on their inputs' signals,
    while the main thread runs ahead."""
    rng = np.random.default_rng(3)
    a, b = rng.random((2000, 300)), rng.random(300)
    x, y = viproc.asarray(a), viproc.asarray(b)
    slow = np.sin(np.sin(viproc.asarray(np.ones(4_000_000))))[:1]  # keeps the rest pending
    before = _stats()
    z, r = x + slow, a + np.sin(np.sin(np.ones(1)))
    for _ in range(20):
        z = np.cos(z) * y + 1.5
        r = np.cos(r) * b + 1.5
    assert _stats() == before  # no sync, no wait for a size
    assert z.shape == r.shape and z.dtype == r.dtype  # metadata query: waits for the size only
    got = np.asarray(z)
    assert got.strides == r.strides and np.array_equal(got, r)  # dense: NumPy's single call


def test_o2_layouts(level2):
    rng = np.random.default_rng(4)
    for a in (np.asfortranarray(rng.random((300, 40))), rng.random((40, 600)).T,
              rng.random((90, 80))[::-1, ::2]):
        x = pending(viproc.asarray(a))
        z, r = np.exp(x) * 2.0, np.exp(a) * 2.0
        got = np.asarray(z)
        assert got.strides == r.strides  # NumPy's result layout (order K)
        np.testing.assert_allclose(got, r, rtol=1e-15)  # strided: rows, may round differently


def test_o2_broadcast_error_is_asynchronous(level2):
    x = pending(viproc.asarray(np.ones((3000, 3))))
    y = pending(viproc.asarray(np.ones(400)))
    msg = "operands could not be broadcast together with shapes \\(3000,3\\) \\(400,\\)"
    with pytest.raises(viproc.AsyncValueError, match=msg) as e:
        w = x + y  # issued: the task finds the shapes do not broadcast
        w.shape  # a metadata query raises the op's error (as a sync point)
    assert isinstance(e.value, ValueError) and "in numpy.add" in str(e.value)
    with pytest.raises(ValueError, match=msg):
        np.asarray(w * 2.0)  # inherited by its consumers
    viproc.wait_all()  # raised once


def test_o2_zero_d_results_are_scalars(level2):
    x = _pending(np.arange(5000.0))
    s = x.sum()  # a pending 0-d result, NumPy's scalar
    t = np.sqrt(s) * 2.0  # sized by its task: still known to be 0-d
    assert t._scalar and t.ndim == 0
    t2 = t
    t2 += 1.0  # rebinds, as on NumPy's scalars
    assert t2 is not t and float(t2) == float(np.sqrt(np.arange(5000.0).sum()) * 2.0 + 1.0)


@pytest.mark.parametrize("order", ["C", "F", "strided"])
def test_o2_reductions_in_tasks(level2, order):
    """O2: sum/prod/max/min/any/all of pending data are issued without the
    input's size (its rank is known at issue); the task sizes, lays out and
    plans them. Reduced axes that coalesce give NumPy's loop calls."""
    rng = np.random.default_rng(6)
    a = rng.standard_normal((300, 40))
    a = {"C": a, "F": np.asfortranarray(a), "strided": a[::-1, ::2]}[order]
    slow = np.sin(np.sin(viproc.asarray(np.ones(2_000_000))))[:1] * 0
    x = np.sin(viproc.asarray(a) + slow)  # deferred, pending
    w = np.sin(a + np.zeros(1))
    before = _stats()
    calls = [lambda t: t.sum(axis=0), lambda t: t.sum(axis=-1), lambda t: t.max(axis=0),
             lambda t: np.min(t, axis=1, keepdims=True), lambda t: t.prod(axis=(0,)),
             lambda t: (t > 0).any(axis=0), lambda t: (t > -5).all(axis=1)]
    got = [f(x) for f in calls]
    full = x.sum()
    assert _stats() == before  # neither a sync nor a size wait
    assert full._scalar and full.ndim == 0
    for g, f in zip(got, calls):
        want = f(w)
        r = np.asarray(g)
        assert r.dtype == want.dtype and r.shape == want.shape
        assert np.array_equal(r, want)  # one reduced axis: NumPy's loop calls
    np.testing.assert_allclose(float(full), w.sum(), rtol=1e-12)  # blocks may differ


def test_o2_reduction_errors_and_casts(level2):
    def e():  # pending each time: a ready empty array would take NumPy's own call
        return _pending(np.zeros((0, 3)))
    assert np.array_equal(np.asarray(e().sum(axis=0)), np.zeros(3))
    assert np.asarray(e().max(axis=1)).shape == (0,)
    with pytest.raises(viproc.AsyncValueError,
                       match="zero-size array to reduction operation maximum which has no identity"):
        np.asarray(e().max(axis=0))
    i = _pending(np.arange(5000, dtype=np.int32))
    s = i.sum(axis=0)  # NumPy accumulates in int64: the regular path
    assert s.dtype == np.arange(3, dtype=np.int32).sum().dtype and int(s) == 4999 * 5000 // 2


def _slow():
    return np.sin(np.sin(viproc.asarray(np.ones(2_000_000))))


def _pending_int(value, dtype=np.int64, slow=None):
    """A 0-d integer viproc array that is still pending: value + 0 * (a
    slow computation)."""
    slow = _slow() if slow is None else slow
    return (slow[:4] * 0).astype(dtype).sum() + value


@pytest.mark.parametrize("call", [
    lambda n, m: viproc.zeros((n, m)),
    lambda n, m: viproc.zeros(n, dtype=np.int32),
    lambda n, m: viproc.ones((m, n, 2), np.float32, "F"),
    lambda n, m: viproc.empty((n,)) * 0.0,
    lambda n, m: viproc.full((2, m), 2.5),
    lambda n, m: viproc.full(n, 7, dtype=np.complex64),
    lambda n, m: viproc.arange(n),
])
def test_o2_pending_sizes(level2, call):
    """S4: a creator whose size is a pending array issues a task that sizes
    its result when the size is computed; the main thread does not sync."""
    slow = _slow()  # one computation behind both: pending together
    n, m = _pending_int(5, slow=slow), _pending_int(3, np.int32, slow)
    assert not n.ready() and not m.ready()
    before = _stats()
    r = call(n, m)
    w = r + r  # a deferred consumer of it
    assert _stats() == before  # no sync, no size wait
    want = call(5, 3)
    got = np.asarray(r)
    assert got.dtype == want.dtype and got.strides == want.strides
    assert np.array_equal(got, want) and np.array_equal(np.asarray(w), want + want)


def test_o2_pending_negative_size(level2):
    n = _pending_int(-1)
    with pytest.raises(viproc.AsyncValueError, match="negative dimensions are not allowed"):
        z = viproc.zeros((n, 2))  # issued: the task finds the size negative
        z.shape
    assert np.arange(-1).size == 0 and viproc.arange(_pending_int(-1)).shape == (0,)


def _pending(a):
    """`a` as a viproc array behind a slow computation (pending for a while)."""
    slow = np.sin(np.sin(viproc.asarray(np.ones(2_000_000))))
    return (slow[:1] * 0).astype(a.dtype) + viproc.asarray(a)


@pytest.mark.parametrize("dtype", [np.float64, np.int32, np.complex64, np.bool_])
@pytest.mark.parametrize("order", ["C", "F"])
def test_o2_data_dependent_sizes(level2, dtype, order):
    """S4: x[mask], np.flatnonzero and np.nonzero (1-d) of pending data are
    tasks that size their results from the data, without a sync."""
    rng = np.random.default_rng(5)
    a = np.asarray(rng.integers(-2, 3, (300, 40)), dtype=dtype, order=order)
    if dtype is np.complex64:
        a = a + 1j * np.asarray(rng.integers(0, 2, a.shape), dtype=np.float32)
    x = _pending(a)
    want = np.zeros(1, a.dtype) + a  # the same arithmetic, in NumPy
    m = x != 0
    before = _stats()
    r = x[m]  # full mask
    rows = x[(x[:, 0] != 0)]  # mask over the leading axis
    flat = np.flatnonzero(x)
    nz = np.nonzero(x[:, 5])
    assert _stats()[0] == before[0]  # no sync
    for got, ref in ((r, want[want != 0]), (rows, want[want[:, 0] != 0]),
                     (flat, np.flatnonzero(want)), (nz[0], np.nonzero(want[:, 5])[0])):
        g = np.asarray(got)
        assert g.dtype == ref.dtype and g.shape == ref.shape and g.strides == ref.strides
        assert np.array_equal(g, ref)
    assert isinstance(nz, tuple) and len(nz) == 1


def _as_numpy(r):
    """A result as NumPy gives it: a pending 0-d `_scalar` as NumPy's scalar."""
    if isinstance(r, viproc.ndarray):
        return np.asarray(r)[()] if getattr(r, "_scalar", False) else np.asarray(r)
    return r


_S4_REDUCTIONS = [
    lambda v: v.sum(),  # int32, bool: accumulated in int64 (a cast)
    lambda v: v.sum(axis=0, keepdims=True),
    lambda v: np.prod(v),
    lambda v: v.sum(dtype=np.float32),  # float sums on NumPy's buffer blocks
    lambda v: v.sum(dtype=np.complex128),
    lambda v: v.sum(initial=3),
    lambda v: v.max(initial=100),
    lambda v: v.mean(),
    lambda v: v.mean(keepdims=True),
    lambda v: np.mean(v, dtype=np.float32),
    lambda v: v.var(),
    lambda v: v.var(ddof=1),
    lambda v: np.std(v, keepdims=True),
]


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int32, np.int64, np.bool_])
@pytest.mark.parametrize("fn", range(len(_S4_REDUCTIONS)))
@pytest.mark.parametrize("n", [5000, 30_000])
def test_o2_reductions_of_data_sizes(level2, dtype, fn, n):
    """Reductions of an S4 output that cast (int32 sums), take dtype= or
    initial=, and mean/var/std issue without waiting for its size: a cast
    task, the reduction sized in its task, the count of mean/var a task.
    Bit-identical to NumPy (sums over a cast in NumPy's buffer blocks)."""
    f = _S4_REDUCTIONS[fn]
    rng = np.random.default_rng(fn)
    scale = 1 / 64 if fn == 2 else 2**20 if dtype is np.int32 else 2**40
    a = (1 + rng.standard_normal(2 * n) * scale).astype(dtype)
    if dtype is np.int64 and fn != 2:
        a = rng.integers(-2**62, 2**62, 2 * n)  # beyond float64's exact integers
    x = _pending(a)
    keep = rng.random(2 * n) < 0.5
    ref = f(a[keep])
    v = x[viproc.asarray(keep)]  # may wait for x's size (a ready mask: not S4)
    before = _stats()
    r = f(v)
    assert _stats() == before  # no sync, no size wait
    got = _as_numpy(r)
    assert type(got) is type(ref) and got.dtype == ref.dtype
    assert np.shape(got) == np.shape(ref) and np.array_equal(got, ref)


@pytest.mark.parametrize("fn, msg", [
    (lambda v: v.mean(axis=0), "Mean of empty slice"),
    (lambda v: v.var(axis=0, keepdims=True), "Degrees of freedom <= 0 for slice"),
    (lambda v: v.std(axis=0, ddof=3), "Degrees of freedom <= 0 for slice"),
])
def test_o2_moments_of_empty_data_sizes(level2, fn, msg):
    """NumPy's warnings for an empty slice come from the count's task, in
    NumPy's order, at the issue site; the values are NumPy's NaNs."""
    a = np.arange(6.0).reshape(3, 2)
    x = _pending(a)
    with warnings.catch_warnings(record=True) as want:
        warnings.simplefilter("always")
        ref = fn(a[a[:, 0] > 10])
    before = _stats()
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        r = np.asarray(fn(x[x[:, 0] > 10]))
    assert _stats()[1] == before[1]  # no size wait
    assert [str(w.message) for w in got] == [str(w.message) for w in want]
    assert [w.filename for w in got][0] == __file__
    assert r.dtype == ref.dtype and r.shape == ref.shape and np.array_equal(r, ref, equal_nan=True)


def test_o2_reductions_of_data_sizes_async(level2):
    """The cast and the reduction are tasks, no fallback, no wait."""
    keep = np.arange(SYNC_BELOW * 4) % 3 == 0
    a = np.arange(SYNC_BELOW * 4, dtype=np.int32)

    def v():  # the gather of a pending mask: it sizes its result from the data (S4)
        return _pending(a)[_pending(keep)]
    for fn, ops in ((lambda r: r.sum(), 2),  # cast, sum
                    (lambda r: r.mean(axis=0), 4),  # count, cast, sum, divide
                    # two counts, cast, sum, divide, subtract, square, sum, divide
                    (lambda r: r.var(), 9)):
        r = v()
        assert not viproc._viproc.sized(r)
        before = _stats()
        assert_async(fn, r, ops=ops)
        assert _stats() == before


def test_o2_data_dependent_sizes_of_deferred_views(level2):
    """S4 tasks reading a view a rule derives (its base still sized by its
    task) take the view's strides, not only its shape."""
    a = np.asarray(np.random.default_rng(6).integers(-2, 3, (300, 40)), dtype=np.float64)
    x = _pending(a)
    nz = np.nonzero(x[:, 5])[0]
    flat = np.flatnonzero(x[::2, 3:7])
    sel = x[:, 1:4][x[:, 1:4] > 0]
    for got, ref in ((nz, np.nonzero(a[:, 5])[0]), (flat, np.flatnonzero(a[::2, 3:7])),
                     (sel, a[:, 1:4][a[:, 1:4] > 0])):
        assert np.array_equal(np.asarray(got), ref)


def test_o2_mask_mismatch(level2):
    x = _pending(np.ones((30, 4)))
    with pytest.raises(viproc.AsyncIndexError, match="boolean index did not match indexed array "
                       "along axis 0; size of axis is 30 but size of corresponding boolean axis is 20"):
        r = x[_pending(np.ones(20)) > 0]
        r.shape


@pytest.mark.parametrize("order", ["C", "F"])
def test_o2_views_of_deferred_arrays(level2, order):
    """O2: basic indexing and transposes of arrays whose task still sizes
    them are views by a rule (pipelining.md, "Views at O2"): their shapes and
    strides follow when the base's are known, without a wait."""
    rng = np.random.default_rng(8)
    a = rng.random((300, 200))
    a = np.asfortranarray(a) if order == "F" else a
    keys = [lambda t: t[:, 3], lambda t: t[1:-1:2, ::-3], lambda t: t[None, ..., 5],
            lambda t: t.T, lambda t: t.T[2:5], lambda t: t[-1], lambda t: t[..., None, 1:4][3],
            lambda t: np.transpose(t, (1, 0))[::2], lambda t: t[500:], lambda t: t[2][::-1]]
    slow = _slow()[:1] * 0
    before = _stats()
    got = [np.cos(f(np.sin(viproc.asarray(a) + slow))).sum(axis=0) for f in keys]
    views = [f(np.sin(viproc.asarray(a) + slow)) for f in keys]
    assert _stats() == before  # neither a sync nor a size wait
    w = np.sin(a + np.zeros(1))
    for g, v, f in zip(got, views, keys):
        want, r, rv = np.cos(f(w)).sum(axis=0), np.asarray(g), np.asarray(v)
        assert r.shape == want.shape and np.array_equal(r, want)
        assert rv.shape == f(w).shape and rv.strides == f(w).strides
        assert np.array_equal(rv, f(w))


def test_o2_reshape_of_deferred_1d(level2):
    """O2: a 1-d array always reshapes into a view (_attempt_nocopy_reshape),
    so one its task still sizes is reshaped by a rule, without a wait; a
    size mismatch is the view's asynchronous ValueError."""
    a = np.random.default_rng(9).random(240)
    calls = [lambda t: t.reshape(12, 20), lambda t: t.reshape((4, -1, 3)),
             lambda t: t[::-2].reshape(4, 30, order="F"), lambda t: np.reshape(t, (2, -1), order="A"),
             lambda t: t[::3].reshape(-1), lambda t: t.reshape(240, 1, 1).T,
             lambda t: t.reshape(6, 40)[1:3, ::4]]
    slow = _slow()[:1] * 0
    before = _stats()
    got = [f(np.sin(viproc.asarray(a) + slow)) for f in calls]
    assert _stats() == before  # neither a sync nor a size wait
    w = np.sin(a + np.zeros(1))
    for g, f in zip(got, calls):
        r, want = np.asarray(g), f(w)
        assert r.shape == want.shape and r.strides == want.strides and np.array_equal(r, want)
    y = np.sin(viproc.asarray(a) + _slow()[:1] * 0)
    v = y.reshape(12, 20)
    v[0, 0] = 5.0  # a view: writes through to its base
    assert v.base is y and float(y[0]) == 5.0
    with pytest.raises(viproc.AsyncValueError,
                       match=r"cannot reshape array of size 240 into shape \(4,20\)"):
        r = np.sin(viproc.asarray(a) + _slow()[:1] * 0).reshape(4, 20) + 1.0
        np.asarray(r)
    viproc.wait_all()  # raised once


@pytest.mark.parametrize("order", ["C", "F"])
def test_o2_casts_in_tasks(level2, order):
    """O2: ufuncs whose loop casts an input and astype size, lay out and cast
    in their tasks (pipelining.md, "Casts at O2"): no wait on the main
    thread, NumPy's values, dtypes and strides."""
    rng = np.random.default_rng(11)
    x = np.asfortranarray(rng.random((60, 50))) if order == "F" else rng.random((60, 50))
    i = (x * 100).astype(np.int32)
    calls = [lambda a, b: b + a, lambda a, b: (a > 0.5) * a.astype(np.float32),
             lambda a, b: a.astype(np.float32) + 1.5, lambda a, b: b / (b + 1),
             lambda a, b: b + a[0], lambda a, b: a.astype(np.int32), lambda a, b: a.astype(np.int64, order="F"),
             lambda a, b: a.T.astype(np.float32, order="C"), lambda a, b: a[::2, ::-3].astype(np.complex64),
             lambda a, b: b.astype(np.bool_), lambda a, b: a.astype(np.float64)]
    slow = _slow()[:1] * 0
    before = _stats()
    va = np.sin(viproc.asarray(x) + slow)
    vb = (viproc.asarray(i) + slow.astype(np.int32))
    got = [f(va, vb) for f in calls]
    assert _stats() == before  # neither a sync nor a size wait
    wa = np.sin(x + np.zeros(1))
    for g, f in zip(got, calls):
        r, want = np.asarray(g), f(wa, i)
        assert r.dtype == want.dtype and r.strides == want.strides and np.array_equal(r, want)


def test_o2_cast_empty_and_warnings(level2):
    """O2 astype: empty arrays get NumPy's all-zero strides; NaN to int warns
    as NumPy does (at the result's sync point)."""
    for a in [np.zeros((0, 3)), np.zeros((3, 0, 2)).transpose(1, 0, 2)]:
        r = np.asarray((viproc.asarray(a) + _slow()[:1] * 0).astype(np.int32))
        want = a.astype(np.int32)
        assert r.shape == want.shape and r.strides == want.strides
    a = np.array([np.nan, 1e300, 2.5] * 1000)
    with pytest.warns(RuntimeWarning, match="invalid value encountered in cast"):
        r = np.asarray((viproc.asarray(a) + _slow()[:1] * 0).astype(np.int32))
        viproc.wait_all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.array_equal(r, a.astype(np.int32))


@pytest.mark.parametrize("dtype", [np.float64, np.complex64, np.int64])
def test_o2_matmul_in_tasks(level2, dtype):
    """O2: a matmul of arrays tasks still size is sized and laid out in its
    task (MatmulKernel::size/layout): no wait, NumPy's values and strides
    (also a @ a.T on the swapaxes view, NumPy's syrk)."""
    rng = np.random.default_rng(5)
    shapes = [((30, 40), (40, 20)), ((40,), (40, 20)), ((30, 40), (40,)), ((40,), (40,)),
              ((5, 30, 40), (40, 20)), ((30, 40), (3, 1, 40, 20))]
    data = [(rng.random(sa) * 10).astype(dtype) for sa, _ in shapes]
    datb = [(rng.random(sb) * 10).astype(dtype) for _, sb in shapes]
    slow = _slow()[:1] * 0
    before = _stats()
    va = [viproc.asarray(a) + slow.astype(dtype) for a in data]
    vb = [viproc.asarray(b) + slow.astype(dtype) for b in datb]
    got = [x @ y for x, y in zip(va, vb)] + [va[4] @ va[4].swapaxes(1, 2)]
    assert _stats() == before  # neither a sync nor a size wait
    want = [a @ b for a, b in zip(data, datb)] + [data[4] @ data[4].swapaxes(1, 2)]
    for g, w in zip(got, want):
        r = np.asarray(g)
        assert r.dtype == w.dtype and r.shape == w.shape and np.array_equal(r, w)
        assert r.strides == np.asarray(w).strides


@pytest.mark.parametrize("sa, sb, msg", [
    ((3, 4), (5, 6), r"matmul: Input operand 1 has a mismatch in its core dimension 0, with "
                     r"gufunc signature \(n\?,k\),\(k,m\?\)->\(n\?,m\?\) \(size 5 is "
                     r"different from 4\)"),
    ((2, 3, 4), (3, 4, 5), r"operands could not be broadcast together with remapped shapes "
                           r"\[original->remapped\]: \(2,3,4\)->\(2,newaxis,newaxis\) "
                           r"\(3,4,5\)->\(3,newaxis,newaxis\)  and requested shape \(3,5\)"),
])
def test_o2_matmul_errors_are_asynchronous(level2, sa, sb, msg):
    with pytest.raises(ValueError, match=msg):
        np.ones(sa) @ np.ones(sb)
    with pytest.raises(viproc.AsyncValueError, match=msg):
        r = (viproc.asarray(np.ones(sa)) + _slow()[:1] * 0) @ (viproc.asarray(np.ones(sb)) + 0.0)
        np.asarray(r)
    viproc.wait_all()  # raised once


@pytest.mark.parametrize("src", ["C", "F", "T", "one"])
def test_o2_gather_in_tasks(level2, src):
    """O2: advanced indexing of pending arrays is placed by the ranks at
    issue and sized, bounds-checked and laid out in the gather task
    (GatherKernel::size/layout): no wait, NumPy's values and strides."""
    rng = np.random.default_rng(8)
    i, j, k1 = np.array([0, 3, 2, 1]), np.array([[0, 1], [1, 2], [2, 0]]), np.array([[1], [0]])
    full = slice(None)
    keys = [(i,), (full, i), (full, full, i), (i, i), (full, i, i), (i, full, i), (j,), (full, j),
            (slice(1, None), i), (i, slice(None, None, -1)), (k1, full, i),
            (full, np.array([[1]]), full)]
    for key in keys:
        x = {"C": lambda: rng.random((4, 5, 6)),
             "F": lambda: np.asfortranarray(rng.random((4, 5, 6))),
             "T": lambda: rng.random((6, 4, 5)).transpose(1, 2, 0),
             "one": lambda: rng.random((4, 1, 6))}[src]()
        try:
            want = x[key]
        except IndexError:
            continue
        slow = _slow()[:1] * 0
        if src == "T":
            vx = (viproc.asarray(np.ascontiguousarray(x.transpose(2, 0, 1))) + slow).transpose(1, 2, 0)
        else:
            vx = viproc.asarray(x) + slow
        before = _stats()
        r = vx[key]
        assert _stats() == before  # neither a sync nor a size wait
        g = np.asarray(r)
        assert g.shape == want.shape and g.strides == want.strides and np.array_equal(g, want)


def test_o2_gather_errors_are_asynchronous(level2):
    with pytest.raises(viproc.AsyncIndexError, match=r"shape mismatch: indexing arrays could "
                       r"not be broadcast together with shapes \(3,\) \(2,\) "):
        r = (viproc.asarray(np.ones((4, 5))) + _slow()[:1] * 0)[np.array([0, 1, 2]),
                                                                np.array([0, 1])]
        np.asarray(r)
    viproc.wait_all()
    with pytest.raises(viproc.AsyncIndexError, match="index 5 is out of bounds for axis 0 with "
                       "size 4"):
        r = (viproc.asarray(np.ones((4, 5))) + _slow()[:1] * 0)[np.array([5])]
        np.asarray(r)
    viproc.wait_all()


def test_o2_view_index_error_is_asynchronous(level2):
    x = _pending(np.ones((300, 4)))
    with pytest.raises(viproc.AsyncIndexError,
                       match="index 400 is out of bounds for axis 0 with size 300") as e:
        r = x[400] + 1.0  # issued: the shape is not known yet
        np.asarray(r)
    assert isinstance(e.value, IndexError) and "in indexing" in str(e.value)
    viproc.wait_all()  # raised once


def test_pending_sizes_sync_at_o1():
    viproc.init()
    n = _pending_int(4)
    z = viproc.zeros(n)  # NumPy's zeros: operator.index syncs on n
    assert z.shape == (4,) and viproc.offload_stats()["size_waits"] == 0


@pytest.mark.parametrize("args, level", [
    ([], "1"), (["-O2"], "2"), (["-O", "3"], "3"), (["--opt=O5"], "5"), (["--opt", "2"], "2"),
])
def test_main_option(args, level, tmp_path):
    import os
    import subprocess
    import sys
    prog = tmp_path / "prog.py"
    prog.write_text("import numpy as np, viproc\nprint(viproc.opt_level(), np.ones(3).sum())\n")
    env = dict(os.environ)
    env.pop("VIPROC_OPT", None)
    out = subprocess.run([sys.executable, "-m", "viproc", *args, str(prog)], env=env,
                         capture_output=True, text=True, check=True).stdout
    assert out.split() == [level, "3.0"]


def test_main_option_invalid(tmp_path):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "viproc", "-O7", "x.py"], capture_output=True,
                       text=True)
    assert r.returncode != 0 and "1-5" in r.stderr
