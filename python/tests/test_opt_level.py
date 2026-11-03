"""Optimization levels (docs/design/goals.md): O1 is the default and
bit-identical to NumPy; O2 runs every BLAS call on one thread and no longer
serializes LAPACK calls, with results that are correct but may deviate."""

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
