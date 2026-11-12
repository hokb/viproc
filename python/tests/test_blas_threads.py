"""Dynamic BLAS threads (docs/design/goals.md, "BLAS threads";
src/kernels/blas_threads.cpp): a BLAS call alone runs with OpenBLAS's own
thread count, as in plain NumPy; concurrent calls run on one thread each."""

import time

import numpy as np
import pytest

import viproc
from viproc import _blas


def _counts():
    viproc.shutdown()
    before = _blas.threads()
    viproc.init()
    if before is None:
        pytest.skip("no BLAS library with a thread setting found")
    if before < 2:
        pytest.skip("OpenBLAS runs one thread here: nothing to switch")
    time.sleep(0.01)  # concurrent BLAS calls of earlier tests hold one thread for 5 ms
    return before


def _numpy_with(funcs, n, f):
    """f() computed by NumPy with n BLAS threads (viproc idle), count restored."""
    viproc.wait_all()
    set_, get = funcs
    old = get()
    set_(n)
    try:
        return f()
    finally:
        set_(old)


def test_alone_uses_openblas_threads():
    n = _counts()
    assert _blas.mode() == "dynamic"
    rng = np.random.default_rng(0)
    a = rng.random((600, 601))
    b = rng.random((601, 600))
    got = viproc.asarray(a) @ viproc.asarray(b)
    got.wait()
    assert _blas.threads() == n
    assert np.array_equal(np.asarray(got), a @ b)  # NumPy with its own thread count


def test_concurrent_use_one_thread():
    n = _counts()
    funcs = _blas._get_funcs()
    rng = np.random.default_rng(1)
    ms = [rng.random((400, 400)) for _ in range(8)]
    one = [_numpy_with(funcs, 1, lambda m=m: m @ m) for m in ms]
    many = [_numpy_with(funcs, n, lambda m=m: m @ m) for m in ms]
    _, changes0 = _blas.stats()
    vs = [viproc.asarray(m) for m in ms]
    got = [v @ v for v in vs]
    viproc.wait_all()
    ones = 0
    for g, e1, en in zip(got, one, many):
        g = np.asarray(g)
        assert np.array_equal(g, e1) or np.array_equal(g, en)
        ones += np.array_equal(g, e1) and not np.array_equal(e1, en)
    # with 8 products in flight most of them overlap: one thread each
    assert ones >= 1 or all(np.array_equal(e1, en) for e1, en in zip(one, many))
    assert _blas.stats()[1] >= changes0


def test_plain_numpy_after_idle_gets_threads_back():
    n = _counts()
    rng = np.random.default_rng(2)
    vs = [viproc.asarray(rng.random((300, 300))) for _ in range(6)]
    for r in [v @ v for v in vs]:
        r.wait()
    viproc.wait_all()
    time.sleep(0.01)  # one thread is held 5 ms after the last concurrent call
    # the next call alone raises the count again
    (vs[0] @ vs[1]).wait()
    assert _blas.threads() == n


def test_single_mode_env(monkeypatch):
    _counts()
    monkeypatch.setenv("VIPROC_BLAS_THREADS", "1")
    try:
        viproc.init()
        assert _blas.single() and _blas.threads() == 1
        a = np.random.default_rng(3).random((500, 500))
        (viproc.asarray(a) @ viproc.asarray(a)).wait()
        assert _blas.threads() == 1
    finally:
        monkeypatch.delenv("VIPROC_BLAS_THREADS")
        viproc.init()


def test_bad_env(monkeypatch):
    monkeypatch.setenv("VIPROC_BLAS_THREADS", "7")
    try:
        with pytest.raises(ValueError, match="VIPROC_BLAS_THREADS"):
            viproc.init()
    finally:
        monkeypatch.delenv("VIPROC_BLAS_THREADS")
        viproc.init()
