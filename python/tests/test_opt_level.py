"""Optimization levels (docs/design/goals.md): O1 is the default and
bit-identical to NumPy; O2 runs every BLAS call on one thread and no longer
serializes LAPACK calls, with results that are correct but may deviate."""

import numpy as np
import pytest

import viproc
from viproc import _blas, _linalg, _ufuncs

from async_path import SYNC_BELOW, assert_async, pending


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
