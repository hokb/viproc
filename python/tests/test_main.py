"""python -m numpy2x, NUMPY2X=1, numpy2x.random and numpy2x.disabled(): NumPy
programs run unchanged, user code gets numpy2x, libraries keep NumPy, results
are bit-identical to plain NumPy."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import numpy2x


def run(args, tmp_path, env=None):
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    child_env.pop("NUMPY2X", None)
    child_env.update(env or {})
    return subprocess.run([sys.executable, *args], cwd=tmp_path, env=child_env,
                          capture_output=True, text=True, timeout=120)


PROGRAM = """\
import importlib
import sys

sys.path.insert(0, sys.path[0] + "/site-packages")
import numpy as np
import numpy.linalg as la
from numpy.random import default_rng

import libx

real = importlib.import_module("numpy")  # the real NumPy, for the reference


def program(np, rng):
    a = rng.standard_normal((300, 400))
    b = np.random.RandomState(1).rand(300, 400)
    c = np.sin(a) * b + np.arange(400) / 7.0
    d = np.zeros((300, 400)) + np.linspace(0, 1, 400)
    e = np.exp(-c * c) - d
    return [e, e.sum(axis=0), e.mean(), c.max(axis=1), np.where(e > 0, e, -e)[::3]]


got = program(np, default_rng(0))
expected = program(real, real.random.default_rng(0))
assert all(real.array_equal(real.asarray(g), x, equal_nan=True) for g, x in zip(got, expected))
print(np.__name__, type(got[0]).__name__, libx.kind(), la.__name__, sys.argv[1:])
"""


@pytest.fixture
def program(tmp_path):
    (tmp_path / "site-packages").mkdir()
    (tmp_path / "site-packages" / "libx.py").write_text(
        "import numpy\n\ndef kind():\n    return type(numpy.zeros(1)).__module__\n")
    (tmp_path / "prog.py").write_text(PROGRAM)
    return tmp_path


def test_python_m_numpy2x_runs_a_numpy_program_with_numpy2x(program):
    r = run(["-m", "numpy2x", "prog.py", "x", "--y"], program)
    assert r.returncode == 0, r.stderr
    # user code: numpy2x; a module in site-packages: NumPy; numpy.linalg: NumPy's
    assert r.stdout.split() == ["numpy2x", "ndarray", "numpy", "numpy.linalg", "['x',", "'--y']"]


def test_the_same_program_without_numpy2x_is_plain_numpy(program):
    r = run(["prog.py"], program)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[:3] == ["numpy", "ndarray", "numpy"]


def test_numpy2x_env_variable_enables_it_at_startup(program):
    # numpy2x.pth exists only in a pip installation; the same call by hand:
    r = run(["-c", "import _numpy2x_hook; _numpy2x_hook.autoload(); import numpy; "
             "print(numpy.__name__)"], program, env={"NUMPY2X": "1"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["numpy2x"]
    r = run(["-c", "import _numpy2x_hook; _numpy2x_hook.autoload(); import numpy; "
             "print(numpy.__name__)"], program)
    assert r.stdout.split() == ["numpy"]


def test_python_m_numpy2x_module_and_command(program):
    (program / "mod.py").write_text("import numpy as np, sys\nprint(np.__name__, sys.argv[1:])\n")
    r = run(["-m", "numpy2x", "-m", "mod", "a"], program)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["numpy2x", "['a']"]
    r = run(["-m", "numpy2x", "-c", "import numpy, sys; print(numpy.__name__, sys.argv)", "b"],
            program)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["numpy2x", "['-c',", "'b']"]


def test_python_m_numpy2x_keeps_exit_codes_and_errors(program):
    (program / "fail.py").write_text("import numpy as np\nnp.zeros(3)\nraise SystemExit(3)\n")
    assert run(["-m", "numpy2x", "fail.py"], program).returncode == 3
    (program / "boom.py").write_text("raise ValueError('boom')\n")
    r = run(["-m", "numpy2x", "boom.py"], program)
    assert r.returncode == 1 and "ValueError: boom" in r.stderr


def test_random_returns_numpy2x_arrays_with_numpy_values():
    for make in (lambda m: m.default_rng(5), lambda m: m.RandomState(5),
                 lambda m: m.Generator(m.PCG64(5))):
        got, expected = make(numpy2x.random), make(np.random)
        for draw in (lambda g: g.random(1000) if hasattr(g, "random") else g.rand(1000),
                     lambda g: g.normal(1.0, 2.0, size=(10, 20)),
                     lambda g: g.integers(0, 9, 50) if hasattr(g, "integers")
                     else g.randint(0, 9, 50)):
            x = draw(got)
            assert isinstance(x, numpy2x.ndarray)
            assert np.array_equal(np.asarray(x), draw(expected))
    numpy2x.random.seed(7)
    x = numpy2x.random.rand(5)
    np.random.seed(7)
    assert isinstance(x, numpy2x.ndarray) and np.array_equal(np.asarray(x), np.random.rand(5))
    assert isinstance(numpy2x.random.default_rng(0).random(), float)  # scalars stay scalars
    out = np.empty(4)
    assert numpy2x.random.default_rng(0).random(4, out=out) is out  # out= is the caller's


def test_disabled_gives_numpy_arrays_and_numpy_results():
    def program(m):
        a = m.random.default_rng(3).standard_normal(5000)
        return m.asarray(np.sin(a) * 2.0 + m.arange(5000) / 3.0)

    with numpy2x.disabled():
        assert not numpy2x.enabled()
        plain = program(numpy2x)
        assert type(plain) is np.ndarray
        assert type(numpy2x.zeros(3)) is np.ndarray and type(numpy2x.array([1, 2])) is np.ndarray
    assert numpy2x.enabled()
    accelerated = program(numpy2x)
    assert isinstance(accelerated, numpy2x.ndarray)
    assert np.array_equal(np.asarray(accelerated), plain)
    numpy2x.enable(False)
    try:
        assert type(numpy2x.ones(2)) is np.ndarray
    finally:
        numpy2x.enable(True)


def test_array_and_creation_functions_follow_numpys_signatures():
    a = numpy2x.array([1.0, 2.0], ndmin=3)
    assert isinstance(a, numpy2x.ndarray) and a.shape == (1, 1, 2)
    v = numpy2x.asarray(np.arange(4.0))
    assert numpy2x.asarray(v) is v and numpy2x.array(v, copy=False) is v
    c = numpy2x.array(v)
    assert c is not v and np.array_equal(np.asarray(c), np.arange(4.0))
    assert numpy2x.array(v, dtype=np.float32).dtype == np.float32
    # dtypes numpy2x does not run asynchronously stay NumPy arrays, as NumPy makes them
    o = numpy2x.array([1, "a"], dtype=object)
    assert type(o) is np.ndarray and o.dtype == object
    assert type(numpy2x.zeros(3, dtype=np.float16)) is np.ndarray
    buf = bytearray(np.arange(3.0).tobytes())
    f = numpy2x.frombuffer(buf)  # a view of the caller's buffer: copied
    buf[:8] = np.float64(9.0).tobytes()
    assert isinstance(f, numpy2x.ndarray) and np.array_equal(np.asarray(f), [0.0, 1.0, 2.0])
    x, step = numpy2x.linspace(0, 1, 5, retstep=True)
    assert isinstance(x, numpy2x.ndarray) and step == 0.25
    assert isinstance(numpy2x.identity(3), numpy2x.ndarray)


def test_user_code_detection():
    import _numpy2x_hook

    assert _numpy2x_hook.is_user_code({"__name__": "__main__"})
    assert _numpy2x_hook.is_user_code({"__name__": "mod", "__file__": __file__})
    assert not _numpy2x_hook.is_user_code({"__name__": "numpy2x._ndarray", "__file__": __file__})
    assert not _numpy2x_hook.is_user_code({"__name__": "json", "__file__": textwrap.__file__})
    assert not _numpy2x_hook.is_user_code({"__name__": "numpy.linalg", "__file__": np.__file__})
    assert not _numpy2x_hook.is_user_code(None)
