"""The numpy.ndarray operator hook (viproc.enable_hook(), python -m viproc):
operators of user code on numpy.ndarrays give NumPy's
results, bit for bit; large results become viproc arrays, small ones stay
NumPy's; operands NumPy would defer to, ndarray subclasses and library code
keep NumPy's own operators. The hook is installed at import, so each check
runs in a child process."""

import os
import subprocess
import sys
import textwrap

import pytest


def run_child(code, tmp_path, module_mode=False, header="", enable=True):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env.pop("VIPROC_HOST_SMALL", None)
    env.pop("VIPROC_NDARRAY_HOOK", None)
    env.pop("VIPROC", None)
    prog = tmp_path / "prog.py"
    enable = "" if module_mode or not enable else "import viproc\nviproc.enable_hook()\n"
    prog.write_text(enable + header + textwrap.dedent(code))
    args = ["-m", "viproc", str(prog)] if module_mode else [str(prog)]
    r = subprocess.run([sys.executable, *args], cwd=tmp_path, env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


HEADER = """\
import numpy as np
import viproc

def bits(x):
    x = np.asarray(x)
    return x.dtype, x.shape, np.atleast_1d(x).view(np.uint8).tobytes()

def check(f, *args):
    with viproc.disabled(), np.errstate(all="ignore"):
        try:
            expected, error = f(*(a.copy() for a in args)), None
        except Exception as e:
            expected, error = None, type(e)
    with np.errstate(all="ignore"):
        try:
            got = f(*(a.copy() for a in args))
            got = got if np.isscalar(got) else np.asarray(got)  # sync
        except Exception as e:
            assert error is not None and isinstance(e, error), (f, e, error)
            return None
    assert error is None, (f, error)
    assert bits(got) == bits(expected), f
    return got
"""


def test_operators_are_bit_identical_to_numpy(tmp_path):
    out = run_child("""
    ops = [
        lambda a: a + a, lambda a: a - 1, lambda a: 2 * a, lambda a: a / 3, lambda a: a // 2,
        lambda a: a % 3, lambda a: a & a, lambda a: a | a, lambda a: a ^ a, lambda a: a << 1,
        lambda a: a >> 1, lambda a: a < 1, lambda a: a <= 1, lambda a: a == a,
        lambda a: a != 1, lambda a: a > 1, lambda a: a >= 1,
        lambda a: -a, lambda a: +a, lambda a: abs(a), lambda a: ~a,
        lambda a: a ** 2, lambda a: a ** -1, lambda a: a ** 0.5, lambda a: a ** 3,
        lambda a: a ** 2.0, lambda a: 2 ** a, lambda a: a @ a.T, lambda a: a @ a[0],
        lambda a: a.T @ a, lambda a: a[::2] @ a[::2].T, lambda a: a[:, ::-1] + a,
    ]
    for dt in ["?", "i4", "i8", "f4", "f8", "c8", "c16"]:
        for shape in [(3, 4), (100, 60)]:
            base = np.resize(np.array([-0.0, -np.inf, 3, 1e30, 0, 2, -1.5, np.nan]), shape)
            with np.errstate(all="ignore"):
                a = base.astype(dt)
            for f in ops:
                check(f, a)
    assert viproc._viproc.hooked_count() > 0
    print("ok")
    """, tmp_path, header=HEADER)
    assert out.strip() == "ok"


def test_large_results_are_viproc_arrays_small_ones_numpys(tmp_path):
    out = run_child("""
    import numpy as np
    import viproc
    a, b = np.ones((100, 50)), np.ones((50, 100))
    for i, r in enumerate((a + 1, -a, a ** 2, a @ b, a > 0, a % 2)):
        assert isinstance(r, viproc.ndarray), (i, type(r))
    s = a[:2, :2]
    for r in (s + 1, -s, s ** 2, s @ s, s > 0):
        assert type(r) is np.ndarray, type(r)
    assert type(a @ a.T) is np.ndarray  # one buffer: NumPy's syrk, not two imports
    with viproc.disabled():
        assert type(a + 1) is np.ndarray
    b = a.copy()
    c = b
    b += viproc.asarray(a)  # in place on an ndarray another name sees: NumPy's
    assert type(b) is np.ndarray and b is c and b[0, 0] == 2
    b = a.copy()
    b += viproc.asarray(a)  # only the name sees it: viproc takes b's memory over
    assert isinstance(b, viproc.ndarray) and b[0, 0] == 2
    print("ok")
    """, tmp_path)
    assert out.strip() == "ok"


def test_operands_numpy_defers_to_and_subclasses_keep_numpys_operators(tmp_path):
    out = run_child("""
    import fractions
    import numpy as np
    import viproc
    a = np.ones((100, 50))

    class Sub(np.ndarray):
        pass

    s = a.view(Sub)
    for r in (a + s, s + a, -s, s ** 2, s * 2):
        assert type(r) is Sub, type(r)

    class Defers:
        __array_ufunc__ = None

        def __radd__(self, other):
            return "deferred"

        def __rmatmul__(self, other):
            return "deferred@"

    assert a + Defers() == "deferred" and a @ Defers() == "deferred@"

    class Priority:
        __array_priority__ = 100

        def __radd__(self, other):
            return "priority"

    assert a + Priority() == "priority"
    m = np.matrix([[1.0, 2.0], [3.0, 4.0]])
    assert type(m * np.ones((2, 2))) is np.matrix
    objs = np.array([fractions.Fraction(1, 2)] * 3000)
    assert (objs + 1)[0] == fractions.Fraction(3, 2)
    assert type(a + [1.0] * 50) is np.ndarray
    print("ok")
    """, tmp_path)
    assert out.strip() == "ok"


def test_library_code_keeps_numpys_operators(tmp_path):
    site = tmp_path / "site-packages"
    site.mkdir()
    (site / "libx.py").write_text("def f(a):\n    return a + 1, -a, a ** 2, a @ a.T\n")
    out = run_child("""
    import sys
    sys.path.insert(0, sys.path[0] + "/site-packages")
    import numpy as np
    import viproc
    import libx
    a = np.ones((100, 50))
    assert all(type(r) is np.ndarray for r in libx.f(a))
    assert isinstance(a + 1, viproc.ndarray)
    print("ok")
    """, tmp_path)
    assert out.strip() == "ok"


def test_module_mode_ufuncs_take_ndarrays(tmp_path):
    out = run_child("""
    import numpy as np
    import viproc
    with viproc.disabled():
        a = np.random.default_rng(0).random((200, 100))
        expected = np.sin(a), np.add(a, 1), np.sin(a[:2, :2])
    big, add, small = np.sin(a), np.add(a, 1), np.sin(a[:2, :2])
    assert isinstance(big, viproc.ndarray) and isinstance(add, viproc.ndarray)
    assert not isinstance(small, viproc.ndarray)
    assert viproc.hook_enabled() and isinstance(a * 2, viproc.ndarray)
    for g, e in zip((big, add, small), expected):
        assert np.array_equal(np.asarray(g), e)
    print("ok")
    """, tmp_path, module_mode=True)
    assert out.strip() == "ok"


def test_import_alone_does_not_hook(tmp_path):
    out = run_child("""
    import numpy as np
    import viproc
    a = np.ones((100, 50))
    assert not viproc.hook_enabled() and type(a + 1) is np.ndarray
    viproc.enable_hook()
    assert isinstance(a + 1, viproc.ndarray)
    with viproc.disabled():
        assert type(a + 1) is np.ndarray
    assert isinstance(a + 1, viproc.ndarray)
    viproc.enable_hook(False)
    assert type(a + 1) is np.ndarray
    with viproc.disabled():
        pass
    assert type(a + 1) is np.ndarray  # leaving disabled() keeps the hook off
    print("ok")
    """, tmp_path, enable=False)
    assert out.strip() == "ok"


def test_ndarray_operands_are_borrowed_only_where_nobody_can_write_them(tmp_path):
    out = run_child("""
    import numpy as np
    import viproc
    count = viproc._viproc.borrowed_count
    x = np.random.default_rng(0).random((600, 600))
    expected = x * 3

    def borrowed(f):
        before = count()
        r = f()
        np.asarray(r)
        return count() - before

    assert borrowed(lambda: x.copy() * 3) == 1  # a temporary that owns its memory
    e = np.asarray(viproc.asarray(x))  # a viproc export: a writable alias, as in NumPy
    assert borrowed(lambda: e * 3) == 0
    b = np.frombuffer(x.tobytes()).reshape(600, 600)  # read-only over bytes
    assert borrowed(lambda: b[:, ::2] * 3) == 1  # strided: borrowed with its layout
    assert np.array_equal(np.asarray(b[:, ::2] * 3), expected[:, ::2])
    assert borrowed(lambda: x * 3) == 0  # the user may write x later
    ro = x.copy()
    ro.setflags(write=False)  # owns its memory: setflags(write=True) may come back
    assert borrowed(lambda: ro * 3) == 0
    view = x[:]
    view.setflags(write=False)  # a read-only view of writable memory
    assert borrowed(lambda: view * 3) == 0
    # a copied operand keeps the values of the issue
    y = x.copy()
    r = y * 3
    y[...] = 0
    assert np.array_equal(np.asarray(r), expected)
    print("ok")
    """, tmp_path)
    assert out.strip() == "ok"


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="redirection needs strong loads")
def test_in_place_operators_take_over_ndarrays_only_where_a_plain_name_is_rebound(tmp_path):
    out = run_child("""
    import numpy as np
    import viproc
    rebound = viproc._viproc.rebound_count
    N = 100000
    base = np.random.default_rng(0).random(N)

    def taken(f):
        before = rebound()
        r = f()
        return rebound() - before, r

    def local():
        x = base.copy()
        x += 1.5
        x *= base
        x **= 2
        return x
    n, r = taken(local)
    assert n == 1 and isinstance(r, viproc.ndarray)
    assert np.asarray(r).tobytes() == (((base + 1.5) * base) ** 2).tobytes()

    def fortran():
        x = np.asfortranarray(base.reshape(400, 250))
        x -= 1
        return x
    n, r = taken(fortran)
    assert n == 1 and r.strides == (8, 3200)  # NumPy's layout kept

    def closure():
        x = base.copy()
        def g():
            nonlocal x
            x /= 3
        g()
        return x
    assert taken(closure)[0] == 1

    G = base.copy()
    def glob():
        global G
        G //= 0.25
        return G
    assert taken(glob)[0] == 1
    M = base.copy()
    before = rebound()
    M %= 0.3
    assert rebound() - before == 1 and isinstance(M, viproc.ndarray)

    # NumPy's in-place write stays visible where something else sees x
    def other_name():
        x = base.copy()
        y = x
        x += 1
        return y
    def view_of_x():
        x = base.copy()
        v = x[::2]
        x += 1
        return v
    def x_is_a_view():
        big = base.copy()
        x = big[: N // 2]
        x += 1
        return big
    def tuple_item():
        x = base.copy()
        t = (x,)
        try:
            t[0] += 1  # NumPy writes x, then the store raises
        except TypeError:
            pass
        return x
    class O:
        pass
    def attribute():
        o = O()
        o.x = base.copy()
        o.x += 1
        return o.x
    import weakref
    def weak_reference():
        x = base.copy()
        w = weakref.ref(x)
        x += 1
        assert w() is x
        return x
    for f in (other_name, view_of_x, x_is_a_view, tuple_item, attribute, weak_reference):
        n, r = taken(f)
        assert n == 0 and type(r) is np.ndarray and r[0] == base[0] + 1, f
    def small():
        x = base[:10].copy()
        x += 1
        return x
    def strided_operand():
        x = base.copy()
        x += np.stack([base, base], axis=1)[:, 1]
        return x
    def readonly():
        x = base.copy()
        x.setflags(write=False)
        try:
            x += 1
        except ValueError as e:
            return e
    for f in (small, strided_operand, readonly):
        assert taken(f)[0] == 0, f

    def cast_error():  # NumPy's error, x unchanged
        x = np.ones(N, np.int64)
        try:
            x += 1.5
        except Exception as e:
            return type(e).__name__, str(e), type(x) is np.ndarray and x[0] == 1
    with viproc.disabled():
        expected = cast_error()
    assert cast_error() == expected

    def loop(a):
        x = a.copy()
        for _ in range(20):
            x += 1
            x *= 0.5
        return x
    with viproc.disabled():
        expected = loop(base)
    assert np.asarray(loop(base)).tobytes() == expected.tobytes()
    print("ok")
    """, tmp_path)
    assert out.strip() == "ok"
