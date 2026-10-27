"""Python bridge: viproc arrays must behave like NumPy arrays (bit-identical
results), with supported ops running asynchronously."""

import platform
import sys
import time

import numpy as np
import pytest

import viproc

rng = np.random.default_rng(3)


def same(got, expected):
    got = np.asarray(got)
    return got.dtype == expected.dtype and got.shape == expected.shape and np.array_equal(
        got, expected, equal_nan=True
    )


def test_arithmetic_expression_with_broadcasting_and_scalars():
    a, b = rng.standard_normal((40, 30)), rng.standard_normal(30)
    va, vb = viproc.asarray(a), viproc.asarray(b)
    r = np.sin(va * vb + 0.5) - va / 3 + np.float64(2.0)
    assert isinstance(r, viproc.ndarray)
    assert same(r, np.sin(a * b + 0.5) - a / 3 + np.float64(2.0))


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]
)
def test_dtypes(dtype):
    x = (rng.standard_normal(50) * 10).astype(dtype)
    vx = viproc.asarray(x)
    assert same(vx * vx, x * x)
    assert same(vx == vx, x == x)
    assert same(np.logical_not(vx), np.logical_not(x))


def test_python_scalars_follow_nep50():
    x = np.arange(10, dtype=np.int32)
    vx = viproc.asarray(x)
    assert same(vx + 1, x + 1)  # stays int32
    assert same(vx * 2.5, x * 2.5)  # float64
    f = np.linspace(0, 1, 7, dtype=np.float32)
    assert same(viproc.asarray(f) + 1.0, f + 1.0)  # stays float32


def test_mixed_dtypes_fall_back_with_numpy_semantics():
    i, f = np.arange(5, dtype=np.int32), np.linspace(0, 1, 5)
    r = viproc.asarray(i) + viproc.asarray(f)
    assert isinstance(r, viproc.ndarray)
    assert same(r, i + f)


def test_integer_division_and_comparisons():
    a, b = np.arange(-6, 6, dtype=np.int64), np.full(12, 4, dtype=np.int64)
    va, vb = viproc.asarray(a), viproc.asarray(b)
    assert same(va / vb, a / b)
    assert same(va // vb, a // b)
    assert same(va % vb, a % b)
    assert same((va < vb) & (va >= -2), (a < b) & (a >= -2))


def test_inplace_operators():
    a = rng.standard_normal(100)
    va = viproc.asarray(a)
    reader = va * 1.0  # pending reader of the old data
    va += 1
    va *= 2
    va -= va / 4
    expected = a.copy()
    expected += 1
    expected *= 2
    expected -= expected / 4
    assert same(va, expected)
    assert same(reader, a * 1.0)


def test_ufunc_out_argument():
    a, b = rng.standard_normal(20), rng.standard_normal(20)
    va, vb, vc = viproc.asarray(a), viproc.asarray(b), viproc.zeros(20)
    r = np.add(va, vb, out=vc)
    assert r is vc
    assert same(vc, a + b)


def test_basic_indexing_views():
    a = np.arange(24.0).reshape(4, 6)
    va = viproc.asarray(a)
    assert same(va[1:3, ::2], a[1:3, ::2])
    assert same(va[::-1, 2], a[::-1, 2])
    assert same(va[None, 1], a[None, 1])
    assert va[2, 3] == a[2, 3] and type(va[2, 3]) is np.float64
    # views share data with their base
    col = va[:, 1]
    col += 100
    a[:, 1] += 100
    assert same(va, a)


def test_setitem():
    a = np.zeros((3, 4))
    va = viproc.asarray(a)
    va[1] = 5
    a[1] = 5
    va[:, 2] = viproc.asarray(np.arange(3.0))
    a[:, 2] = np.arange(3.0)
    va[0, 0] = 1.75
    a[0, 0] = 1.75
    va[np.array([2, 0]), 3] = -1  # advanced index: fallback
    a[np.array([2, 0]), 3] = -1
    assert same(va, a)


def test_overlapping_inplace_matches_numpy():
    a = np.arange(10.0)
    va = viproc.asarray(a)
    va += va[::-1]
    a += a[::-1].copy()  # NumPy semantics: inputs read before writing
    assert same(va, a)


def test_fallback_functions_and_reductions():
    a = rng.standard_normal((5, 7))
    va = viproc.asarray(a)
    assert np.sum(va) == np.sum(a)
    assert same(np.sum(va, axis=0), np.sum(a, axis=0))
    assert same(va.max(axis=1), a.max(axis=1))
    assert same(np.concatenate([va, va]), np.concatenate([a, a]))
    assert same(va.T @ va, a.T @ a)
    assert isinstance(np.sum(va, axis=0), viproc.ndarray)


def test_main_thread_runs_ahead():
    # Best of three: one stall of the main thread on a shared CI machine
    # (seen once on a Windows runner) must not fail it; a stall on every
    # pool miss (Windows large pages, ~30 ms each) fails every attempt.
    x = viproc.asarray(rng.standard_normal(3_000_000))
    times = []
    for _ in range(3):
        viproc.wait_all()
        t0 = time.perf_counter()
        y = x
        for _ in range(20):
            y = np.sqrt(y * y + 1.0)
        issue_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        y.wait()
        compute_time = time.perf_counter() - t1
        times.append((issue_time, compute_time))
        if issue_time < compute_time:
            break
    assert any(i < c for i, c in times), times


def test_import_viproc_as_np():
    import viproc as vnp

    a = vnp.arange(10.0)
    b = vnp.sin(a) + vnp.ones(10)
    assert isinstance(b, viproc.ndarray)
    assert same(b, np.sin(np.arange(10.0)) + 1)
    assert vnp.float64 is np.float64


# --- operators and the C fast path ------------------------------------------

OPS = [
    lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b, lambda a, b: a / b,
    lambda a, b: a // b, lambda a, b: a % b, lambda a, b: a ** b,
    lambda a, b: a < b, lambda a, b: a <= b, lambda a, b: a == b, lambda a, b: a != b,
    lambda a, b: a > b, lambda a, b: a >= b,
]


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.float32])
def test_binary_operators_match_numpy(op, dtype):
    a = (rng.standard_normal(20) * 5 + 10).astype(dtype)
    b = (rng.standard_normal(20) * 2 + 3).astype(dtype)
    va, vb = viproc.asarray(a), viproc.asarray(b)
    with np.errstate(all="ignore"):
        expected = op(a, b)
        assert same(op(va, vb), expected)  # both viproc
        assert same(op(va, b), expected)  # NumPy array on the right
        assert same(op(a, vb), expected)  # NumPy array on the left (reflected)
        assert same(op(va, 2), op(a, 2))  # Python scalar
        assert same(op(3.5, va), op(3.5, a))  # reflected Python scalar
        assert same(op(va, dtype(2)), op(a, dtype(2)))  # NumPy scalar


def test_bitwise_and_unary_operators():
    a = np.arange(-8, 8, dtype=np.int32)
    va = viproc.asarray(a)
    assert same(va & 3, a & 3) and same(va | 4, a | 4) and same(va ^ 5, a ^ 5)
    assert same(va << 2, a << 2) and same(va >> 1, a >> 1)
    assert same(-va, -a) and same(+va, +a) and same(abs(va), abs(a)) and same(~va, ~a)


def test_divmod_matmul_and_pow_with_mod_fall_back():
    a = np.arange(1.0, 10.0)
    va = viproc.asarray(a)
    q, r = divmod(va, 4.0)
    assert same(q, a // 4.0) and same(r, a % 4.0)
    m = rng.standard_normal((4, 4))
    assert same(viproc.asarray(m) @ viproc.asarray(m), m @ m)
    with pytest.raises(TypeError):
        pow(viproc.asarray(np.arange(3)), 2, 5)


def bits(x):
    x = np.asarray(x)
    return x.dtype, x.shape, np.atleast_1d(x).view(np.uint8).tobytes()


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize("n", [5, 5000])  # direct path and task
def test_power_takes_numpys_fast_scalar_power(dtype, n):
    """`a ** 2` is square(a), `a ** -1` reciprocal(a), `a ** 0.5` sqrt(a) for
    inexact a, as NumPy's array_power: other dtypes (bool ** 2 is int8) and
    other values (complex signs and infinities) than power."""
    with np.errstate(all="ignore"):
        a = np.resize(np.array([-0.0, -np.inf, 3.0, 1e30, 0.0, np.nan, -2.5]), n).astype(dtype)
    exps = [2, 0.5, 3, 2.0, np.float64(2.0), True]
    if np.issubdtype(dtype, np.inexact):
        exps.append(-1)
    for e in exps:
        with np.errstate(all="ignore"):
            expected = a ** e
            assert bits(viproc.asarray(a) ** e) == bits(expected), e
            va, x = viproc.asarray(a), a.copy()
            if np.can_cast(expected.dtype, a.dtype, "same_kind"):
                va **= e
                x **= e
                assert bits(va) == bits(x), e


def test_inplace_operators_all_kinds():
    a = np.arange(1, 13, dtype=np.int64)
    va = viproc.asarray(a)
    for op in ("__iadd__", "__isub__", "__imul__", "__ifloordiv__", "__imod__",
               "__ilshift__", "__irshift__", "__iand__", "__ior__", "__ixor__"):
        getattr(va, op)(3)
        getattr(a, op)(3)
    assert same(va, a)


def test_inplace_cast_error_like_numpy():
    va = viproc.asarray(np.arange(4))
    with pytest.raises(TypeError):  # UFuncTypeError: float64 result into int64
        va /= 2
    with pytest.raises(TypeError):
        np.true_divide(np.arange(4), 2, out=np.arange(4))


def test_python_int_overflow_like_numpy():
    v = viproc.asarray(np.arange(3, dtype=np.int32))
    with pytest.raises(OverflowError):
        v + 2**40
    with pytest.raises(OverflowError):
        np.arange(3, dtype=np.int32) + 2**40


def test_comparison_with_non_array_operands():
    va = viproc.asarray(np.arange(3.0))
    assert same(va == None, np.arange(3.0) == None)  # noqa: E711


def test_classes_opting_out_of_numpy_get_their_reflected_operator():
    class OptOut:
        __array_ufunc__ = None

        def __radd__(self, other):
            return "OptOut.__radd__"

    assert viproc.asarray(np.arange(3.0)) + OptOut() == "OptOut.__radd__"


def test_operator_results_are_pending_until_computed():
    x = viproc.asarray(rng.standard_normal(2_000_000))
    y = x
    for _ in range(5):
        y = np.sqrt(y * y + 1.0)  # issued here; result pending
    assert not y.ready()
    y.wait()
    assert y.ready()


def test_issue_site_names_the_user_code_line():
    from viproc import _viproc

    def caller():
        return _viproc._issue_site()  # the line below must be reported

    site = caller()
    assert site.endswith("in test_issue_site_names_the_user_code_line.<locals>.caller")
    assert "test_bridge.py:" in site
    line = int(site.split("test_bridge.py:")[1].split(" ")[0])
    import inspect
    assert line == inspect.getsourcelines(caller)[1] + 1
    sites = {caller() for _ in range(3)}  # cached: same interned site
    assert sites == {site}


def test_offload_policy_small_ops_inline_and_deterministic_mode():
    try:
        viproc.init(workers=2, sync_below=1024)
        small = viproc.asarray(np.arange(10.0))
        before = viproc.offload_stats()
        r = small * 2
        assert r.ready()  # ran on the main thread: nothing to wait for
        assert viproc.offload_stats()["direct"] == before["direct"] + 1  # without a task
        assert same(r, np.arange(10.0) * 2)

        viproc.init(workers=2, sync_below=0, adaptive=False)  # deterministic: always async
        big = viproc.asarray(rng.standard_normal(1_000_000))
        for _ in range(50):
            (np.sin(big) + 1).wait()  # always waited on right away
        assert viproc.offload_stats()["inline"] == 0
    finally:
        viproc.init()


def test_adaptive_policy_switches_site_waited_on_right_away():
    try:
        viproc.init(workers=2, sync_below=0, adaptive=True)
        x = viproc.asarray(rng.standard_normal(4096))
        for _ in range(64):
            (x * 2).wait()  # same issue site, result needed immediately
        assert viproc.offload_stats()["inline"] > 30
    finally:
        viproc.init()


def test_element_pool_reuses_memory_and_trims():
    viproc.trim()
    x = viproc.asarray(rng.standard_normal(100_000))
    expected = ((np.asarray(x) + 1.0) * 2.0) + 3.0
    before = viproc.pool_stats()
    for _ in range(20):
        r = ((x + 1.0) * 2.0) + 3.0
        assert same(r, expected)
    after = viproc.pool_stats()
    # x + 1 takes a buffer each time; the other two ops may write into the
    # temporary instead (D5, D11). Every block released before the next
    # allocation is found again (one shared LIFO per bin): misses only for
    # the first iteration and blocks still held when the next op allocates.
    misses = after["misses"] - before["misses"]
    assert misses <= 3
    assert after["hits"] - before["hits"] >= 20 - misses
    assert after["cached"] <= after["limit"]
    viproc.trim()
    assert viproc.pool_stats()["cached"] == 0


def test_element_pool_limit_zero_disables_caching():
    limit = viproc.pool_stats()["limit"]
    try:
        viproc.set_pool_limit(0)
        x = viproc.asarray(rng.standard_normal(100_000))
        for _ in range(5):
            (x * 2).wait()
        assert viproc.pool_stats()["cached"] == 0
    finally:
        viproc.set_pool_limit(limit)


_ELISION = viproc._viproc.elision_mode()


def test_temporary_elision_is_available_on_linux_and_windows():
    if sys.version_info >= (3, 14):
        assert _ELISION == "exact"
    elif sys.platform == "win32" or (sys.platform == "linux" and platform.libc_ver()[0] == "glibc"):
        assert _ELISION == "backtrace"
        # every operator calibrated to a chain that reaches the interpreter's
        # call site: 12 binary ones in three forms (p + p, p + 1.0, 1.0 + p:
        # different dispatch paths), 3 unary ones
        assert len(viproc._viproc.elision_chains()) == 39


@pytest.mark.skipif(_ELISION == "off", reason="no safe temporary test on this platform")
def test_temporary_elision_reuses_buffers_and_keeps_results():
    x = viproc.asarray(rng.standard_normal(100_000))  # 800 KB: above every bound
    xn = np.asarray(x)
    e0 = viproc._viproc.elided()
    r = ((x + 1.0) * 2.0) + 3.0  # two ops write into their temporary operand
    assert viproc._viproc.elided() - e0 == 2
    assert same(r, ((xn + 1.0) * 2.0) + 3.0)
    e0 = viproc._viproc.elided()
    r = -(x * 2.0)
    assert viproc._viproc.elided() - e0 == 1
    assert same(r, -(xn * 2.0))


@pytest.mark.skipif(_ELISION == "off", reason="no safe temporary test on this platform")
def test_temporary_elision_never_touches_referenced_arrays():
    import operator

    x = viproc.asarray(rng.standard_normal(100_000))
    xn = np.asarray(x)
    t = x * 2.0  # a named variable is not a temporary
    e0 = viproc._viproc.elided()
    y = t + 1.0
    assert viproc._viproc.elided() == e0
    assert same(t, xn * 2.0) and same(y, xn * 2.0 + 1.0)
    items = [x * 1.0, x * 2.0]  # held by a list; sum() calls the operator from C
    s = sum(items)
    assert viproc._viproc.elided() == e0
    assert same(items[0], xn) and same(s, xn * 1.0 + xn * 2.0)
    w = operator.add(x * 2.0, 1.0)  # called from C, not by the interpreter
    # The backtrace check refuses any C caller. CPython >= 3.14 knows that the
    # temporary's only reference is the calling frame's stack (the argument
    # operator.add borrows), so reusing it is safe there.
    e0 += 1 if _ELISION == "exact" else 0
    assert viproc._viproc.elided() == e0
    assert same(w, xn * 2.0 + 1.0)
    i = viproc.asarray(np.arange(100_000))
    f = (i + 1) / 2  # int64 temporary, float64 result: different dtype
    assert viproc._viproc.elided() == e0
    assert same(f, (np.arange(100_000) + 1) / 2)


@pytest.mark.skipif(_ELISION == "off", reason="no safe temporary test on this platform")
def test_temporary_elision_and_views():
    x = viproc.asarray(rng.standard_normal(200_000))
    xn = np.asarray(x).copy()
    t = x * 2.0
    keep = []

    def stash(a):
        keep.append(a[::2])  # a view outlives the call
        return a

    cases = [
        (lambda: x[:] + 1.0, xn + 1.0, 0),  # view of a named array
        (lambda: x[::2] + 1.0, xn[::2] + 1.0, 0),
        (lambda: t[:] + 1.0, xn * 2.0 + 1.0, 0),
        (lambda: (x * 2.0)[1:] + 1.0, (xn * 2.0)[1:] + 1.0, 0),  # offset: not the whole buffer
        (lambda: (x * 2.0)[::2] + 1.0, (xn * 2.0)[::2] + 1.0, 0),  # strided
        (lambda: stash(x * 2.0) + 1.0, xn * 2.0 + 1.0, 0),  # storage still has a view
        # the view's base keeps the temporary alive (NumPy does not elide a
        # view either: can_elide_temp needs OWNDATA)
        (lambda: (x * 2.0)[:] + 1.0, xn * 2.0 + 1.0, 0),
        (lambda: (x * 2.0) + 1.0, xn * 2.0 + 1.0, 1),  # only array on its storage
    ]
    for f, expected, elisions in cases:
        e0 = viproc._viproc.elided()
        assert same(f(), expected)
        assert viproc._viproc.elided() - e0 == elisions
    assert same(x, xn) and same(t, xn * 2.0) and same(keep[0], (xn * 2.0)[::2])


def test_export_is_copy_free_and_aliases_like_numpy():
    """np.asarray of a NumPy array is the array itself: viproc's export is a
    writable NumPy view of the same memory (D7)."""
    x = viproc.asarray(rng.standard_normal(1000))
    a, b = np.asarray(x), x.numpy()
    assert np.shares_memory(a, b)  # both views of the viproc buffer
    assert a.flags.writeable
    a[0] = 1.0  # writes through the export show in x, as in NumPy
    assert x[0] == 1.0 and b[0] == 1.0
    c = np.array(x)  # explicit copy: writable, independent
    assert c.flags.writeable and not np.shares_memory(c, a)
    assert np.shares_memory(np.asarray(x, copy=False), a)
    assert np.asarray(x, dtype=np.float32).dtype == np.float32


def test_export_aliases_across_in_place_writes():
    for module in (np, viproc):
        x = module.asarray(np.arange(100_000, dtype=np.float64))
        before = np.asarray(x)  # held across in-place writes
        x += 1.0  # runs now: the export sees it, as NumPy's view would
        x[:10] = -5.0
        before[10] = 7.0  # and x sees writes through the export
        expected = np.arange(100_000, dtype=np.float64) + 1.0
        expected[:10] = -5.0
        expected[10] = 7.0
        assert same(before, expected) and same(x, expected)
        y = x * 2.0  # reads x as it is now
        before[11] = 0.0  # after y was computed
        assert same(y, expected * 2.0)
        del before
        x *= 2.0  # no export alive: asynchronous again
        expected[11] = 0.0
        assert same(x, expected * 2.0)


def test_export_view_of_view_and_strides():
    base = viproc.asarray(rng.standard_normal((40, 30)))
    v = base[::2, 1::3]
    a = np.asarray(v)
    assert np.shares_memory(a, np.asarray(base))
    assert same(a, np.asarray(base)[::2, 1::3])


def test_creation_functions_and_fallback_results_are_adopted():
    z = viproc.zeros(1000)
    assert same(z, np.zeros(1000))
    user = np.ones(5)
    r = np.atleast_1d(viproc.asarray(user))  # fallback: returns a copy-free export view
    assert same(r, user)
    user[0] = 7.0  # user's array: never shared with viproc (asarray copied)
    assert same(r, np.ones(5))


def test_fallback_out_argument_is_written_back():
    x = viproc.asarray(rng.standard_normal((4, 5)))
    out = viproc.zeros(5)
    np.sum(x, axis=0, out=out)  # function fallback with out=
    assert same(out, np.asarray(x).sum(axis=0))
    i = viproc.asarray(np.arange(6))
    o = viproc.zeros(6)
    np.add(i, 0.5, out=o)  # ufunc fallback (casting) with out=
    assert same(o, np.arange(6) + 0.5)


def _results(r):
    return list(r) if isinstance(r, (list, tuple)) else [r]


def _check_numpy_semantics(f, *arrays):
    """f on NumPy arrays and on viproc arrays: same values, same identity,
    same writability, same aliasing (a write through a result shows in the
    arguments exactly when it does in NumPy)."""
    import warnings

    np_args = [a.copy() for a in arrays]
    vp_args = [viproc.asarray(a) for a in arrays]
    np_res, vp_res = _results(f(*np_args)), _results(f(*vp_args))
    assert len(np_res) == len(vp_res)
    for rn, rv in zip(np_res, vp_res):
        assert same(rv, rn)
        for an, av in zip(np_args, vp_args):
            assert (rn is an) == (rv is av)
        idx = (0,) * np.ndim(rn)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # broadcast_arrays warns on write
            try:
                rn[idx] = 42
                np_ok = True
            except ValueError:
                np_ok = False
            try:
                rv[idx] = 42
                vp_ok = True
            except ValueError:
                vp_ok = False
        assert np_ok == vp_ok
        for an, av in zip(np_args, vp_args):
            assert same(av, an)  # the write reached the argument iff in NumPy


@pytest.mark.parametrize(
    "f",
    [
        np.atleast_1d,
        np.atleast_2d,
        lambda a: np.reshape(a, (4, 6)),
        lambda a: a.reshape(2, 12),
        np.transpose,
        lambda a: a.T,
        np.ravel,
        lambda a: np.squeeze(a[None]),
        lambda a: np.swapaxes(a, 0, 1),
        lambda a: np.moveaxis(a, 0, -1),
        np.flip,
        np.diagonal,
        lambda a: np.broadcast_to(a, (2, 4, 6)),
        lambda a: np.split(a, 2),
        lambda a: np.expand_dims(a, 0),
        lambda a: np.lib.stride_tricks.sliding_window_view(a, (2, 2)),
        lambda a: a[[0, 2]],  # advanced indexing: a copy in NumPy
        lambda a: np.concatenate([a, a]),
        lambda a: np.where(a > 0, a, 0.0),
    ],
)
def test_fallbacks_have_numpy_view_semantics(f):
    _check_numpy_semantics(f, rng.standard_normal((4, 6)))


def test_fallback_dtype_views_and_plain_arguments():
    _check_numpy_semantics(np.real, rng.standard_normal(6) + 1j)
    _check_numpy_semantics(np.imag, rng.standard_normal(6) + 1j)
    plain = np.zeros((2, 1))
    x = viproc.asarray(rng.standard_normal((1, 3)))
    bx, bp = np.broadcast_arrays(x, plain)
    assert bp.base is plain or np.shares_memory(bp, plain)  # NumPy's own view of plain
    assert not isinstance(bp, viproc.ndarray)
    r = np.atleast_1d(x, plain)
    assert r[0] is x and r[1] is plain


@pytest.mark.parametrize(
    "f",
    [
        lambda a: np.put(a, [0, 5], [-1.0, -2.0]),
        lambda a: np.place(a, a > 0, [7.0]),
        lambda a: np.putmask(a, a < 0, 0.0),
        lambda a: np.fill_diagonal(a, 3.0),
        lambda a: np.copyto(a, 1.5, where=a > 0),
        lambda a: np.add.at(a, ([0, 0, 1], [1, 1, 2]), 1.0),
        lambda a: np.negative(a, out=a),
        lambda a: np.cumsum(a, axis=0, out=a),
        lambda a: a.__setitem__((a > 0), 0.0),  # boolean index assignment
        lambda a: a.__setitem__(([0, 1], [2, 3]), a[([1, 0], [3, 2])]),  # overlapping
    ],
)
def test_mutating_functions_work_in_place_like_numpy(f):
    an = rng.standard_normal((4, 6))
    av = viproc.asarray(an)
    rn, rv = f(an), f(av)
    assert same(av, an)
    if isinstance(rn, np.ndarray):
        assert rv is av if rn is an else same(rv, rn)


def test_fallback_writes_respect_exports_and_pending_readers():
    x = viproc.asarray(np.arange(100_000, dtype=np.float64))
    keep = np.asarray(x)  # export alive: an alias, as in NumPy
    pending = np.sin(x)  # an issued reader (may still run)
    np.put(x, [0], [-1.0])  # NumPy writes x after the reader read it
    assert keep[0] == -1.0 and same(pending, np.sin(np.arange(100_000.0)))
    assert np.asarray(x)[0] == -1.0


def test_read_only_viproc_views_raise_like_numpy():
    x = viproc.asarray(rng.standard_normal(6))
    b = np.broadcast_to(x, (3, 6))
    with pytest.raises(ValueError, match="read-only"):
        b[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        b += 1.0
    with pytest.raises(ValueError, match="read-only"):
        np.add(b, 1.0, out=b)
    with pytest.raises(ValueError, match="read-only"):
        b[0][0] = 1.0  # views of a read-only array are read-only
    x[0] = 5.0  # the base stays writable, and the view sees the write
    assert np.asarray(b)[2, 0] == 5.0


@pytest.mark.parametrize("code", ["i", "l", "q"])
def test_every_c_integer_type_of_a_supported_width_imports(code):
    # int32 and int64 each have two NumPy type numbers (Linux: "l" and "q" for
    # int64; Windows: "i" and "l" for int32); NumPy treats them as one dtype.
    a = np.arange(-3, 4, dtype=code)
    v = viproc.asarray(a)
    assert isinstance(v, viproc.ndarray) and v.dtype == a.dtype
    assert np.array_equal(np.asarray(v * 3 + a), a * 3 + a)
    assert np.asarray(v ** 2).dtype == (a ** 2).dtype


def test_waits_inside_an_issue_release_the_gil():
    """An op on memory a live export aliases runs before the issue returns,
    and back-pressure waits for running tasks; a task may need the GIL
    (NumPy's integer power reports a negative exponent through it), so the
    main thread waits without holding it."""
    base = np.arange(1, 50_001, dtype=np.int64)
    exponent = viproc.asarray(np.full(50_000, -1, np.int64))
    x = viproc.asarray(base)
    e = np.asarray(x)  # live export: the next op on x runs synchronously
    with pytest.raises(ValueError, match="negative integer powers"):
        np.asarray(x ** exponent)
    del e
    try:
        viproc.init(max_active_tasks=4, sync_below=0, adaptive=False)
        x = viproc.asarray(base)
        exponent = viproc.asarray(np.full(50_000, -1, np.int64))
        with pytest.raises(ValueError, match="negative integer powers"):
            r = x ** exponent  # needs the GIL to fail
            rs = [x + i for i in range(16)]  # the main thread waits for slots meanwhile
            viproc.wait_all()
    finally:
        viproc.init()
