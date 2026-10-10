"""Python bridge: viproc arrays must behave like NumPy arrays (bit-identical
results), with supported ops running asynchronously."""

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
    x = viproc.asarray(rng.standard_normal(3_000_000))
    viproc.wait_all()
    t0 = time.perf_counter()
    y = x
    for _ in range(20):
        y = np.sqrt(y * y + 1.0)
    issue_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    y.wait()
    compute_time = time.perf_counter() - t1
    assert issue_time < compute_time, (issue_time, compute_time)


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
