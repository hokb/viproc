"""Small ops on ready data: the direct path (no task) and the scalar
shortcut (ops on scalars only compute right away, as NumPy scalars)."""

import operator

import numpy as np
import pytest

import viproc


def fallbacks():
    return viproc._ndarray.fallback_calls


def test_small_ops_on_ready_data_run_directly_without_a_task():
    x = viproc.asarray(np.arange(10.0))
    viproc.wait_all()
    before = viproc.offload_stats()
    y = np.sin(x) * 2.0 + x
    assert y.ready()
    after = viproc.offload_stats()
    assert after["direct"] == before["direct"] + 3
    assert after["inline"] == before["inline"] and after["offloaded"] == before["offloaded"]
    e = np.sin(np.arange(10.0)) * 2.0 + np.arange(10.0)
    assert np.array_equal(np.asarray(y), e)


def test_full_reductions_of_ready_small_data_return_numpy_scalars():
    x = np.random.default_rng(0).standard_normal(100)
    v = viproc.asarray(x)
    viproc.wait_all()
    s = np.sum(v)
    assert type(s) is np.float64 and s == np.sum(x)
    assert type(np.argmax(v)) is np.intp
    assert type(v[3]) is np.float64


@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, operator.truediv,
                                operator.floordiv, operator.mod, operator.pow, divmod,
                                operator.lt, operator.eq, operator.ge])
def test_ops_on_ready_scalars_are_numpy_scalar_ops(op):
    f0 = fallbacks()
    a = viproc.asarray(np.array(7.25))
    b = viproc.asarray(np.array(-2.5))
    viproc.wait_all()
    for x, y, ex, ey in [(a, b, np.float64(7.25), np.float64(-2.5)),
                         (a, 3, np.float64(7.25), 3), (1.5, b, 1.5, np.float64(-2.5)),
                         (a, np.float32(2), np.float64(7.25), np.float32(2))]:
        got, expected = op(x, y), op(ex, ey)
        assert type(got) is type(expected)
        assert got == expected
    assert fallbacks() == f0


def test_unary_and_ufunc_calls_on_ready_scalars():
    a = viproc.asarray(np.array(0.75, np.float32))
    viproc.wait_all()
    e = np.float32(0.75)
    for got, expected in [(-a, -e), (abs(a), abs(e)), (np.sin(a), np.sin(e)),
                          (np.arctan2(a, 2.0), np.arctan2(e, 2.0)), (np.add(a, a), e + e)]:
        assert type(got) is type(expected) and got == expected


def test_any_nd_operand_keeps_the_array_path_and_shape():
    s = viproc.asarray(np.array(2.0))
    one = viproc.asarray(np.array([[3.0]]))
    viproc.wait_all()
    r = s * one  # (1, 1): NumPy's broadcast shape, not a scalar
    assert isinstance(r, viproc.ndarray) and r.shape == (1, 1)
    assert np.array_equal(np.asarray(r), [[6.0]])


def test_inplace_operators_on_scalar_results_rebind_like_numpy_scalars():
    big = viproc.asarray(np.ones(100_000))
    s = np.sum(big * 2.0)  # pending 0-d: a NumPy scalar in NumPy
    t = s
    s += 1.0
    assert float(t) == 200_000.0 and float(s) == 200_001.0
    a = viproc.asarray(np.array(5.0))  # a 0-d array: in place, as NumPy's
    b = a
    a += 1.0
    assert float(b) == 6.0


def test_errors_of_direct_ops_surface_at_the_sync_point():
    small = viproc.asarray(np.arange(10.0))
    idx = viproc.asarray(np.array([1, 99]))
    viproc.wait_all()
    r = small[idx]  # ran directly; the error is stored on the result
    with pytest.raises(IndexError, match="index 99 is out of bounds"):
        np.asarray(r)
