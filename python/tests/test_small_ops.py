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


REDUCTION_CALLS = [
    ("sum", {}), ("sum", {"axis": 0}), ("sum", {"axis": (0, 1), "keepdims": True}),
    ("sum", {"dtype": np.float32}), ("sum", {"initial": 2.5}), ("prod", {"axis": -1}),
    ("max", {}), ("max", {"axis": 1, "keepdims": True}), ("min", {"initial": -1.0}),
    ("any", {}), ("all", {"axis": 0}), ("mean", {}), ("mean", {"axis": 1}),
    ("var", {"ddof": 1}), ("std", {"axis": 0}), ("argmax", {}), ("argmin", {"axis": 1}),
]


@pytest.mark.parametrize("name, kwargs", REDUCTION_CALLS)
def test_small_ready_reductions_are_numpy_on_the_buffer(name, kwargs):
    x = np.random.default_rng(1).standard_normal((6, 7))
    v = viproc.asarray(x)
    viproc.wait_all()
    f0 = fallbacks()
    for got, expected in [(getattr(v, name)(**kwargs), getattr(x, name)(**kwargs)),
                          (getattr(np, name)(v, **kwargs), getattr(np, name)(x, **kwargs))]:
        assert type(got) is (viproc.ndarray if isinstance(expected, np.ndarray)
                             else type(expected))
        got = np.asarray(got)
        assert got.dtype == expected.dtype and got.shape == np.shape(expected)
        assert np.array_equal(got, expected)
    assert fallbacks() == f0  # NumPy ran it directly, no sync and no fallback


def test_small_ready_reductions_raise_numpy_errors_and_keep_data():
    v = viproc.asarray(np.arange(6.0).reshape(2, 3))
    viproc.wait_all()
    with pytest.raises(np.exceptions.AxisError):
        v.sum(axis=2)
    with pytest.raises(ValueError):
        viproc.asarray(np.zeros(0)).max()
    with pytest.raises(TypeError):
        v.sum(bogus=1)
    out = viproc.asarray(np.zeros(3))
    r = np.sum(v, axis=0, out=out)  # a viproc out=: not delegated
    assert r is out and np.array_equal(np.asarray(out), [3.0, 5.0, 7.0])
    v += 1  # the borrowed view did not stay a reader
    assert np.array_equal(np.asarray(v), np.arange(1.0, 7.0).reshape(2, 3))


@pytest.mark.parametrize("name, args, kwargs", [
    ("copy", (), {}), ("copy", ("F",), {}), ("astype", (np.float32,), {}),
    ("astype", (np.int64,), {"order": "F"}), ("astype", (np.float64,), {"copy": False}),
    ("astype", (np.complex128,), {}),
])
def test_small_ready_copy_and_astype_are_numpy_on_the_buffer(name, args, kwargs):
    v = viproc.asarray(np.asfortranarray(np.random.default_rng(2).standard_normal((5, 4))))
    viproc.wait_all()
    x = np.asarray(v).copy(order="K")  # NumPy's array with v's layout
    f0 = fallbacks()
    got, expected = getattr(v, name)(*args, **kwargs), getattr(x, name)(*args, **kwargs)
    assert fallbacks() == f0
    assert type(got) is viproc.ndarray
    assert got.dtype == expected.dtype and got.strides == expected.strides
    assert np.array_equal(np.asarray(got), expected)
    if expected is x:
        assert got is v  # astype(copy=False) of the same dtype: the array itself
    else:
        got += 1  # a copy of its own
        assert np.array_equal(np.asarray(v), x)


def test_reshape_and_swapaxes_methods_match_numpy():
    x = np.arange(24.0).reshape(2, 3, 4)
    v = viproc.asarray(x)
    for args in [((4, 6),), (4, 6), ((-1,),), (-1,), ((24,),),
                 (np.int64(4), 6), ([2, 12],)]:
        assert np.array_equal(np.asarray(v.reshape(*args)), x.reshape(*args))
    assert np.array_equal(np.asarray(v.reshape(6, 4, order="F")), x.reshape(6, 4, order="F"))
    assert np.array_equal(np.asarray(v.T.reshape(-1)), x.T.reshape(-1))  # a copy
    assert np.array_equal(np.asarray(v.swapaxes(0, -1)), x.swapaxes(0, -1))
    for shape in [(5, 5), (5,), (-1, 5), (5, -1), (-1, -1)]:
        with pytest.raises(ValueError) as e_np:
            x.reshape(shape)
        with pytest.raises(ValueError) as e_vp:
            v.reshape(shape)
        assert str(e_vp.value) == str(e_np.value)
    with pytest.raises(np.exceptions.AxisError):
        v.swapaxes(0, 3)
    w = v.reshape(4, 6)  # a view: writes show through
    w[0, 0] = 100.0
    assert float(v[0, 0, 0]) == 100.0
