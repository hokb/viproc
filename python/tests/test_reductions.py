"""Asynchronous reductions: NumPy's own reduction loops, called by viproc in
the blocks NumPy would use. Every result must be bit-identical to NumPy (also
the sign of zeros and NaNs), for all shapes, axes and memory layouts."""

import numpy as np
import pytest

import viproc

ASYNC = np.lib.NumpyVersion(np.__version__) >= "2.3.0"
UFUNCS = {
    "f": [np.add, np.multiply, np.minimum, np.maximum, np.fmin, np.fmax],
    "c": [np.add, np.multiply],
    "i": [np.add, np.multiply, np.minimum, np.maximum, np.bitwise_and, np.bitwise_or,
          np.bitwise_xor],
    "b": [np.logical_and, np.logical_or, np.logical_xor, np.minimum, np.maximum],
}


def fallbacks():
    return viproc._ndarray.fallback_calls


def same(got, expected):
    got = np.asarray(got)
    expected = np.asarray(expected)
    if got.shape != expected.shape or got.dtype != expected.dtype:
        return False
    if expected.dtype.kind in "fc":
        return (np.array_equal(got, expected, equal_nan=True)
                and np.array_equal(np.signbit(got.real), np.signbit(expected.real)))
    return np.array_equal(got, expected)


def sensitive(rng, shape, dtype):
    """Data whose sums depend on the summation order."""
    kind = np.dtype(dtype).kind
    if kind == "b":
        return np.asarray(rng.random(shape) < 0.5)
    if kind == "i":
        return np.asarray(rng.integers(-50, 50, shape)).astype(dtype)
    x = np.asarray(rng.standard_normal(shape) * 10.0 ** rng.integers(-6, 6, shape))
    if rng.random() < 0.3:
        x[np.asarray(rng.random(shape) < 0.05)] = np.nan
    if rng.random() < 0.3:
        x[np.asarray(rng.random(shape) < 0.1)] = rng.choice([0.0, -0.0])
    if kind == "c":
        x = x + 1j * rng.standard_normal(shape) * 10.0 ** rng.integers(-6, 6, shape)
    return x.astype(dtype)


def view_ops(rng, nd):
    """Transpose / reverse / step, applied identically to NumPy and viproc."""
    ops = []
    for _ in range(int(rng.integers(0, 3))):
        k = int(rng.integers(0, 3))
        if k == 0 and nd > 1:
            ops.append(("T", tuple(int(i) for i in rng.permutation(nd))))
        elif k == 1 and nd:
            ops.append(("rev", int(rng.integers(0, nd))))
        elif k == 2 and nd:
            ops.append(("step", int(rng.integers(0, nd))))
    return ops


def apply(a, ops):
    for op, arg in ops:
        if op == "T":
            a = np.transpose(a, arg)
        elif op == "rev":
            a = a[(slice(None),) * arg + (slice(None, None, -1),)]
        else:
            a = a[(slice(None),) * arg + (slice(None, None, 2),)]
    return a


@pytest.mark.parametrize("seed", range(4))
def test_ufunc_reduce_is_bit_identical_for_random_shapes_axes_and_layouts(seed):
    rng = np.random.default_rng(seed)
    checked = 0
    with np.errstate(all="ignore"):
        for _ in range(120):
            nd = int(rng.integers(0, 5))
            shape = tuple(int(rng.choice([1, 2, 3, 7, 9, 16, 33, 100, 257])) for _ in range(nd))
            if np.prod(shape) > 200_000:
                continue
            dtype = str(rng.choice(["f4", "f8", "c8", "c16", "i4", "i8", "b1"]))
            base = sensitive(rng, shape, dtype)
            if rng.random() < 0.3:
                base = np.asfortranarray(base)
            ops = view_ops(rng, nd)
            x = apply(base, ops)
            v = apply(viproc.asarray(base), ops)
            assert np.asarray(v).strides == x.strides
            for uf in UFUNCS[np.dtype(dtype).kind]:
                if x.size == 0:
                    continue
                axis = tuple(i for i in range(x.ndim) if rng.random() < 0.5)
                if rng.random() < 0.3:
                    axis = None
                keep = bool(rng.random() < 0.3)
                expected = uf.reduce(x, axis=axis, keepdims=keep)
                got = uf.reduce(v, axis=axis, keepdims=keep)
                assert same(got, expected), (uf.__name__, dtype, x.shape, x.strides, ops, axis)
                checked += 1
    assert checked > 200


@pytest.mark.parametrize("dtype", ["f8", "f4", "c16"])
def test_long_strided_and_reversed_reductions(dtype):
    rng = np.random.default_rng(1)
    b = sensitive(rng, 300_007, dtype)
    vb = viproc.asarray(b)
    m = sensitive(rng, (1200, 997), dtype)
    vm = viproc.asarray(m)
    f = np.asfortranarray(m)
    vf = viproc.asarray(f)
    cases = [(b, vb), (b[::2], vb[::2]), (b[::-3], vb[::-3]), (m, vm), (f, vf),
             (m[:, ::2], vm[:, ::2]), (m[::-1], vm[::-1]), (m.T, np.transpose(vm))]
    with np.errstate(all="ignore"):
        for x, v in cases:
            for axis in [None] + list(range(x.ndim)):
                for uf in UFUNCS["c" if dtype.startswith("c") else "f"]:
                    assert same(uf.reduce(v, axis=axis), uf.reduce(x, axis=axis)), (uf, axis)


@pytest.mark.skipif(not ASYNC, reason="async reductions need NumPy >= 2.3")
def test_reductions_run_asynchronously_without_syncs():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((400, 300))
    v = viproc.asarray(x)
    f0 = fallbacks()
    results = [
        (np.sum(v), np.sum(x)), (v.sum(axis=0), x.sum(axis=0)),
        (np.prod(v * 1e-3 + 1, axis=1), np.prod(x * 1e-3 + 1, axis=1)),
        (v.max(axis=1, keepdims=True), x.max(axis=1, keepdims=True)),
        (np.min(v), np.min(x)), (np.amax(v, axis=0), np.amax(x, axis=0)),
        (np.any(v > 3), np.any(x > 3)), (np.all(v != 0, axis=0), np.all(x != 0, axis=0)),
        (np.any(v), np.any(x)), (v.all(), x.all()),
        (np.mean(v), np.mean(x)), (v.mean(axis=1, keepdims=True), x.mean(axis=1, keepdims=True)),
        (np.var(v), np.var(x)), (v.var(axis=0, ddof=1), x.var(axis=0, ddof=1)),
        (np.std(v, axis=1), np.std(x, axis=1)), (np.add.reduce(v), np.add.reduce(x)),
        (np.maximum.reduce(v, axis=None), np.maximum.reduce(x, axis=None)),
    ]
    assert fallbacks() == f0  # nothing ran eagerly, nothing synced
    for got, expected in results:
        assert isinstance(got, viproc.ndarray)
        assert same(got, expected)


def test_reductions_of_computed_fortran_and_transposed_arrays():
    # Element-wise results take NumPy's memory order (F stays F), so a
    # reduction visits the elements in NumPy's order.
    rng = np.random.default_rng(3)
    x = np.asfortranarray(rng.standard_normal((300, 200)) * 10.0 ** rng.integers(-8, 8, (300, 200)))
    v = viproc.asarray(x)
    y, w = np.sin(x) * 1e3, np.sin(v) * 1e3
    assert w.strides == y.strides
    t, u = x.T * 2.0 + x.T, np.transpose(v) * 2.0 + np.transpose(v)
    assert u.strides == t.strides
    for got, expected in [(np.sum(w), np.sum(y)), (np.sum(u), np.sum(t)),
                          (w.sum(axis=0), y.sum(axis=0)), (np.var(w), np.var(y)),
                          (np.std(u, axis=1), np.std(t, axis=1))]:
        assert same(got, expected)
    c = rng.standard_normal((300, 200))
    assert (viproc.asarray(c) + v).strides == (c + x).strides  # mixed: C order


def test_copy_on_write_keeps_the_layout():
    rng = np.random.default_rng(4)
    x = np.asfortranarray(rng.standard_normal((200, 100)) * 10.0 ** rng.integers(-8, 8, (200, 100)))
    v = viproc.asarray(x)
    pending = np.sin(v)  # a reader of v
    v += 1.0  # copy-on-write: v moves to a new buffer
    x += 1.0
    assert v.strides == x.strides
    assert same(np.sum(v), np.sum(x)) and same(np.sum(pending), np.sum(np.sin(x - 1.0)))


@pytest.mark.parametrize("dtype", ["f8", "f4", "c16", "c8", "i4", "i8", "b1"])
def test_reduction_functions_match_numpy_for_every_dtype(dtype):
    rng = np.random.default_rng(5)
    x = sensitive(rng, (60, 70), dtype)
    v = viproc.asarray(x)
    funcs = [np.sum, np.prod, np.max, np.min, np.any, np.all, np.mean, np.var, np.std]
    with np.errstate(all="ignore"):
        for fn in funcs:
            for kw in [{}, {"axis": 0}, {"axis": 1, "keepdims": True}, {"axis": (0, 1)}]:
                if fn in (np.var, np.std) and dtype == "b1":
                    continue
                expected = fn(x, **kw)
                got = fn(v, **kw)
                assert same(got, expected), (fn.__name__, kw)


def test_reduction_arguments_and_errors_follow_numpy():
    x = np.arange(12.0).reshape(3, 4)
    v = viproc.asarray(x)
    assert same(np.sum(v, axis=-1), np.sum(x, axis=-1))
    assert same(np.sum(v, dtype=np.float32), np.sum(x, dtype=np.float32))  # casts: NumPy
    assert same(np.sum(v, initial=5.0), np.sum(x, initial=5.0))
    assert same(np.sum(v, where=x > 3), np.sum(x, where=x > 3))
    i = np.arange(10, dtype=np.int32)
    assert same(np.sum(viproc.asarray(i)), np.sum(i))  # int32 sums in int64
    with pytest.raises(np.exceptions.AxisError):
        np.sum(v, axis=2)
    with pytest.raises(ValueError):
        np.max(viproc.asarray(np.zeros((0, 3))))
    assert same(np.sum(viproc.asarray(np.zeros((0, 3)))), 0.0)


def test_scalar_results_behave_like_numpy_scalars():
    x = np.array([0.5, 1.25, 3.0])
    s = np.sum(viproc.asarray(x))
    assert f"{s:.3f}" == f"{np.sum(x):.3f}"
    assert round(s, 1) == round(np.sum(x), 1)
    assert float(s) == np.sum(x) and int(s) == 4 and s == np.sum(x)
    assert s.item() == np.sum(x).item()
