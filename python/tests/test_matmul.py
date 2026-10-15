"""Asynchronous matrix products: np.matmul / `@` with NumPy's own matmul loop
and np.dot where it makes the same loop calls. Every result must be
bit-identical to NumPy (values, signs of zeros, NaNs, dtype, shape and
memory layout), for all shapes, broadcasts and operand layouts."""

import numpy as np
import pytest

import viproc

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")  # NumPy's FP warnings
N = 150  # random cases per dtype
DTYPES = [np.bool_, np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]


def fallbacks():
    return viproc._ndarray.fallback_calls


def same(got, expected):
    got_a = np.asarray(got)
    expected_a = np.asarray(expected)
    if got_a.shape != expected_a.shape or got_a.dtype != expected_a.dtype:
        return False
    if np.isscalar(expected) != (np.isscalar(got) or getattr(got, "_scalar", False)):
        return False  # NumPy's scalar: viproc's, or a pending 0-d array marked as one
    if expected_a.ndim and expected_a.size > 1 and got_a.strides != expected_a.strides:
        return False
    if expected_a.dtype.kind in "fc":
        return (np.array_equal(got_a, expected_a, equal_nan=True)
                and np.array_equal(np.signbit(got_a.real), np.signbit(expected_a.real))
                and np.array_equal(np.signbit(got_a.imag), np.signbit(expected_a.imag)))
    return np.array_equal(got_a, expected_a)


def data(rng, shape, dtype):
    kind = np.dtype(dtype).kind
    if kind == "b":
        return np.asarray(rng.random(shape) < 0.5)
    if kind == "i":
        return np.asarray(rng.integers(-1000, 1000, shape)).astype(dtype)
    x = np.asarray(rng.standard_normal(shape) * 10.0 ** rng.integers(-3, 3, shape))
    if rng.random() < 0.2:
        x[np.asarray(rng.random(shape) < 0.05)] = rng.choice([0.0, -0.0, np.nan, np.inf])
    if kind == "c":
        x = x + 1j * rng.standard_normal(shape)
    return x.astype(dtype)


def layout(rng, shape):
    """A random transform producing `shape`: (source shape, fn) with fn
    applied identically to the NumPy and the viproc source."""
    nd = len(shape)
    k = int(rng.integers(0, 5))
    if k == 1 and nd >= 2:  # transposed source (F order for 2-d)
        perm = rng.permutation(nd)
        inv = np.argsort(perm)
        src = tuple(shape[i] for i in inv)
        return src, lambda x: np.transpose(x, tuple(int(i) for i in perm))
    if k == 2 and nd >= 1:  # stepped, maybe reversed, along one axis
        ax = int(rng.integers(0, nd))
        step = int(rng.choice([2, -1, -2]))
        src = list(shape)
        src[ax] = shape[ax] * abs(step)
        key = tuple(slice(None, None, step) if i == ax else slice(None) for i in range(nd))
        return tuple(src), lambda x: x[key]
    if k == 3 and nd >= 1:  # a column block of a wider array (BLAS-able, not contiguous)
        src = list(shape)
        src[-1] = shape[-1] + 3
        key = (Ellipsis, slice(1, 1 + shape[-1]))
        return tuple(src), lambda x: x[key]
    if k == 4 and nd >= 1:  # broadcast along the first axis (zero stride)
        src = (1,) + tuple(shape[1:])
        return src, lambda x: np.broadcast_to(x, shape)
    return tuple(shape), lambda x: x


def operand(rng, shape, dtype, pending):
    src, fn = layout(rng, shape)
    x = data(rng, src, dtype)
    v = viproc.asarray(x)
    if pending:  # make it the result of an op: issued, maybe not ready yet
        x = x.copy()
        v = v.copy()
    return fn(x), fn(v)


def random_shapes(rng):
    dims = [0, 1, 2, 3, 5, 8, 17, 40]
    m, n, p = (int(rng.choice(dims)) for _ in range(3))
    if rng.random() < 0.1:
        m, n, p = (int(rng.integers(60, 200)) for _ in range(3))
    batch = tuple(int(rng.choice([1, 2, 3])) for _ in range(int(rng.integers(0, 3))))
    a_batch = tuple(1 if rng.random() < 0.3 else e for e in batch)
    b_batch = tuple(1 if rng.random() < 0.3 else e for e in batch)
    a_batch = a_batch[int(rng.integers(0, len(a_batch) + 1)):]
    b_batch = b_batch[int(rng.integers(0, len(b_batch) + 1)):]
    a = a_batch + (m, n) if rng.random() < 0.85 else (n,)
    b = b_batch + (n, p) if rng.random() < 0.85 else (n,)
    return a, b


@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_fuzz(dtype):
    rng = np.random.default_rng(np.dtype(dtype).num)
    for _ in range(N):
        sa, sb = random_shapes(rng)
        a, va = operand(rng, sa, dtype, rng.random() < 0.5)
        b, vb = operand(rng, sb, dtype, rng.random() < 0.5)
        expected = a @ b
        before = fallbacks()
        got = va @ vb
        assert fallbacks() == before, (sa, sb)
        assert isinstance(got, viproc.ndarray) or np.isscalar(got)
        assert same(got, expected), (sa, sb, dtype)
        assert same(np.matmul(va, vb), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_matmul_blas_paths(dtype):
    """gemm (C/F, both transposes), syrk (a @ a.T), gemv in both directions,
    dot, buffers for non-BLAS-able operands, and sizes where OpenBLAS
    splits the work across its threads."""
    rng = np.random.default_rng(7)
    for n in [2, 9, 64, 130, 600]:
        a = data(rng, (n, n + 1), dtype)
        b = data(rng, (n + 1, n), dtype)
        v = data(rng, (n + 1,), dtype)
        va, vb, vv = viproc.asarray(a), viproc.asarray(b), viproc.asarray(v)
        cases = [
            (lambda x, y, z: x @ y),
            (lambda x, y, z: x @ x.T),
            (lambda x, y, z: x.T @ x),
            (lambda x, y, z: y.T @ x.T),
            (lambda x, y, z: x @ z),
            (lambda x, y, z: z @ y),
            (lambda x, y, z: z @ z),
            (lambda x, y, z: x[:, ::2] @ y[::2]),
            (lambda x, y, z: x[::-1] @ y[:, ::-1]),
            (lambda x, y, z: x[::2, ::2] @ y[::2, ::2]),
        ]
        for f in cases:
            assert same(f(va, vb, vv), f(a, b, v)), n


def test_matmul_many_concurrent():
    """Independent products overlap on the workers; values as NumPy's."""
    rng = np.random.default_rng(3)
    ms = [data(rng, (300, 300), np.float64) for _ in range(8)]
    vs = [viproc.asarray(m) for m in ms]
    expected = [(m @ m) @ m for m in ms]
    got = [(v @ v) @ v for v in vs]
    for g, e in zip(got, expected):
        assert same(g, e)


def test_matmul_scalar_result():
    a = np.arange(5.0)
    va = viproc.asarray(a)
    r = va @ va
    assert np.isscalar(r) or r._scalar
    assert float(r) == a @ a
    big = np.arange(5000.0)
    vb = viproc.asarray(big)
    assert same(np.asarray(float(vb @ vb)), np.asarray(big @ big))


def test_matmul_errors_as_numpy():
    a = viproc.asarray(np.ones((2, 3)))
    with pytest.raises(ValueError):
        a @ a
    with pytest.raises(ValueError):
        np.matmul(a, viproc.asarray(np.float64(2.0)))
    with pytest.raises(ValueError):
        viproc.asarray(np.ones((2, 2, 3))) @ viproc.asarray(np.ones((3, 3, 4)))


def test_matmul_mixed_operands():
    rng = np.random.default_rng(5)
    a = data(rng, (4, 6), np.float64)
    b = data(rng, (6, 3), np.float64)
    va = viproc.asarray(a)
    assert same(va @ b, a @ b)
    assert same(a @ viproc.asarray(b), a @ b)
    # different dtypes: NumPy casts in its iterator; viproc lets NumPy run it
    assert same(va @ b.astype(np.float32), a @ b.astype(np.float32))
    assert same(va @ b.astype(np.int32), a @ b.astype(np.int32))


def dot_operand(rng, shape, dtype):
    src, fn = layout(rng, shape)
    x = data(rng, src, dtype)
    return fn(x), fn(viproc.asarray(x).copy())


@pytest.mark.parametrize("dtype", DTYPES)
def test_dot_fuzz(dtype):
    rng = np.random.default_rng(100 + np.dtype(dtype).num)
    dims = [1, 2, 3, 7, 33, 150] if np.dtype(dtype).kind in "fc" else [0, 1, 2, 5, 33]
    for _ in range(N):
        m, n, p = (int(rng.choice(dims)) for _ in range(3))
        sa = (m, n) if rng.random() < 0.7 else (n,)
        sb = (n, p) if rng.random() < 0.7 else (n,)
        a, va = dot_operand(rng, sa, dtype)
        b, vb = dot_operand(rng, sb, dtype)
        expected = np.dot(a, b)
        assert same(np.dot(va, vb), expected), (sa, sb, dtype)
        assert same(va.dot(vb), expected), (sa, sb, dtype)


def test_dot_async_cases():
    """dot of vectors and matrices runs asynchronously (no fallback)."""
    rng = np.random.default_rng(11)
    a = viproc.asarray(data(rng, (30, 20), np.float64))
    b = viproc.asarray(data(rng, (20, 10), np.float64))
    v = viproc.asarray(data(rng, (20,), np.float64))
    before = fallbacks()
    for r in [np.dot(a, b), a.dot(b), np.dot(a, v), np.dot(v, b), np.dot(v, v),
              np.dot(a[:, ::2], b[::2]), np.dot(a, a.T)]:
        np.asarray(r)
    assert fallbacks() == before


def test_dot_fallbacks_stay_numpy():
    rng = np.random.default_rng(12)
    a = data(rng, (3, 4, 5), np.float64)
    b = data(rng, (2, 5, 3), np.float64)
    assert same(np.dot(viproc.asarray(a), viproc.asarray(b)), np.dot(a, b))
    assert same(np.dot(viproc.asarray(a), 2.5), np.dot(a, 2.5))
    out = np.empty((3, 4, 2, 3))
    np.dot(viproc.asarray(a), viproc.asarray(b), out=out)
    assert same(out, np.dot(a, b))
