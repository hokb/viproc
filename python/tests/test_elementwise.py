"""Element-wise ops on every layout, with casts, and metadata-only views.

NumPy's loops are not layout-independent: their SIMD paths (complex
multiply, float32 exp/sin, ...) round differently from the scalar fallbacks,
and which one runs depends on the strides NumPy's ufunc machinery passes.
viproc must call the loops exactly as NumPy does: bit-identical results and
NumPy's result layouts for strided, reversed, transposed and broadcast
operands, with and without casts."""

import warnings

import numpy as np
import pytest

import viproc

DTYPES = ["b1", "i4", "i8", "f4", "f8", "c8", "c16"]
UNARY = [np.exp, np.log, np.sin, np.cos, np.tanh, np.sqrt, np.absolute, np.negative,
         np.square, np.reciprocal]
BINARY = [np.multiply, np.divide, np.add, np.subtract, np.power, np.maximum, np.arctan2,
          np.hypot, np.less, np.equal, np.logical_and, np.floor_divide]


def fallbacks():
    return viproc._ndarray.fallback_calls


def data(rng, shape, dtype):
    x = rng.standard_normal(shape) * 5
    if dtype.startswith("c"):
        x = x + 1j * rng.standard_normal(shape)
    if dtype == "b1":
        return np.asarray(x > 0)
    return np.asarray(x).astype(dtype)


LAYOUTS = {
    "C": lambda a: a,
    "T": lambda a: a.T,
    "rev": lambda a: a[..., ::-1] if a.ndim else a,
    "rev0": lambda a: a[::-1] if a.ndim else a,
    "step": lambda a: a[..., ::2] if a.ndim else a,
}


def identical(e, g):
    g = np.asarray(g)
    return (e.dtype == g.dtype and e.shape == g.shape and e.strides == g.strides
            and np.array_equal(np.ascontiguousarray(e).view(np.uint8),
                               np.ascontiguousarray(g).view(np.uint8)))


@pytest.mark.parametrize("seed", range(3))
def test_ops_on_strided_reversed_and_broadcast_operands_are_bit_identical(seed):
    rng = np.random.default_rng(seed)
    checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(250):
            dt = str(rng.choice(["f4", "f8", "c8", "c16"]))
            shape = tuple(int(s) for s in rng.integers(1, 30, size=int(rng.integers(1, 4))))
            k1, k2 = rng.choice(list(LAYOUTS), 2)
            x, y = data(rng, shape, dt), data(rng, shape, dt)
            a, va = LAYOUTS[k1](x), LAYOUTS[k1](viproc.asarray(x))
            if rng.random() < 0.4:
                op = UNARY[int(rng.integers(len(UNARY)))]
                assert identical(op(a), op(va)), (op.__name__, dt, shape, k1)
            else:
                op = BINARY[int(rng.integers(8))]
                if dt.startswith("c") and op in (np.maximum, np.arctan2, np.hypot):
                    continue
                if rng.random() < 0.3:  # broadcast a trailing part
                    y = data(rng, a.shape[int(rng.integers(0, a.ndim)):], dt)
                    b, vb = y, viproc.asarray(y)
                else:
                    b, vb = LAYOUTS[k2](y), LAYOUTS[k2](viproc.asarray(y))
                    if b.shape != a.shape:
                        continue
                assert identical(op(a, b), op(va, vb)), (op.__name__, dt, shape, k1, k2)
            checked += 1
    assert checked > 150


def test_single_elements_get_numpys_strides():
    # One element, reversed: NumPy calls its loop with the negative stride,
    # and float32 arctan2's SIMD and scalar paths differ in the last bit.
    x = np.array([1.5], np.float32)
    y = np.array([-2.25], np.float32)
    vx, vy = viproc.asarray(x), viproc.asarray(y)
    assert identical(np.arctan2(x[::-1], y[::-1]), np.arctan2(vx[::-1], vy[::-1]))
    assert identical(np.arctan2(x, y), np.arctan2(vx, vy))


@pytest.mark.parametrize("seed", range(3))
def test_mixed_dtypes_are_cast_asynchronously_as_numpy_casts(seed):
    rng = np.random.default_rng(seed)
    f0 = fallbacks()
    checked = 0
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for _ in range(300):
            d1, d2 = rng.choice(DTYPES, 2)
            shape = tuple(int(s) for s in rng.integers(1, 12, size=int(rng.integers(0, 4))))
            if rng.random() < 0.15:
                shape = (int(rng.integers(8000, 9000)),)  # longer than the buffer
            x = data(rng, shape, str(d1))
            y = data(rng, shape if rng.random() < 0.7 else shape[1:], str(d2))
            k1, k2 = rng.choice(["C", "T", "rev", "step"], 2)
            a, b = LAYOUTS[k1](x), LAYOUTS[k2](y)
            op = BINARY[int(rng.integers(len(BINARY)))]
            try:
                e = op(a, b)
            except (TypeError, ValueError):
                continue
            if e.dtype not in [np.dtype(d) for d in DTYPES]:
                continue  # e.g. bool ** bool is int8: not a viproc dtype
            g = op(LAYOUTS[k1](viproc.asarray(x)), LAYOUTS[k2](viproc.asarray(y)))
            assert identical(e, g), (op.__name__, d1, d2, a.shape, b.shape, k1, k2)
            checked += 1
    assert checked > 150
    assert fallbacks() == f0  # all of them ran asynchronously


def test_astype_casts_asynchronously_with_numpys_layout():
    rng = np.random.default_rng(7)
    f0 = fallbacks()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(150):
            shape = tuple(int(s) for s in rng.integers(1, 6, size=int(rng.integers(1, 4))))
            x = data(rng, shape, str(rng.choice(DTYPES)))
            for k in ["C", "T", "rev", "F"]:
                if k == "F":
                    a = np.asfortranarray(x)
                    v = viproc.asarray(a)
                else:
                    a, v = LAYOUTS[k](x), LAYOUTS[k](viproc.asarray(x))
                to, order = str(rng.choice(DTYPES)), str(rng.choice(["K", "C"]))
                assert identical(a.astype(to, order=order), v.astype(to, order=order))
    assert fallbacks() == f0
    c = viproc.asarray(np.array([1 + 2j]))
    with pytest.warns(np.exceptions.ComplexWarning):
        c.astype(np.float64)


def test_views_need_no_sync():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((4, 5, 6))
    c = (x + 1j * x[::-1]).astype(np.complex128)
    v, vc = viproc.asarray(x), viproc.asarray(c)
    f0 = fallbacks()
    cases = [
        lambda a: a.T, lambda a: np.transpose(a, (1, 0, 2)), lambda a: a.transpose(2, 0, 1),
        lambda a: np.swapaxes(a, 0, 2), lambda a: a.swapaxes(1, 2), lambda a: np.moveaxis(a, 0, -1),
        lambda a: a[:, None].squeeze(1), lambda a: np.expand_dims(a, 1), lambda a: np.flip(a, 1),
        lambda a: np.rot90(a), lambda a: np.broadcast_to(a[:1], (3, 5, 6)),
        lambda a: np.diagonal(a), lambda a: a.reshape(20, 6), lambda a: a.T.reshape(-1),
        lambda a: np.ravel(a), lambda a: a[:, ::2].ravel(), lambda a: a.flatten(),
        lambda a: a.mT, lambda a: np.atleast_3d(a[0]), lambda a: np.permute_dims(a, (2, 1, 0)),
    ]
    results = [(f(x), f(v)) for f in cases]
    results += [(f(c), f(vc)) for f in (lambda a: a.real, lambda a: a.imag,
                                        lambda a: a.view(np.float64), np.real)]
    assert fallbacks() == f0
    for e, g in results:
        assert isinstance(g, viproc.ndarray)
        assert identical(e, g)


def test_writes_through_views_and_read_only_views():
    x = np.arange(24.0).reshape(4, 6)
    v = viproc.asarray(x)
    v.T[0] = -1.0
    x.T[0] = -1.0
    r = viproc.asarray(np.array([1 + 2j, 3 + 4j]))
    r.real[...] = 7.0
    assert np.array_equal(np.asarray(v), x)
    assert np.array_equal(np.asarray(r), [7 + 2j, 7 + 4j])
    with pytest.raises(ValueError):
        np.broadcast_to(v, (2, 4, 6))[0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        np.transpose(v, (0, 0))  # NumPy's own errors
    assert np.real(v) is v  # NumPy returns real data itself


@pytest.mark.parametrize("n", [5, 100_000])  # run directly / on a worker
@pytest.mark.parametrize("dtype", ["i4", "i8"])
def test_integer_power_with_negative_exponent_raises_numpys_value_error(n, dtype):
    # NumPy's integer power loop raises through the Python API, which a worker
    # cannot report: viproc checks the exponents first, on any thread.
    base = np.arange(n, dtype=dtype) % 7
    exp = np.full(n, 2, dtype=dtype)
    exp[-1] = -1
    with pytest.raises(ValueError, match="Integers to negative integer powers"):
        np.power(base, exp)
    x, y = viproc.asarray(base), viproc.asarray(exp)
    for f in (lambda: np.power(x, y), lambda: viproc.power(x, y), lambda: x ** y,
              lambda: x ** -1, lambda: np.power(x, exp[::-1])):
        with pytest.raises(ValueError, match="Integers to negative integer powers"):
            np.asarray(f())
    assert np.array_equal(np.asarray(x + 1), base + 1)  # no stray Python error
    ok = np.abs(exp)
    assert np.array_equal(np.asarray(x ** viproc.asarray(ok)), base ** ok)
