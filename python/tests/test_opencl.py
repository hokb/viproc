"""Optimization level O4: numpy.linalg (det, slogdet, inv, solve, cholesky),
numpy.fft and matmul on an OpenCL device (docs/design/opencl.md).

O4 results are correct but not bit-identical: values within the rounding of
the device's computation in NumPy's precision, and NumPy's dtypes, shapes,
layouts, errors and warnings. The tests run on whatever OpenCL device
VIPROC_TEST_OPENCL_DEVICE names (default "any": the first one, e.g. pocl's
CPU device in CI: `apt install pocl-opencl-icd libclblast1`) and are skipped
without one. Every op runs on the device (device_min_work=0) and the device
counter proves it."""

import os
import warnings

import numpy as np
import pytest

import viproc
from async_path import pending

L = np.linalg
rng = np.random.default_rng(11)
SPEC = os.environ.get("VIPROC_TEST_OPENCL_DEVICE", "any")


def start(level=4, device=SPEC, min_work=0):
    viproc.init(sync_below=0, adaptive=False, level=level, device=device,
                device_min_work=min_work)


@pytest.fixture(autouse=True, scope="module")
def o4():
    saved = os.environ.pop("VIPROC_OPT", None)
    start()
    info = viproc.device_info()
    try:
        if not info["selected"]:
            pytest.skip(f"no OpenCL device: {info['reason']}")
        yield info
    finally:
        if saved is not None:
            os.environ["VIPROC_OPT"] = saved
        viproc.init(device_min_work=-1)


def sample(shape, dtype):
    dtype = np.dtype(dtype)
    x = rng.standard_normal(shape)
    if dtype.kind == "c":
        x = x + 1j * rng.standard_normal(shape)
    return x.astype(dtype)


def spd(a):
    n = a.shape[-1]
    return a @ np.conj(np.swapaxes(a, -1, -2)) + n * np.eye(n, dtype=a.dtype)


def tolerance(dtype, n=1):
    single = np.dtype(dtype) in (np.float32, np.complex64)
    return (2e-4 if single else 1e-10) * max(n, 1)


def close(got, want, rtol):
    """NumPy's dtype, shape and layout; NaN and inf where NumPy has them,
    other values within rtol of the largest finite one."""
    if isinstance(want, tuple):
        assert type(got) is type(want) and len(got) == len(want)
        for g, w in zip(got, want):
            close(g, w, rtol)
        return
    g, w = np.asarray(got), np.asarray(want)
    assert g.dtype == w.dtype and g.shape == w.shape
    assert g.size == 0 or [s for s, n in zip(g.strides, g.shape) if n > 1] == [
        s for s, n in zip(w.strides, w.shape) if n > 1]
    finite = np.isfinite(w)
    assert np.array_equal(g[~finite], w[~finite], equal_nan=True)
    scale = max(np.abs(w[finite]).max(initial=0), 1e-300)
    np.testing.assert_allclose(g[finite], w[finite], rtol=0, atol=rtol * scale)


def on_device(fn, *args, ops=1, want=None):
    """fn(*args) on viproc arrays; asserts `ops` device ops, no fallback to
    the CPU and NumPy's warnings (those of fn on the NumPy arrays, or none
    when `want` is given); returns the result."""
    if want is None:
        with warnings.catch_warnings(record=True) as want_w:
            warnings.simplefilter("always")
            fn(*args)
    else:
        want_w = []
    d0 = viproc.offload_stats()
    with warnings.catch_warnings(record=True) as got_w:
        warnings.simplefilter("always")
        r = fn(*[viproc.asarray(a) if isinstance(a, np.ndarray) else a for a in args])
        viproc.wait_all()
    d1 = viproc.offload_stats()
    assert d1["device"] - d0["device"] == ops
    assert d1["device_fallbacks"] == d0["device_fallbacks"]
    assert sorted({str(w.message) for w in got_w}) == sorted({str(w.message) for w in want_w})
    return r


def same(fn, tol, *args, ops=1):
    """fn on the device within tol of fn in NumPy (values, dtype, layout,
    warnings)."""
    close(on_device(fn, *args, ops=ops), fn(*args), tol)


def test_device_info(o4):
    assert o4["selected"] and o4["name"] and o4["platform"]
    assert viproc.opt_level() == 4
    assert o4["min_work"] == 0


# n <= 64: one work-group per matrix, the batch in one launch; larger: blocked
# (panels and CLBlast's trsm/gemm/syrk), panels of 64 columns.
SQUARE = [(5, 5), (40, 6, 6), (3, 64, 64), (130, 130), (2, 97, 97)]


@pytest.mark.parametrize("shape", SQUARE)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.float32, np.complex64])
def test_factorizations(shape, dtype, o4):
    if (shape[-1] > 64 and not o4["clblast"]) or not o4["fp64"]:
        pytest.skip("needs CLBlast and fp64")
    a = sample(shape, dtype)
    n = shape[-1]
    tol = tolerance(dtype, n)
    for f in (L.det, L.slogdet, L.inv):
        same(f, tol, a)
    b = sample(shape[:-1] + (3,), dtype)
    same(L.solve, tol, a, b)
    same(L.solve, tol, a if a.ndim == 2 else a[0], sample(shape[-1:], dtype))
    h = spd(a)
    same(L.cholesky, tol, h)
    same(lambda x: L.cholesky(x, upper=True), tol, h)


def test_strided_and_broadcast_operands(o4):
    a = sample((4, 30, 30), np.float64)
    close(on_device(lambda x: L.inv(x.swapaxes(-1, -2)[::2]), a), L.inv(a.swapaxes(-1, -2)[::2]),
          1e-9)
    b = sample((30, 2), np.float64)
    close(on_device(L.solve, a, b[None]), L.solve(a, b[None]), 1e-9)
    # pending inputs: the device op waits for them like any task
    v = pending(viproc.asarray(a))
    d0 = viproc.offload_stats()["device"]
    close(L.det(v), L.det(a), 1e-9)
    assert viproc.offload_stats()["device"] == d0 + 1


def test_errors(o4):
    a = sample((3, 20, 20), np.float64)
    a[1, :, 4] = 0  # singular: an exact zero pivot
    with pytest.raises(L.LinAlgError, match="Singular matrix"):
        np.asarray(L.inv(viproc.asarray(a)))
    with pytest.raises(L.LinAlgError, match="Singular matrix"):
        np.asarray(L.solve(viproc.asarray(a), viproc.asarray(sample((3, 20, 2), np.float64))))
    viproc.init(sync_below=0, adaptive=False, level=4, device=SPEC)  # the errors raised
    # det and slogdet of a singular matrix: 0, (0, -inf) as NumPy
    close(on_device(L.det, a), L.det(a), 1e-9)
    close(on_device(L.slogdet, a), L.slogdet(a), 1e-9)
    h = spd(sample((2, 10, 10), np.float64))
    h[1, 3, 3] = -1  # not positive definite
    with pytest.raises(L.LinAlgError, match="Matrix is not positive definite"):
        np.asarray(L.cholesky(viproc.asarray(h)))
    viproc.init(sync_below=0, adaptive=False, level=4, device=SPEC)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.complex128, np.complex64])
def test_matmul(dtype, o4):
    if not o4["clblast"]:
        pytest.skip("needs CLBlast")
    if np.dtype(dtype) in (np.float64, np.complex128) and not o4["fp64"]:
        pytest.skip("needs fp64")
    tol = tolerance(dtype, 50)
    a, b = sample((3, 40, 50), dtype), sample((3, 50, 20), dtype)
    close(on_device(np.matmul, a, b), a @ b, tol)
    close(on_device(lambda x, y: x @ y, a[0], b[0]), a[0] @ b[0], tol)
    close(on_device(np.dot, a[0], b[0]), np.dot(a[0], b[0]), tol)
    close(on_device(np.matmul, a, b[0]), a @ b[0], tol)  # broadcast batch
    close(on_device(np.matmul, a[0], b), a[0] @ b, tol)
    v = sample((50,), dtype)
    close(on_device(np.matmul, a, v), a @ v, tol)  # 1-d operands
    close(on_device(np.matmul, v, b), v @ b, tol)
    close(on_device(lambda x, y: x.T @ y, b[0], b[1]), b[0].T @ b[1], tol)
    out = viproc.asarray(np.zeros((3, 40, 20), dtype))
    on_device(lambda x, y: np.matmul(x, y, out=out), a, b, want=[])
    close(out, a @ b, tol)


def test_compositions_on_device(o4):
    if not o4["clblast"]:
        pytest.skip("needs CLBlast")
    a = sample((30, 30), np.float64) / 30
    close(on_device(lambda x: L.matrix_power(x, 5), a, ops=3), L.matrix_power(a, 5), 1e-9)
    b, c = sample((30, 10), np.float64), sample((10, 40), np.float64)
    close(on_device(lambda x: L.multi_dot([x, b, c]), a, ops=2), L.multi_dot([a, b, c]), 1e-9)
    t = sample((3, 4, 5), np.float64)
    close(on_device(lambda x: np.tensordot(x, t, axes=([0, 1], [0, 1])), t),
          np.tensordot(t, t, axes=([0, 1], [0, 1])), 1e-9)


FFT_SHAPES = [((8, 1000), None, -1), ((8, 1000), 700, -1), ((8, 1000), 1500, -1),
              ((3, 999), None, -1), ((5, 97), 101, -1), ((64, 6), None, 0), ((2, 3, 64), 40, 1)]


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64, np.float64, np.float32])
@pytest.mark.parametrize("shape,n,axis", FFT_SHAPES)
def test_fft(dtype, shape, n, axis, o4):
    double_loop = True  # norm=None: NumPy's double loop for every dtype
    if not o4["fp64"] and double_loop:
        pytest.skip("needs fp64")
    x = sample(shape, dtype)
    tol = tolerance(np.float32 if np.dtype(dtype).itemsize <= 8 and
                    np.dtype(dtype) != np.float64 else np.float64, 10)
    for norm in (None, "ortho", "forward"):
        if np.dtype(dtype).kind == "c":
            for f in (np.fft.fft, np.fft.ifft, np.fft.irfft, np.fft.hfft):
                close(on_device(lambda v, f=f: f(v, n=n, axis=axis, norm=norm), x),
                      f(x, n=n, axis=axis, norm=norm), tol)
        else:
            for f in (np.fft.rfft, np.fft.ihfft):
                close(on_device(lambda v, f=f: f(v, n=n, axis=axis, norm=norm), x),
                      f(x, n=n, axis=axis, norm=norm), tol)


def test_fft_nd(o4):
    x = sample((16, 24, 10), np.complex128)
    r = sample((16, 24, 10), np.float64)
    close(on_device(np.fft.fft2, x, ops=2), np.fft.fft2(x), 1e-12)
    close(on_device(np.fft.ifftn, x, ops=3), np.fft.ifftn(x), 1e-12)
    close(on_device(np.fft.rfftn, r, ops=3), np.fft.rfftn(r), 1e-12)
    y = np.fft.rfftn(r)
    close(on_device(lambda v: np.fft.irfftn(v, s=r.shape, axes=(0, 1, 2)), y, ops=3),
          np.fft.irfftn(y, s=r.shape, axes=(0, 1, 2)), 1e-12)


def test_small_work_stays_on_cpu(o4):
    """Below device_min_work the CPU kernel runs: NumPy's bits."""
    start(min_work=1 << 40)
    try:
        a = sample((50, 50), np.float64)
        x = sample((4, 256), np.complex128)
        d0 = viproc.offload_stats()["device"]
        for got, want in [(L.inv(viproc.asarray(a)), L.inv(a)),
                          (viproc.asarray(a) @ viproc.asarray(a), a @ a),
                          (np.fft.fft(viproc.asarray(x)), np.fft.fft(x))]:
            assert np.array_equal(np.asarray(got), want)
        assert viproc.offload_stats()["device"] == d0
    finally:
        start()


@pytest.mark.parametrize("level", [1, 2])
def test_lower_levels_never_offload(level):
    """A device is used from O4 on only; O1 stays bit-identical."""
    start(level=level)
    try:
        a = sample((60, 60), np.float64)
        d0 = viproc.offload_stats()["device"]
        got = L.inv(viproc.asarray(a))
        if level == 1:
            assert np.array_equal(np.asarray(got), L.inv(a))
        np.asarray(np.fft.fft(viproc.asarray(a)))
        assert viproc.offload_stats()["device"] == d0
        assert not viproc.device_info()["selected"]
    finally:
        start()


def test_no_device_selected():
    start(device="none")
    try:
        info = viproc.device_info()
        assert not info["selected"] and info["reason"]
        a = sample((60, 60), np.float64)
        d0 = viproc.offload_stats()["device"]
        close(L.inv(viproc.asarray(a)), L.inv(a), 1e-12)
        assert viproc.offload_stats()["device"] == d0
    finally:
        start()
