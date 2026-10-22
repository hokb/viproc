"""numpy.fft on NumPy's pocketfft gufuncs (hokb/viproc#25 step 6c):
bit-identical to NumPy (values, dtypes, layouts, errors) and on the
asynchronous path."""

import numpy as np
import pytest

import viproc
from async_path import assert_async, pending
from test_compositions import DTYPES, check, det, same, sample  # noqa: F401 (det: fixture)
from test_joining import path  # noqa: F401 (fixture)

F = np.fft
ONE_D = [F.fft, F.ifft, F.rfft, F.irfft, F.hfft, F.ihfft]
N_D = [F.fftn, F.ifftn, F.fft2, F.ifft2, F.rfftn, F.irfftn, F.rfft2, F.irfft2]


@pytest.mark.parametrize("shape", [(16,), (5,), (6, 8), (3, 4, 10), (2, 0, 4)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_one_d(shape, dtype, path):
    x = sample(shape, dtype, special=False)
    for f in ONE_D:
        check(f, x)
        check(lambda a: f(a, n=7), x)
        check(lambda a: f(a, n=12, axis=0), x)
        for norm in ("backward", "ortho", "forward"):
            check(lambda a: f(a, norm=norm), x)
        if len(shape) > 1:
            check(lambda a: f(a, axis=-2), x)
            check(lambda a: f(a, n=3, axis=0, norm="ortho"), x)


@pytest.mark.parametrize("dtype", DTYPES)
def test_n_d(dtype, path):
    x = sample((4, 6, 5), dtype, special=False)
    for f in N_D:
        check(f, x)
        check(lambda a: f(a, axes=(0, 2)), x)
        check(lambda a: f(a, s=(4, 3), axes=(2, 0)), x)
        check(lambda a: f(a, norm="ortho"), x)
        check(lambda a: f(a, s=(3, -1), axes=(1, 2), norm="forward"), x)


def test_errors(path):
    x = sample((8,), np.float64)
    check(lambda a: F.fft(a, n=0), x)  # ValueError
    check(lambda a: F.fft(a, norm="x"), x)
    check(lambda a: F.irfft(a, norm="x"), x)
    check(lambda a: F.fft(a, axis=3), x)  # AxisError
    check(F.fft, np.ones(()))  # IndexError
    check(lambda a: F.fftn(a, s=(2, 3), axes=(0,)), x)
    check(lambda a: F.rfft(a), x.astype(np.complex128))  # NumPy's TypeError


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.float32, np.complex64])
def test_out(dtype, path):
    x = sample((6, 8), dtype, special=False)
    for f, n_out, od in [(F.fft, 8, np.result_type(dtype, 1j)),
                         (F.rfft, 5, np.result_type(dtype, 1j)),
                         (F.irfft, 14, np.result_type(np.finfo(dtype).dtype, 1.0))]:
        if f is F.rfft and np.dtype(dtype).kind == "c":
            continue
        for norm in ("backward", "ortho"):
            o = np.zeros((6, n_out), od)
            vo = viproc.asarray(o)
            r = f(x, norm=norm, out=o)
            vr = f(viproc.asarray(x), norm=norm, out=vo)
            assert r is o and vr is vo
            same(vo, o)
    with pytest.raises(ValueError):
        F.fft(viproc.asarray(x), out=viproc.asarray(np.zeros((6, 7), np.complex128)))


ASYNC = [  # name, call, shape, dtype, tasks
    ("fft", F.fft, (100, 64), "f8", 2),  # the input cast to complex128 first
    ("fft_c", F.fft, (100, 64), "c16", 1),
    ("fft_axis0", lambda a: F.fft(a, axis=0), (64, 100), "c16", 1),
    ("ifft_n", lambda a: F.ifft(a, n=100), (100, 64), "c16", 1),
    ("rfft", F.rfft, (100, 64), "f8", 1),
    ("rfft_odd", lambda a: F.rfft(a, n=63), (100, 64), "f8", 1),
    ("irfft", F.irfft, (100, 64), "c16", 1),
    ("fft_ortho", lambda a: F.fft(a, norm="ortho"), (100, 64), "c16", 1),
    ("fft_f4", F.fft, (100, 64), "f4", 3),  # cast in, double loop, cast back
    ("fft_c8_ortho", lambda a: F.fft(a, norm="ortho"), (100, 64), "c8", 1),
    ("hfft", F.hfft, (100, 64), "c16", 2),  # conjugate, irfft
    ("ihfft", F.ihfft, (100, 64), "f8", 2),  # rfft, conjugate in place
    ("fft2", F.fft2, (100, 64), "c16", 2),
    ("rfftn", F.rfftn, (10, 20, 32), "f8", 3),
    ("irfft2", F.irfft2, (100, 33), "c16", 2),
]


@pytest.mark.parametrize("call,shape,dtype,ops", [c[1:] for c in ASYNC],
                         ids=[c[0] for c in ASYNC])
def test_async(call, shape, dtype, ops, det):
    x = sample(shape, dtype, special=False)
    v = pending(viproc.asarray(x))
    got = assert_async(call, v, ops=ops)
    same(got, call(x))
