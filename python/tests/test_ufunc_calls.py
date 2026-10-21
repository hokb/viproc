"""ufunc calls the C fast path leaves to _ufuncs.call (hokb/viproc#25 step
5b): ufuncs with several outputs (divmod, modf, frexp), dtype=, signature=,
casting= and out= arrays of another dtype. Bit-identical to NumPy (values,
dtypes, layouts, errors) and on the asynchronous path."""

import warnings

import numpy as np
import pytest

import viproc
from async_path import SIZE, assert_async
from test_compositions import DTYPES, big, check, det, same, sample  # noqa: F401 (det: fixture)
from test_joining import path  # noqa: F401 (fixture)

SHAPES = [(7,), (4, 5), (3, 4, 2), (3000,)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_several_outputs(shape, dtype, path):
    x, y = sample(shape, dtype), sample(shape, dtype)
    for f in (np.divmod, np.modf, np.frexp):
        if f.nin == 2:
            check(f, x, y)
            check(lambda a: f(a, 3), x)
            check(lambda a: f(a, 2.5), x)
            check(lambda a: f(7, a), x)
            check(lambda a: f(a, a[..., :1]), x)  # broadcasting
            check(lambda a: divmod(a, 4), x)
        else:
            check(f, x)
            check(lambda a: f(a[..., ::-1]), x)


@pytest.mark.parametrize("dtype", DTYPES)
def test_dtype_signature_casting(dtype, path):
    x, y = sample((4, 5), dtype), sample((4, 5), np.int32)
    for d in DTYPES:
        check(lambda a, b: np.add(a, b, dtype=d), x, y)
        check(lambda a: np.multiply(a, 3, dtype=d), x)
        check(lambda a: np.add(a, 1.5, dtype=d, casting="unsafe"), x)
        check(lambda a, b: np.subtract(a, b, signature=(None, None, d)), x, y)
        check(lambda a, b: np.maximum(a, b, signature=(d, d, d), casting="unsafe"), x, y)
        check(lambda a: np.sqrt(a, dtype=d), x)
        check(lambda a: np.divmod(a, 3, dtype=d), x)
    check(lambda a, b: np.add(a, b, casting="no"), x, y)
    check(lambda a, b: np.add(a, b, casting="equiv"), x, y)
    check(lambda a, b: np.true_divide(a, b, casting="safe", dtype=np.int64), x, y)
    check(lambda a, b: np.add(a, b, dtype=np.float32, signature=(None, None, None)), x, y)
    check(lambda a: np.add(a, 300, dtype=np.int8), x)  # int8: not a viproc dtype
    check(lambda a: np.add(a, 2**40, dtype=np.int32), x)  # NumPy's OverflowError


@pytest.mark.parametrize("dtype", DTYPES)
def test_out_of_another_dtype(dtype, path):
    x, y = sample((4, 5), dtype), sample((4, 5), np.float32)
    for od in DTYPES:
        for casting in ("same_kind", "unsafe"):
            for layout in ("c", "f", "step"):
                out = np.zeros((4, 5) if layout != "step" else (8, 5), od)
                out = np.asfortranarray(out) if layout == "f" else out
                vout = viproc.asarray(out)
                if layout == "step":
                    out, vout = out[::2], vout[::2]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        np.add(x, y, out=out, casting=casting)
                except TypeError:
                    with pytest.raises(TypeError):
                        np.add(viproc.asarray(x), viproc.asarray(y), out=vout, casting=casting)
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = np.add(viproc.asarray(x), viproc.asarray(y), out=vout, casting=casting)
                    assert r is vout
                    same(vout, out)
                    np.absolute(x, out=out, casting=casting)
                    np.absolute(viproc.asarray(x), out=(vout,), casting=casting)
                    same(vout, out)
    # inputs broadcast to out's shape
    out = np.zeros((3, 4, 5), np.float32)
    vout = viproc.asarray(out)
    np.add(x, 1, out=out, casting="unsafe")
    np.add(viproc.asarray(x), 1, out=vout, casting="unsafe")
    same(vout, out)


def test_in_place_with_cast():
    x = sample((60, 50), np.int64)
    v = viproc.asarray(x)
    np.multiply(x, 1.5, out=x, casting="unsafe")
    np.multiply(v, 1.5, out=v, casting="unsafe")
    same(v, x)
    r = v[::2]
    np.add(v[::2], 0.5, out=r, casting="unsafe")  # aliasing view
    np.add(x[::2], 0.5, out=x[::2], casting="unsafe")
    same(v, x)


ASYNC = [  # name, call, inputs (shape, dtype), tasks
    ("divmod", np.divmod, [((SIZE,), "f8"), ((SIZE,), "f8")], 1),
    ("divmod_int_scalar", lambda a: divmod(a, 7), [((SIZE,), "i8")], 1),
    ("modf", np.modf, [((SIZE,), "f8")], 1),
    ("modf_strided", lambda a: np.modf(a[::2].T), [((100, 100), "f4")], 1),
    ("frexp", np.frexp, [((SIZE,), "f8")], 1),
    # NumPy casts short 1-d inputs into copies first (check_for_trivial_loop:
    # a cast task each), n-d ones in its iterator's buffers (no task)
    ("divmod_cast", np.divmod, [((SIZE,), "f8"), ((SIZE,), "i4")], 2),
    ("divmod_cast_2d", np.divmod, [((100, 60), "f8"), ((100, 60), "i4")], 1),
    ("add_dtype", lambda a, b: np.add(a, b, dtype=np.float32), [((SIZE,), "f8"), ((SIZE,), "f8")],
     3),
    ("add_dtype_2d", lambda a, b: np.add(a, b, dtype=np.float32),
     [((100, 60), "f8"), ((100, 60), "f8")], 1),
    ("multiply_signature", lambda a: np.multiply(a, 2, signature=(None, None, np.int64)),
     [((SIZE,), "i4")], 2),
    ("sqrt_dtype", lambda a: np.sqrt(a, dtype=np.float32), [((60, 100), "f8")], 1),
]


@pytest.mark.parametrize("call, inputs, tasks", [c[1:] for c in ASYNC], ids=[c[0] for c in ASYNC])
def test_async_path(det, call, inputs, tasks):
    pairs = [big(shape, dtype) for shape, dtype in inputs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = call(*(x for x, _ in pairs))
        got = assert_async(call, *(v for _, v in pairs), ops=tasks)
    same(got, want)


def test_cast_into_out_takes_async_path(det):
    x, v = big((SIZE,), np.float64)
    out = np.zeros(SIZE, np.float32)
    vout = viproc.asarray(out)
    viproc.wait_all()
    np.add(x, 1, out=out)
    assert assert_async(lambda a: np.add(a, 1, out=vout), v) is vout
    same(vout, out)


def test_complex_warnings_every_call():
    """NumPy warns once per complex-to-real cast it sets up, on every call."""
    x = sample((40, 50), np.complex128)
    for v, o in [(x, np.zeros((40, 50))), (viproc.asarray(x), viproc.asarray(np.zeros((40, 50))))]:
        for _ in range(3):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                np.add(v, 1, out=o, casting="unsafe")
                np.add(v, v, dtype=np.float64, casting="unsafe")
            assert [type(m.message) for m in w] == [np.exceptions.ComplexWarning] * 3
            assert w[0].filename == __file__
