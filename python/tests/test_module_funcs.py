"""np.where in C (select or NumPy's own where on borrowed views) and the
module's ufunc proxies (viproc.add, ...: NumPy's ufunc without its override
dispatch): results, dtypes and errors as NumPy's."""

import pickle

import numpy as np
import pytest

import viproc

DTYPES = ["?", "i4", "i8", "f4", "f8", "c8", "c16"]


def _value(rng, kind, shape, dtype):
    data = (rng.standard_normal(shape) * 4).astype(dtype)
    if kind == "viproc":
        return viproc.asarray(data), data
    if kind == "pending":
        v = viproc.asarray(np.broadcast_to(data, (300, 30) if shape == (300, 30) else shape))
        return v.copy() if v.size > 4096 else v, data
    if kind == "numpy":
        return data, data
    if kind == "list":
        return data.tolist(), data.tolist()
    if kind == "npscalar":
        return data.reshape(-1)[0], data.reshape(-1)[0]
    s = data.reshape(-1)[0].item()  # Python scalar
    return s, s


def _outcome(f, *args):
    try:
        a = np.asarray(f(*args))  # errors of ops run directly surface at the sync
    except Exception as e:  # noqa: BLE001
        return type(e)
    return a.dtype, a.shape, a.tobytes()


@pytest.mark.parametrize("big", [False, True])
def test_where_matches_numpy(big):
    rng = np.random.default_rng(11 + big)
    shapes = [(300, 30), (30,), (300, 1), ()] if big else [(4, 5), (5,), (4, 1), (), (3,)]
    kinds = ["viproc", "pending", "numpy", "list", "npscalar", "scalar"] if big else [
        "viproc", "numpy", "list", "npscalar", "scalar"]
    for trial in range(400):
        args, ref = [], []
        for i in range(3):
            kind = kinds[rng.integers(len(kinds))]
            shape = shapes[rng.integers(len(shapes))]
            dtype = "?" if i == 0 and rng.random() < 0.7 else DTYPES[rng.integers(len(DTYPES))]
            v, n = _value(rng, kind, shape, dtype)
            args.append(v)
            ref.append(n)
        expected = _outcome(np.where, *ref)
        for where in (np.where, viproc.where):
            assert _outcome(where, *args) == expected, (trial, [type(a) for a in args])


def test_where_small_ready_runs_without_tasks_or_fallbacks():
    c = viproc.asarray(np.arange(20) % 3 == 0)
    x = viproc.asarray(np.arange(20.0))
    viproc.wait_all()
    f0 = viproc._ndarray.fallback_calls
    s0 = viproc.offload_stats()
    for where in (np.where, viproc.where):
        r = where(c, x, -1.0)
        assert isinstance(r, viproc.ndarray) and r.ready()
        assert np.array_equal(np.asarray(r), np.where(np.arange(20) % 3 == 0, np.arange(20.0), -1.0))
        r = where(c, x, np.float32(2))  # a NumPy scalar: NumPy's where on borrowed views
        assert isinstance(r, viproc.ndarray) and r.dtype == np.float64
    s1 = viproc.offload_stats()
    assert viproc._ndarray.fallback_calls == f0
    assert s1["offloaded"] == s0["offloaded"] and s1["inline"] == s0["inline"]


UFUNCS = ["add", "subtract", "multiply", "divide", "power", "maximum", "minimum", "arctan2",
          "absolute", "abs", "negative", "sin", "exp", "sqrt", "floor", "logical_and", "less",
          "equal", "isnan", "bitwise_and", "left_shift", "remainder", "floor_divide"]


def test_ufunc_proxies_match_numpy():
    rng = np.random.default_rng(5)
    kinds = ["viproc", "numpy", "scalar", "npscalar", "list"]
    shapes = [(4, 5), (5,), ()]
    for name in UFUNCS:
        uf, proxy = getattr(np, name), getattr(viproc, name)
        assert proxy == uf and isinstance(proxy, np.ufunc) and proxy.nin == uf.nin
        assert pickle.loads(pickle.dumps(proxy)) is uf
        for trial in range(60):
            args, ref = [], []
            for _ in range(uf.nin):
                v, n = _value(rng, kinds[rng.integers(len(kinds))], shapes[rng.integers(3)],
                              DTYPES[rng.integers(len(DTYPES))])
                args.append(v)
                ref.append(n)
            with np.errstate(all="ignore"):
                expected = _outcome(uf, *ref)
                got = _outcome(proxy, *args)
                # viproc raises a failed op's error as a subclass of NumPy's
                # (AsyncValueError for a negative integer exponent).
                assert got == expected or (isinstance(expected, type) and isinstance(got, type)
                                           and issubclass(got, expected)), (name, trial)
    x = viproc.asarray(np.arange(6.0))
    out = viproc.asarray(np.zeros(6))
    assert viproc.add(x, 1.0, out=out) is out
    assert np.array_equal(np.asarray(out), np.arange(6.0) + 1)
    assert float(viproc.add.reduce(x)) == 15.0
    assert np.array_equal(np.asarray(viproc.multiply.outer(x, x)), np.multiply.outer(
        np.arange(6.0), np.arange(6.0)))
    assert np.asarray(viproc.sqrt(x, dtype=np.float32)).dtype == np.float32


REDUCTIONS = ["sum", "prod", "max", "min", "amax", "amin", "any", "all", "mean", "var", "std",
              "argmax", "argmin"]


def _reduction_kwargs(rng, name, ndim):
    kw = {}
    axes = [None] + list(range(-ndim, ndim)) + ([(0, ndim - 1)] if ndim > 1 else [])
    if name not in ("argmax", "argmin") or rng.random() < 0.5:
        kw["axis"] = axes[rng.integers(len(axes))]
        if isinstance(kw["axis"], tuple) and name in ("argmax", "argmin"):
            kw["axis"] = 0 if ndim else None
    if rng.random() < 0.3:
        kw["keepdims"] = bool(rng.integers(2))
    if name in ("sum", "prod", "mean") and rng.random() < 0.2:
        kw["dtype"] = ["f8", "i8", "c16"][rng.integers(3)]
    if name in ("sum", "prod", "max", "min", "amax", "amin") and rng.random() < 0.2:
        kw["initial"] = 3
    if name in ("var", "std") and rng.random() < 0.3:
        kw["ddof"] = 1
    return kw


def test_reduction_proxies_match_numpy():
    """viproc.sum, ... (module mode): NumPy's function without its
    __array_function__ dispatch; results, dtypes and errors as NumPy's."""
    rng = np.random.default_rng(9)
    kinds = ["viproc", "pending", "numpy", "list", "scalar"]
    shapes = [(4, 5), (5,), (), (300, 30)]
    for name in REDUCTIONS:
        f, proxy = getattr(np, name), getattr(viproc, name)
        assert proxy == f and isinstance(proxy, type(f)) and proxy.__name__ == f.__name__
        assert pickle.loads(pickle.dumps(proxy)) is f
        for trial in range(40):
            shape = shapes[rng.integers(len(shapes))]
            kind = kinds[rng.integers(len(kinds))]
            v, n = _value(rng, kind, shape, DTYPES[rng.integers(len(DTYPES))])
            kw = _reduction_kwargs(rng, name, len(shape))
            if rng.random() < 0.15:
                args = (v, kw.pop("axis", None))  # the axis as a positional argument
                ref = (n, args[1])
            else:
                args, ref = (v,), (n,)
            with np.errstate(all="ignore"):
                expected = _outcome(lambda *a: f(*a, **kw), *ref)
                got = _outcome(lambda *a: proxy(*a, **kw), *args)
            assert got == expected, (name, trial, kind, shape, kw)


def test_reduction_proxies_take_the_c_path():
    from async_path import SIZE, assert_async, deterministic, pending

    with deterministic():
        data = np.arange(SIZE, dtype=np.float64).reshape(50, -1)
        x = viproc.asarray(data)
        r = assert_async(viproc.sum, pending(x), axis=0)
        assert np.array_equal(np.asarray(r), data.sum(axis=0))
        r = assert_async(viproc.max, pending(x))
        assert float(r) == data.max()
        small = viproc.asarray(np.arange(10.0))
        viproc.wait_all()
        f0 = viproc._ndarray.fallback_calls
        assert viproc.sum(small) == 45.0 and viproc.mean(small, axis=0) == 4.5
        assert viproc._ndarray.fallback_calls == f0
