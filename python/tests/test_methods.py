"""ndarray methods on viproc arrays: every method of numpy.ndarray exists,
gives NumPy's result, and those whose function form runs asynchronously take
the asynchronous path (hokb/viproc#25, step 1)."""

import io
import operator
import pickle

import numpy as np
import pytest

import viproc
from async_path import SIZE, assert_async, deterministic, pending

rng = np.random.default_rng(1)

# Attributes of numpy.ndarray viproc does not offer: `base` (viproc does not
# track which Python object a view came from) and `ctypes` (a raw pointer to
# runtime memory would bypass the runtime).
NOT_OFFERED = {"base", "ctypes"}


def same(r, expected):
    if isinstance(expected, tuple):
        assert isinstance(r, tuple) and len(r) == len(expected)
        for a, b in zip(r, expected):
            same(a, b)
        return
    got = np.asarray(r)
    expected = np.asarray(expected)
    assert got.dtype == expected.dtype and got.shape == expected.shape
    assert np.array_equal(got, expected, equal_nan=True)


def pair(shape=(6, 7), dtype=np.float64):
    x = (rng.standard_normal(shape) * 10).astype(dtype)
    return x, viproc.asarray(x)


def test_every_ndarray_attribute_exists():
    missing = [name for name in dir(np.ndarray)
               if not name.startswith("_") and name not in NOT_OFFERED
               and not any(name in k.__dict__ for k in type(viproc.asarray(np.ones(3))).__mro__)]
    assert missing == []


METHODS = [
    ("conj", lambda a: a.conj()),
    ("conjugate", lambda a: a.conjugate()),
    ("round", lambda a: a.round()),
    ("round2", lambda a: a.round(2)),
    ("clip", lambda a: a.clip(-3, 4)),
    ("clip_min", lambda a: a.clip(min=0)),
    ("cumsum", lambda a: a.cumsum()),
    ("cumsum_axis", lambda a: a.cumsum(axis=1)),
    ("cumprod", lambda a: a.cumprod(axis=0)),
    ("trace", lambda a: a.trace()),
    ("trace_offset", lambda a: a.trace(1)),
    ("nonzero", lambda a: (a > 0).nonzero()),
    ("argsort", lambda a: a.argsort()),
    ("argsort_axis0_stable", lambda a: a.argsort(axis=0, kind="stable")),
    ("argpartition", lambda a: a.argpartition(3)),
    ("searchsorted", lambda a: a[0].copy().reshape(-1)),
    ("take", lambda a: a.take([0, 5, -1])),
    ("take_axis", lambda a: a.take([1, 0], axis=1)),
    ("choose", lambda a: (a > 0).astype(np.int64).choose([a, -a])),
    ("compress", lambda a: a.compress([True, False, True], axis=0)),
    ("repeat", lambda a: a.repeat(2, axis=1)),
    ("byteswap", lambda a: a.byteswap()),
    ("getfield", lambda a: a.getfield(np.float64)),
    ("tobytes", lambda a: np.frombuffer(a.tobytes(), a.dtype)),
    ("tobytes_F", lambda a: np.frombuffer(a.tobytes("F"), a.dtype)),
    ("data", lambda a: np.frombuffer(bytes(a.data), a.dtype) if a.flags.c_contiguous
     else np.zeros(0)),
    ("flat_get", lambda a: a.flat[3]),
    ("flat_slice", lambda a: a.flat[2:9]),
    ("flat_iter", lambda a: np.array(list(a.flat))),
    ("flat_len", lambda a: len(a.flat)),
    ("dumps", lambda a: pickle.loads(a.dumps())),
]


@pytest.mark.parametrize("call", [c for _, c in METHODS], ids=[n for n, _ in METHODS])
@pytest.mark.parametrize("layout", ["c", "t", "step"])
def test_method_gives_numpys_result(call, layout):
    x, v = pair()
    if layout == "t":
        x, v = x.T, v.T
    elif layout == "step":
        x, v = x[::2, ::-1], v[::2, ::-1]
    same(call(v), call(x))


def test_searchsorted():
    x = np.sort(rng.standard_normal(50))
    q = rng.standard_normal(20)
    v = viproc.asarray(x)
    same(v.searchsorted(q), x.searchsorted(q))
    same(v.searchsorted(viproc.asarray(q), side="right"), x.searchsorted(q, side="right"))


@pytest.mark.parametrize("layout", ["c", "t"])
def test_in_place_methods(layout):
    x, v = pair()
    if layout == "t":
        x, v = x.T, v.T
    before = x.copy()
    snapshot = np.asarray(v)  # an export alive: the writes copy-on-write
    x.sort(axis=0)
    assert v.sort(axis=0) is None
    same(v, x)
    x.partition(2, axis=1)
    v.partition(2, axis=1)
    same(v, x)
    x.put([0, 3], [7.5, -1.0])
    v.put([0, 3], [7.5, -1.0])
    same(v, x)
    x.fill(2.25)
    v.fill(2.25)
    same(v, x)
    x.flat[[1, 4]] = 9.0
    v.flat[[1, 4]] = 9.0
    same(v, x)
    x.setfield(3.0, np.float64)
    v.setfield(3.0, np.float64)
    same(v, x)
    same(snapshot, before)


def test_fill_casts_as_numpy_and_rejects_sequences():
    x, v = pair(dtype=np.int32)
    x.fill(3.7)
    v.fill(3.7)
    same(v, x)
    with pytest.raises(ValueError):
        np.ones(3).fill([1, 2, 3])
    with pytest.raises(ValueError):
        viproc.asarray(np.ones(3)).fill([1, 2, 3])


def test_resize_raises_numpys_error():
    v = viproc.asarray(np.arange(6.0))
    with pytest.raises(ValueError):
        v.resize(3)


def test_flags_without_sync():
    x, v = pair((4, 5))
    for a, b in [(x, v), (x.T, v.T), (x[::2], v[::2]), (x[:, 1], v[:, 1])]:
        for name in ("c_contiguous", "f_contiguous", "contiguous", "aligned", "writeable"):
            assert getattr(b.flags, name) == getattr(a.flags, name), name
        for key in ("C_CONTIGUOUS", "F", "W", "WRITEABLE", "A"):
            assert b.flags[key] == a.flags[key], key
    assert "WRITEABLE : True" in repr(v.flags)
    v.flags.writeable = False
    assert not v.flags.writeable and "WRITEABLE : False" in repr(v.flags)
    with pytest.raises(ValueError):
        v[0, 0] = 1.0
    with pytest.raises(ValueError):
        v.setflags(write=True)
    w = viproc.asarray(np.ones(3))
    w.setflags(write=False)
    with pytest.raises(ValueError):
        w.fill(0.0)
    assert not np.broadcast_to(viproc.asarray(np.ones(3)), (2, 3)).flags.writeable


def test_to_device_and_files(tmp_path):
    x, v = pair()
    assert v.device == x.device == "cpu"
    assert v.to_device("cpu") is v
    with pytest.raises(ValueError):
        v.to_device("gpu")
    f = tmp_path / "a.bin"
    v.tofile(str(f))
    same(np.fromfile(str(f)).reshape(x.shape), x)
    buf = io.BytesIO()
    v.dump(buf)
    buf.seek(0)
    same(pickle.load(buf), x)


# --- asynchronous path ------------------------------------------------------------


@pytest.fixture
def det():
    with deterministic():
        yield


def big(dtype=np.float64):
    x = (rng.random(SIZE) + 0.5).astype(dtype)
    return x, pending(viproc.asarray(x))


def test_conj_is_an_async_ufunc(det):
    x = (rng.random(SIZE) + 1j * rng.random(SIZE))
    v = pending(viproc.asarray(x))
    same(assert_async(v.conj), x.conj())
    same(assert_async(v.conjugate), x.conjugate())


@pytest.mark.parametrize("dtype", [bool, np.int32, np.float64])
def test_conj_of_real_data_is_the_array(dtype):
    """NumPy's conj()/conjugate() return a real array itself (no copy)."""
    x = np.ones((3, 4), dtype)
    v = viproc.asarray(x)
    assert x.conj() is x and v.conj() is v and v.conjugate() is v
    assert v[:, ::2].conj().shape == (3, 2)


def test_fill_is_an_assignment_task(det):
    x, v = big()
    assert_async(v.fill, 1.5)
    x.fill(1.5)
    same(v, x)


def test_metadata_methods_issue_nothing(det):
    x, v = big()
    assert assert_async(lambda: v.flags.c_contiguous, ops=0) is True
    assert assert_async(lambda: v.device, ops=0) == "cpu"
    assert assert_async(v.to_device, "cpu", ops=0) is v


@pytest.mark.parametrize("call, ops", [
    (lambda a: a.round(), 1), (lambda a: a.round(2), 3), (lambda a: a.clip(0.7, 1.2), 1),
    (lambda a: a.clip(max=1.0), 1),
])
def test_composed_methods_take_async_path(det, call, ops):
    x, v = big()
    same(assert_async(call, v, ops=ops), call(x))


def test_trace_takes_async_path(det):
    x = rng.random((3000, 3000))
    v = pending(viproc.asarray(x))
    same(assert_async(v.trace), x.trace())


# Methods whose function form still falls back: each step of #25 that makes
# the function asynchronous turns its entry green (strict xfail).
FALLING_BACK = [
    ("cumsum", lambda a: a.cumsum(), 1),
    ("cumprod", lambda a: a.cumprod(), 1),
    ("take", lambda a: a.take([0, 5, 9]), 1),
    ("repeat", lambda a: a.repeat(2), 1),
    ("compress", lambda a: a.compress(np.arange(SIZE) % 2 == 0), 1),
    ("argsort", lambda a: a.argsort(), 1),
    ("argpartition", lambda a: a.argpartition(5), 1),
    ("searchsorted", lambda a: a.searchsorted(0.9), 1),
    ("sort", lambda a: a.sort(), 1),
    ("partition", lambda a: a.partition(5), 1),
    ("put", lambda a: a.put([1, 2], [0.0, 0.0]), 1),
]


@pytest.mark.xfail(strict=True, reason="the function form falls back (hokb/viproc#25)")
@pytest.mark.parametrize("call, ops", [(c, o) for _, c, o in FALLING_BACK],
                         ids=[n for n, _, _ in FALLING_BACK])
def test_method_takes_async_path(det, call, ops):
    x, v = big()
    assert_async(call, v, ops=ops)
