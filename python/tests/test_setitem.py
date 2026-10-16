"""a[key] = value through _Array's C slot: NumPy itself for small work on
ready data, issued assignments otherwise. Results and errors follow NumPy."""

import numpy as np
import pytest

import viproc

DTYPES = [np.bool_, np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]


def random_key(rng, shape):
    keys = []
    for m in shape:
        r = rng.random()
        if r < 0.4:
            keys.append(int(rng.integers(-m, m)))
        elif r < 0.7:
            keys.append(slice(int(rng.integers(0, m)), None, int(rng.choice([1, 2, -1]))))
        elif r < 0.8:
            keys.append(slice(None))
        elif r < 0.9:
            keys.append(list(rng.integers(0, m, size=2)))
        else:
            break
    if rng.random() < 0.2:
        keys.insert(int(rng.integers(0, len(keys) + 1)), Ellipsis)
    if rng.random() < 0.1:
        keys.insert(int(rng.integers(0, len(keys) + 1)), None)
    return keys[0] if len(keys) == 1 and rng.random() < 0.5 else tuple(keys)


def random_value(rng, dtype):
    r = rng.random()
    if r < 0.2:
        return int(rng.integers(-5, 5))
    if r < 0.35:
        return float(rng.normal())
    if r < 0.45:
        return complex(rng.normal(), rng.normal())
    if r < 0.5:
        return bool(rng.random() < 0.5)
    if r < 0.65:
        return np.dtype(rng.choice(DTYPES)).type(rng.integers(0, 3))
    if r < 0.75:
        return 2**40  # overflows int32: NumPy's OverflowError
    return np.asarray(rng.normal(size=())).astype(dtype)  # a 0-d array


def outcome(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - the error type is compared
        return type(e)
    return None


@pytest.mark.parametrize("state", ["ready", "pending", "read", "exported", "large"])
def test_setitem_matches_numpy(state):
    rng = np.random.default_rng({"ready": 1, "pending": 2, "read": 3, "exported": 4,
                                 "large": 5}[state])
    for trial in range(300):
        dtype = DTYPES[trial % len(DTYPES)]
        big = state in ("large", "pending") or (state == "read" and trial % 2)
        shape = (60, 70) if big else tuple(int(n) for n in rng.integers(1, 6, 2))
        x = (rng.normal(size=shape) * 3).astype(dtype)
        key = random_key(rng, shape)
        value = random_value(rng, dtype)
        a = viproc.asarray(x)
        if state == "pending":
            a = a.copy()  # large: computed by a task
        reader = a + 0 if state == "read" else None
        export = np.asarray(a) if state == "exported" else None
        before = x.copy()
        e = x.copy()
        with np.errstate(all="ignore"), _noop():
            err_np = outcome(lambda: e.__setitem__(key, value))
            err_vp = outcome(lambda: a.__setitem__(key, value))
        assert err_vp == err_np, (dtype, shape, key, value)
        assert np.array_equal(np.asarray(a), e, equal_nan=True), (dtype, shape, key, value)
        if reader is not None:
            assert np.array_equal(np.asarray(reader), before + 0, equal_nan=True)
        if export is not None:
            assert np.array_equal(export, before, equal_nan=True)


class _noop:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)


def test_setitem_with_viproc_values_and_keys():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(50, 60))
    for shape_value, key in [((60,), 3), ((), (2, 5)), ((3, 60), slice(4, 7)),
                             ((50,), (slice(None), 7)), ((60,), -1)]:
        v = rng.normal(size=shape_value)
        for pending in (False, True):
            e = x.copy()
            e[key] = v
            a = viproc.asarray(x)
            vv = viproc.asarray(v)
            if pending:
                a, vv = a * 1.0, vv * 1.0
            a[key] = vv
            assert np.array_equal(np.asarray(a), e)
    idx = viproc.asarray(np.array([1, 3, 3, 7]))
    e = x.copy()
    e[np.asarray(idx)] = 2.5
    a = viproc.asarray(x)
    a[idx] = 2.5  # a ready index array of viproc
    assert np.array_equal(np.asarray(a), e)
    small = viproc.asarray(np.arange(6.0))
    small[viproc.asarray(np.array([0, 2]))] = -1.0
    assert np.array_equal(np.asarray(small), [-1.0, 1, -1, 3, 4, 5])


def test_setitem_overlap_and_errors_as_numpy():
    x = np.arange(10.0)
    a = viproc.asarray(x)
    a[1:] = a[:-1]  # overlapping views of one storage
    x[1:] = x[:-1]
    assert np.array_equal(np.asarray(a), x)
    with pytest.raises(ValueError, match="cannot delete"):
        del a[0]
    with pytest.raises(IndexError):
        a[10] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        np.broadcast_to(a, (3, 10))[0, 0] = 1.0
    ints = viproc.asarray(np.zeros(4, np.int32))
    with pytest.raises(OverflowError):
        ints[0] = 2**40
    ints[1] = 3.9  # truncated as by NumPy
    assert np.asarray(ints)[1] == 3


def test_small_ready_assignment_runs_in_numpy_without_tasks():
    a = viproc.asarray(np.zeros(100))
    viproc.wait_all()
    before = viproc.offload_stats()
    for i in range(100):
        a[i] = i
        a[i:i + 2] = 0.5 * i
    after = viproc.offload_stats()
    assert after == before  # neither tasks nor direct ops: NumPy wrote it
    e = np.zeros(100)
    for i in range(100):
        e[i] = i
        e[i:i + 2] = 0.5 * i
    assert np.array_equal(np.asarray(a), e)


def test_recurrence_in_place_with_pending_readers_matches_numpy():
    # x[i] = f(x[i-1], x[i]) on a pending array: every write meets pending
    # readers (copy-on-write); the readers feed the write, so the copy takes
    # the old buffer over. Readers that do not feed it keep the old data.
    d = np.random.default_rng(3).random((200, 3000))
    for keep_reader in (False, True):
        x, xn = viproc.asarray(d).copy(), d.copy()
        kept, kept_n = [], []
        for i in range(1, x.shape[0]):
            if keep_reader and i % 7 == 0:
                kept.append(x * 1.0)  # an independent reader of the old data
                kept_n.append(xn * 1.0)
            x[i] = np.cos(x[i - 1]) * 0.5 + x[i] * 0.5
            xn[i] = np.cos(xn[i - 1]) * 0.5 + xn[i] * 0.5
        assert np.array_equal(np.asarray(x), xn)
        for k, kn in zip(kept, kept_n):
            assert np.array_equal(np.asarray(k), kn)
        e = np.asarray(x)  # an export keeps its values over later writes
        x[0] = -1.0
        assert np.array_equal(e, xn) and np.asarray(x)[0, 0] == -1.0
