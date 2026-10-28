"""Indexing without syncs: integer array indices (gather / scatter tasks),
boolean masks, scalar indexing. Results, layouts and errors follow NumPy."""

import numpy as np
import pytest

import viproc


def fallbacks():
    return viproc._ndarray.fallback_calls


def strides_of_real_axes(a):
    # Strides of extent-1 axes carry no information; NumPy's choice for them
    # depends on how a key is split into basic and advanced parts.
    return tuple(s for s, n in zip(a.strides, a.shape) if n != 1)


def random_key(rng, shape):
    keys, ax = [], 0
    while ax < len(shape) and rng.random() < 0.8:
        r, m = rng.random(), shape[ax]
        if r < 0.3:
            keys.append(slice(None, None, int(rng.choice([1, 2, -1]))))
        elif r < 0.5:
            size = tuple(int(s) for s in rng.integers(1, 4, size=int(rng.integers(0, 3))))
            keys.append(rng.integers(-m, m, size=size))
        elif r < 0.6:
            keys.append(int(rng.integers(-m, m)))
        elif r < 0.7:
            keys.append(None)
            continue
        elif r < 0.8:
            keys.append(rng.random(m) < 0.5)
        else:
            keys.append(list(rng.integers(0, m, size=2)))
        ax += 1
    if rng.random() < 0.2:
        keys.insert(int(rng.integers(0, len(keys) + 1)), Ellipsis)
    return tuple(keys) if len(keys) != 1 or rng.random() < 0.5 else keys[0]


@pytest.mark.parametrize("seed", range(3))
def test_advanced_indexing_matches_numpy(seed):
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(400):
        shape = tuple(int(s) for s in rng.integers(1, 6, size=int(rng.integers(1, 4))))
        dtype = str(rng.choice(["f8", "i4", "c8", "?"]))
        x = (rng.standard_normal(shape) * 10).astype(dtype)
        k = int(rng.integers(3))

        def layout(b):
            return [b, b.T, b[..., ::-1]][k]

        a, v = layout(x), layout(viproc.asarray(x))
        key = random_key(rng, a.shape)
        try:
            expected = a[key]
        except IndexError:
            with pytest.raises(IndexError):
                np.asarray(v[key])  # at the latest at the sync point
            continue
        got = v[key]
        if isinstance(expected, np.generic):
            assert np.asarray(got) == expected
            continue
        g = np.asarray(got)
        assert g.shape == expected.shape, key
        assert strides_of_real_axes(g) == strides_of_real_axes(expected), key
        assert np.array_equal(g, expected), key
        # The same key on the left-hand side.
        eshape = expected.shape
        values = (rng.standard_normal(eshape[int(rng.integers(0, len(eshape) + 1)):])
                  * 10).astype(dtype)
        y, w = x.copy(), viproc.asarray(x.copy())
        layout(y)[key] = values
        layout(w)[key] = values
        assert np.array_equal(np.asarray(w), y), key
        checked += 1
    assert checked > 250


def test_integer_array_indexing_runs_asynchronously():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((300, 200))
    v = viproc.asarray(x)
    rows, cols = rng.integers(0, 300, 50), rng.integers(0, 200, 50)
    vr = viproc.asarray(rows)  # index arrays may be viproc arrays (pending)
    f0 = fallbacks()
    y = np.sin(v)  # pending
    r1 = y[vr]
    r2 = y[:, cols]
    r3 = y[vr, viproc.asarray(cols)]
    r4 = y[rows[:, None], cols]
    z = viproc.asarray(np.zeros((300, 200)))
    z[vr] = y[vr] * 2.0
    z[rows, cols] += 1.0
    assert fallbacks() == f0
    sx = np.sin(x)
    zx = np.zeros((300, 200))
    zx[rows] = sx[rows] * 2.0
    zx[rows, cols] += 1.0
    for got, expected in [(r1, sx[rows]), (r2, sx[:, cols]), (r3, sx[rows, cols]),
                          (r4, sx[rows[:, None], cols]), (z, zx)]:
        assert np.array_equal(np.asarray(got), expected)
        assert np.asarray(got).strides == expected.strides


def test_out_of_bounds_indices_raise_index_error():
    v = viproc.asarray(np.arange(10000.0))  # large: a gather task
    idx = viproc.asarray(np.array([1, 12000]))
    with pytest.raises(IndexError, match="index 12000 is out of bounds"):
        r = v[idx]  # raised as soon as the task failed, at the latest here:
        np.asarray(r)
    with pytest.raises(IndexError):
        np.asarray(v[[3, -10001]])
    small = viproc.asarray(np.arange(10.0))  # small and ready: NumPy, right away
    with pytest.raises(IndexError):
        small[[1, 12]]
    with pytest.raises(IndexError, match="out of bounds for axis 0 with size 10"):
        small[10]


def test_repeated_indices_last_write_wins_as_in_numpy():
    x = np.zeros(5)
    v = viproc.asarray(x)
    idx = np.array([1, 3, 1, 1])
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    x[idx] = vals
    v[idx] = vals
    assert np.array_equal(np.asarray(v), x)
    x[idx] += 1.0
    v[idx] += 1.0  # NumPy: no accumulation for repeated indices
    assert np.array_equal(np.asarray(v), x)


def test_mask_assignment_of_a_scalar_needs_no_sync():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((40, 30))
    v = viproc.asarray(x)
    f0 = fallbacks()
    y = np.cos(v)
    mask = y > 0.5  # pending
    y[mask] = 0.0
    row_mask = viproc.asarray(rng.random(40) < 0.5)
    y[row_mask] = -1.0  # rows: the mask covers the leading axis
    assert fallbacks() == f0
    e = np.cos(x)
    e[e > 0.5] = 0.0
    e[np.asarray(row_mask)] = -1.0
    assert np.array_equal(np.asarray(y), e)


def test_partial_writes_after_copy_on_write_keep_the_other_elements():
    # A live export (or a pending reader) makes the write copy-on-write. The
    # array is the only view of its storage, but masked assignment and
    # scatter write only some elements: the others must come from the old
    # buffer (they were left as fresh memory before).
    rng = np.random.default_rng(8)
    x = rng.uniform(0.5, 1.5, size=(300, 100))  # large: tasks, not the direct path
    idx = np.arange(0, 300, 3)
    for write in (lambda a: a.__setitem__(a > 1.4, 0.6),
                  lambda a: a.__setitem__(idx, -1.0),
                  lambda a: a.__setitem__((idx, slice(None, None, 2)), a[idx, ::2] * 2.0)):
        e = x * 1.07
        before = e.copy()
        write(e)
        y = viproc.asarray(x) * 1.07
        exported = np.asarray(y)  # an alias of y's memory while it lives
        write(y)
        assert np.array_equal(np.asarray(y), e)
        assert np.array_equal(exported, e)
        y = viproc.asarray(x) * 1.07
        t = y + 0.0  # a (possibly still pending) task reader
        write(y)
        assert np.array_equal(np.asarray(y), e)
        assert np.array_equal(np.asarray(t), before)


def test_boolean_mask_indexing():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((20, 7))
    v = viproc.asarray(x)
    m = x[:, 0] > 0
    assert np.array_equal(np.asarray(v[v > 0]), x[x > 0])
    assert np.array_equal(np.asarray(v[m]), x[m])
    assert np.array_equal(np.asarray(v[m, 2:5]), x[m, 2:5])
    w = viproc.asarray(x.copy())
    w[w < 0] = np.arange(int((x < 0).sum()), dtype=float)
    e = x.copy()
    e[e < 0] = np.arange(int((x < 0).sum()), dtype=float)
    assert np.array_equal(np.asarray(w), e)


def test_scalar_indexing_of_pending_arrays_does_not_sync():
    x = np.arange(12.0).reshape(3, 4)
    v = viproc.asarray(x)
    y = v * 2.0
    f0 = fallbacks()
    s = y[1, 2]
    assert fallbacks() == f0
    y[1, 2] = -5.0  # s is a snapshot, as NumPy's scalar
    assert float(s) == 12.0
    assert float(y[1, 2]) == -5.0
    ready = viproc.asarray(x)
    viproc.wait_all()
    assert isinstance(ready[0, 1], np.float64)  # ready data: NumPy's scalar


def test_where_runs_as_a_select_task():
    rng = np.random.default_rng(8)
    f0 = fallbacks()
    for dx, dy in [("f8", "f8"), ("f4", "i8"), ("i4", "f4"), ("?", "c8")]:
        c = rng.random((30, 20)) < 0.5
        x = (rng.standard_normal((30, 20)) * 5).astype(dx)
        y = (rng.standard_normal(20) * 5).astype(dy)
        vc, vx, vy = viproc.asarray(c), viproc.asarray(x), viproc.asarray(y)
        for got, expected in [(np.where(vc, vx, vy), np.where(c, x, y)),
                              (np.where(vx > 0, vx, 0.5), np.where(x > 0, x, 0.5)),
                              (np.where(vc.T, vx.T, 1), np.where(c.T, x.T, 1)),
                              (np.where(vx, vx, vy), np.where(x, x, y))]:
            g = np.asarray(got)
            assert g.dtype == expected.dtype and g.strides == expected.strides
            assert np.array_equal(g, expected)
    assert fallbacks() == f0
