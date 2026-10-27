"""Views computed from metadata only (no NumPy array is built for them):
basic indexing, transpositions, reshape, broadcast_to, diagonal, real/imag,
dtype views. Shapes, strides, values and errors follow NumPy."""

import numpy as np
import pytest

import viproc


def fallbacks():
    return viproc._ndarray.fallback_calls


def layouts(rng):
    """(numpy array, matching viproc array) pairs in various layouts."""
    for shape in [(), (5,), (4, 6), (3, 1, 5), (2, 3, 4), (0, 3), (3, 4, 1, 2)]:
        base = rng.standard_normal(shape)
        yield base, viproc.asarray(base)
        if base.ndim >= 2:
            f = np.asfortranarray(base)
            yield f, viproc.asarray(f)
            yield base.T, viproc.asarray(base).T
        if base.ndim >= 1 and base.shape[0] > 1:
            yield base[::2], viproc.asarray(base)[::2]
            yield base[::-1], viproc.asarray(base)[::-1]


def same(v, n):
    assert isinstance(v, viproc.ndarray), type(v)
    assert v.shape == n.shape
    assert v.dtype == n.dtype
    assert v.strides == n.strides, (v.strides, n.strides)
    np.testing.assert_array_equal(np.asarray(v), n)


def random_basic_key(rng, ndim):
    keys, ax = [], 0
    while ax < ndim and rng.random() < 0.85:
        r = rng.random()
        if r < 0.5:
            pick = lambda: None if rng.random() < 0.3 else int(rng.integers(-8, 8))  # noqa: E731
            step = None if rng.random() < 0.4 else int(rng.choice([1, 2, 3, -1, -2]))
            keys.append(slice(pick(), pick(), step))
        elif r < 0.7:
            keys.append(np.int64(0) if rng.random() < 0.3 else 0)
        elif r < 0.85:
            keys.append(None)
            continue
        else:
            keys.append(-1)
        ax += 1
    if rng.random() < 0.25:
        keys.insert(int(rng.integers(0, len(keys) + 1)), Ellipsis)
    return keys[0] if len(keys) == 1 and rng.random() < 0.5 else tuple(keys)


def test_basic_indexing_fuzz():
    rng = np.random.default_rng(1)
    for n, v in layouts(rng):
        for _ in range(60):
            key = random_basic_key(rng, n.ndim)
            try:
                expected = n[key]
            except IndexError as e:
                with pytest.raises(IndexError, match=str(e)[:20]):
                    v[key]
                continue
            got = v[key]
            if np.isscalar(expected) or not isinstance(expected, np.ndarray):
                assert got == expected
            else:
                same(got, expected)


@pytest.mark.parametrize("fn,args", [
    (np.transpose, ()), (np.transpose, ((1, 0, 2),)), (np.transpose, ((-1, 0, 1),)),
    (np.swapaxes, (0, 2)), (np.moveaxis, (0, -1)), (np.moveaxis, ((0, 1), (2, 0))),
    (np.matrix_transpose, ()), (np.squeeze, ()), (np.squeeze, (1,)),
    (np.expand_dims, (0,)), (np.expand_dims, ((1, 4),)), (np.flip, ()), (np.flip, (1,)),
    (np.fliplr, ()), (np.flipud, ()), (np.rot90, ()), (np.rot90, (2,)), (np.rot90, (3, (2, 0))),
    (np.atleast_1d, ()), (np.atleast_2d, ()), (np.atleast_3d, ()),
    (lambda a: np.broadcast_to(a, (2,) + a.shape), ()), (np.diagonal, ()), (np.diagonal, (1,)),
    (np.diagonal, (-1, 2, 0)), (np.real, ()), (np.ravel, ()), (np.ravel, ("F",)),
    (np.reshape, ((-1,),)), (np.reshape, ((3, -1), "F")), (np.reshape, ((-1,), "A")),
])
def test_view_functions(fn, args):
    rng = np.random.default_rng(2)
    base = rng.standard_normal((3, 1, 5))
    vbase = viproc.asarray(base)
    f = np.asfortranarray(base)
    for n, v in [(base, vbase), (f, viproc.asarray(f)), (base[::-1, :, ::2], vbase[::-1, :, ::2]),
                 (base.transpose(2, 1, 0), vbase.transpose(2, 1, 0))]:
        before = fallbacks()
        got = fn(v, *args)
        assert fallbacks() == before
        same(got, fn(n, *args))


def test_view_errors_are_numpys():
    v, n = viproc.asarray(np.ones((2, 3))), np.ones((2, 3))
    for fn in [lambda a: np.transpose(a, (0, 0)), lambda a: np.broadcast_to(a, (3, 3)),
               lambda a: a.reshape(4, 2), lambda a: a.reshape(-1, -1),
               lambda a: np.squeeze(a, 0), lambda a: a[2], lambda a: a[:, :, :],
               lambda a: np.swapaxes(a, 0, 5), lambda a: np.diagonal(a, 0, 1, 1)]:
        with pytest.raises(Exception) as expected:
            fn(n)
        with pytest.raises(type(expected.value), match=str(expected.value)[:25].replace("(", r"\(")
                           .replace(")", r"\)")):
            fn(v)


def test_reshape_copies_like_numpy():
    n = np.arange(24.0).reshape(4, 6)[:, :3]
    v = viproc.asarray(np.arange(24.0).reshape(4, 6))[:, :3]
    for order in "CFA":
        same(v.reshape(-1, order=order), n.reshape(-1, order=order))
    with pytest.raises(ValueError, match="Unable to avoid creating a copy"):
        np.reshape(v, (12,), copy=False)
    r = np.reshape(viproc.asarray(n.copy()), (12,), copy=True)
    same(r, np.reshape(n.copy(), (12,), copy=True))


def test_ravel_copies_like_numpy():
    """ravel views only contiguous data (PyArray_Ravel), unlike reshape: a
    strided 1-d array is copied, which decides the strides of later calls."""
    base = np.arange(24.0).reshape(4, 6)
    vbase = viproc.asarray(base)
    for key in [(slice(None, None, -3),), (slice(None), slice(None, None, 2)), (slice(None),),
                (slice(None, None, -1), slice(None))]:
        for t in (False, True):
            n, v = base[key], vbase[key]
            n, v = (n.T, v.T) if t else (n, v)
            for x, y in [(n, v), (n.ravel(), v.ravel())]:
                for order in "CFAK":
                    want = np.ravel(x, order)
                    got = np.ravel(y, order)
                    same(got, want)
                    assert (np.shares_memory(want, base)
                            == np.shares_memory(np.asarray(got), np.asarray(vbase)))


def test_views_share_storage_and_readonly():
    v = viproc.asarray(np.arange(12.0).reshape(3, 4))
    t = v.T
    t[0, 1] = 100.0
    assert float(v[1, 0]) == 100.0
    b = np.broadcast_to(v[0], (2, 4))
    with pytest.raises(ValueError):
        b[0, 0] = 1.0
    d = np.diagonal(v)
    with pytest.raises(ValueError):
        d[0] = 1.0


def test_complex_parts_and_dtype_views():
    n = (np.arange(6.0) + 1j * np.arange(6.0)[::-1]).reshape(2, 3)
    v = viproc.asarray(n)
    same(v.real, n.real)
    same(v.imag, n.imag)
    same(v.T.imag, n.T.imag)
    same(v.view(np.float64), n.view(np.float64))
    i = np.arange(6, dtype=np.int64)
    same(viproc.asarray(i).view(np.float64), i.view(np.float64))
    same(viproc.asarray(i).view(np.int32), i.view(np.int32))


def test_reshape_fuzz():
    rng = np.random.default_rng(3)
    for n, v in layouts(rng):
        for _ in range(20):
            dims, size = [], n.size
            while n.size and (size > 1 or (not dims and rng.random() < 0.5)):
                d = int(rng.choice([k for k in range(1, size + 1) if size % k == 0]))
                dims.append(d)
                size //= d
                if rng.random() < 0.3:
                    dims.append(1)
            if n.size == 0:
                dims = [0] + dims
            rng.shuffle(dims)
            order = str(rng.choice(["C", "F", "A"]))
            expected = n.reshape(dims, order=order)
            got = v.reshape(dims, order=order)
            same(got, expected)
            try:
                np.reshape(n, dims, order=order, copy=False)
                np.reshape(v, dims, order=order, copy=False)
            except ValueError:
                with pytest.raises(ValueError):
                    np.reshape(v, dims, order=order, copy=False)


def test_getitem_slot_paths():
    n = np.arange(12.0).reshape(3, 4)
    v = viproc.asarray(n)
    assert type(v[1, 2]) is np.float64 and v[1, 2] == n[1, 2]  # ready: NumPy's scalar
    big = np.arange(1e5).reshape(100, 1000)
    p = np.sin(viproc.asarray(big))  # above sync_below: pending (async)
    s = p[1, 2]  # pending: a 0-d snapshot instead of a sync
    assert not isinstance(s, np.ndarray) and float(s) == np.sin(big)[1, 2]
    e = v[1, ...][2:3][..., 0]  # Ellipsis: a 0-d array, as NumPy
    assert isinstance(e, viproc.ndarray) and e.shape == () and float(e) == n[1, 2]
    same(v[np.array([2, 0])], n[np.array([2, 0])])  # advanced: the Python hook
    same(v[[True, False, True]], n[[True, False, True]])
    for key in [(3, 0), (0, 0, 0), 1.5, "x"]:
        with pytest.raises(Exception) as expected:
            n[key]
        with pytest.raises(type(expected.value)):
            v[key]
