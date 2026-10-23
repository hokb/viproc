"""numpy.fft on NumPy's pocketfft gufuncs (hokb/viproc#25, step 6c): fft,
ifft, rfft, irfft, hfft, ihfft and the n-d transforms built on them (fftn,
ifftn, fft2, ifft2, rfftn, irfftn, rfft2, irfft2).

The code follows NumPy's numpy/fft/_pocketfft.py (BSD-3-Clause): the same
checks and errors, the same normalization factor `fct` (computed on the
main thread as NumPy computes it), the same gufunc (rfft_n_even/odd by the
parity of n), called as `ufunc(a, fct, axes=[(axis,), (), (axis,)],
out=out)`. The loop pads or truncates the input to n itself. NumPy
allocates `out` with empty_like(a, shape=...), order K; viproc gives the
result that layout (vp_gufunc's out_order). pocketfft makes its plan per
call (POCKETFFT_CACHE_SIZE 0 in NumPy's build), so its loops run on any
thread. A handler returns _NOPE for arguments it does not cover: NumPy then
runs the call in the eager fallback.
"""

import math

import numpy as np
from numpy.fft import _pocketfft_umath as pfu
from numpy.lib.array_utils import normalize_axis_index

from . import _ndarray as _nd
from . import _ufuncs
from ._linalg import _small_ready

_NOPE = _nd._NOPE
ndarray = _nd.ndarray

_SWAP_DIRECTION_MAP = {"backward": "forward", None: "forward",
                       "ortho": "ortho", "forward": "backward"}


def _swap_direction(norm):
    try:
        return _SWAP_DIRECTION_MAP[norm]
    except KeyError:
        raise ValueError(f'Invalid norm value {norm}; should be "backward", '
                         '"ortho" or "forward".') from None


def _keep_order(a):
    """The axis order (outermost first) of empty_like(a, shape=...) with
    a's number of axes: PyArray_NewLikeArrayWithShape, order K."""
    if a.ndim <= 1 or a.flags.c_contiguous:
        return list(range(a.ndim))
    if a.flags.f_contiguous:
        return list(range(a.ndim - 1, -1, -1))
    return sorted(range(a.ndim), key=lambda i: (-abs(a.strides[i]), i))


# (dtype, shape, strides, n, axis, is_real, is_forward, norm) -> the gufunc
# call of _raw_fft without out= (all of it depends only on the key): numpy.fft
# in a loop issues one gufunc call per transform, no planning in Python.
_fft_plans = {}


def _raw_fft(a, n, axis, is_real, is_forward, norm, out=None):
    if out is None:
        try:
            key = (a.dtype, a.shape, a.strides, n, axis, is_real, is_forward, norm)
            plan = _fft_plans.get(key)
        except TypeError:  # an unhashable norm: NumPy's error below
            key = plan = None
        if plan is not None:
            ufunc, fct, sizes, work, order, kwargs, out_dtype = plan
            r = _ufuncs.gufunc(ufunc, (a, fct), kwargs, sizes, None, 0, work, order)
            if r is _NOPE or r.dtype == out_dtype:
                return r
            return r.astype(out_dtype)
    else:
        key = None
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified.")
    if not is_forward:
        norm = _swap_direction(norm)
    if a.dtype.kind == "c":
        real_dtype = np.dtype(np.float32 if a.dtype == np.complex64 else np.float64)
    else:
        real_dtype = np.result_type(a.dtype, 1.0)
    if norm is None or norm == "backward":
        fct = 1
    elif norm == "ortho":
        fct = np.reciprocal(np.sqrt(n, dtype=real_dtype))
    elif norm == "forward":
        fct = np.reciprocal(n, dtype=real_dtype)
    else:
        raise ValueError(f'Invalid norm value {norm}; should be "backward", '
                         '"ortho" or "forward".')
    n_out = n
    if is_real:
        if is_forward:
            ufunc = pfu.rfft_n_even if n % 2 == 0 else pfu.rfft_n_odd
            n_out = n // 2 + 1
        else:
            ufunc = pfu.irfft
    else:
        ufunc = pfu.fft if is_forward else pfu.ifft
    if is_real and not is_forward:
        out_dtype = real_dtype
    else:
        out_dtype = np.result_type(a.dtype, 1j)
    # NumPy's loop for (a, fct, out): a Python int fct (norm "backward")
    # selects the double loop also for single precision, which NumPy's
    # iterator then casts into the complex64/float32 `out`.
    try:
        signature = ufunc.resolve_dtypes(
            (a.dtype, int if type(fct) is int else fct.dtype, out_dtype))
    except (TypeError, ValueError):
        return _NOPE
    axis = normalize_axis_index(axis, a.ndim)
    if out is not None:
        if type(out) is not ndarray:
            return _NOPE
        if len(out.shape) != a.ndim or out.shape[axis] != n_out:
            raise ValueError("output array has wrong shape.")
        if signature[-1] != out.dtype:
            return _NOPE
    if a.dtype not in _nd._CODES:
        return _NOPE
    labels, _, _, core = _ufuncs._core_signature(ufunc)
    sizes = {labels[core[-1][0]]: n_out}
    # the result's axes in the gufunc's operand order (the core axis last)
    remap = [i for i in range(a.ndim) if i != axis] + [axis]
    order = () if out is not None else tuple(remap.index(i) for i in _keep_order(a))
    batch = a.size // max(a.shape[axis], 1)
    work = max(batch * max(n, n_out) * max(1, int(math.log2(max(n, n_out)))), 1)
    kwargs = {"axes": [(axis,), (), (axis,)], "signature": signature}
    if out is not None:
        kwargs["out"] = out
    elif key is not None:
        # fct as a NumPy 0-d array of the loop's dtype (the value NumPy
        # converts the scalar to), so the gufunc's own plan covers the call
        # too; no viproc array in the plan: it would outlive its runtime.
        fct = np.asarray(fct, dtype=signature[1])
        if len(_fft_plans) >= 256:
            _fft_plans.clear()
        _fft_plans[key] = (ufunc, fct, sizes, work, order, kwargs, out_dtype)
    r = _ufuncs.gufunc(ufunc, (a, fct), kwargs, sizes, None, 0, work, order)
    if r is _NOPE or out is not None or r.dtype == out_dtype:
        return r
    return r.astype(out_dtype)  # NumPy's cast into `out`


def _array(a):
    if type(a) is not ndarray:
        a = _nd.asarray(a)
    return a if type(a) is ndarray else None


def _fft(a, n=None, axis=-1, norm=None, out=None):
    a = _array(a)
    if a is None:
        return _NOPE
    if n is None:
        n = a.shape[axis]
    return _raw_fft(a, n, axis, False, True, norm, out)


def _ifft(a, n=None, axis=-1, norm=None, out=None):
    a = _array(a)
    if a is None:
        return _NOPE
    if n is None:
        n = a.shape[axis]
    return _raw_fft(a, n, axis, False, False, norm, out=out)


def _rfft(a, n=None, axis=-1, norm=None, out=None):
    a = _array(a)
    if a is None:
        return _NOPE
    if n is None:
        n = a.shape[axis]
    return _raw_fft(a, n, axis, True, True, norm, out=out)


def _irfft(a, n=None, axis=-1, norm=None, out=None):
    a = _array(a)
    if a is None:
        return _NOPE
    if n is None:
        n = (a.shape[axis] - 1) * 2
    return _raw_fft(a, n, axis, True, False, norm, out=out)


def _hfft(a, n=None, axis=-1, norm=None, out=None):
    a = _array(a)
    if a is None:
        return _NOPE
    if n is None:
        n = (a.shape[axis] - 1) * 2
    new_norm = _swap_direction(norm)
    return _irfft(np.conjugate(a), n, axis, norm=new_norm, out=out)


def _ihfft(a, n=None, axis=-1, norm=None, out=None):
    a = _array(a)
    if a is None:
        return _NOPE
    if n is None:
        n = a.shape[axis]
    new_norm = _swap_direction(norm)
    out = _rfft(a, n, axis, norm=new_norm, out=out)
    if out is _NOPE:
        return _NOPE
    return np.conjugate(out, out=out)


def _cook_nd_args(a, s=None, axes=None, invreal=0):
    if s is None:
        shapeless = True
        if axes is None:
            s = list(a.shape)
        else:
            s = np.take(a.shape, axes)
    else:
        shapeless = False
    s = list(s)
    if axes is None:
        if not shapeless:
            return None  # NumPy's DeprecationWarning (stacklevel into NumPy's code)
        axes = list(range(-len(s), 0))
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    if invreal and shapeless:
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    if None in s:
        return None  # NumPy's DeprecationWarning
    s = [a.shape[_a] if _s == -1 else _s for _s, _a in zip(s, axes)]
    return s, axes


def _nd_function(one, kind):
    """fftn & co. as NumPy composes them of 1-d transforms; out= falls back
    (each step would write the array the next one reads)."""
    def h(a, s=None, axes=None, norm=None, out=None):
        a = _array(a)
        if a is None or out is not None:
            return _NOPE
        cooked = _cook_nd_args(a, s, axes, invreal=1 if kind == "irfft" else 0)
        if cooked is None:
            return _NOPE
        s, axes = cooked
        if kind == "c2c":
            for ii in reversed(range(len(axes))):
                a = one(a, n=s[ii], axis=axes[ii], norm=norm)
                if a is _NOPE:
                    return _NOPE
            return a
        if kind == "rfft":
            a = _rfft(a, s[-1], axes[-1], norm)
            for ii in range(len(axes) - 2, -1, -1):
                if a is _NOPE:
                    return _NOPE
                a = _fft(a, s[ii], axes[ii], norm)
            return a
        for ii in range(len(axes) - 1):
            a = _ifft(a, s[ii], axes[ii], norm)
            if a is _NOPE:
                return _NOPE
        return _irfft(a, s[-1], axes[-1], norm)

    return h


def _two(h):
    def h2(a, s=None, axes=(-2, -1), norm=None, out=None):
        return h(a, s, axes, norm, out)

    return h2


_fftn = _nd_function(_fft, "c2c")
_ifftn = _nd_function(_ifft, "c2c")
_rfftn = _nd_function(None, "rfft")
_irfftn = _nd_function(None, "irfft")

FFT = {f: _small_ready(f, h) for f, h in {
    np.fft.fft: _fft, np.fft.ifft: _ifft, np.fft.rfft: _rfft, np.fft.irfft: _irfft,
    np.fft.hfft: _hfft, np.fft.ihfft: _ihfft,
    np.fft.fftn: _fftn, np.fft.ifftn: _ifftn, np.fft.fft2: _two(_fftn), np.fft.ifft2: _two(_ifftn),
    np.fft.rfftn: _rfftn, np.fft.irfftn: _irfftn, np.fft.rfft2: _two(_rfftn),
    np.fft.irfft2: _two(_irfftn),
}.items()}
