"""ufunc calls and methods the C fast path does not take (hokb/viproc#25,
step 5): ufuncs with several outputs, dtype=, signature=, casting=, out= of
another dtype; ufunc.accumulate (np.cumsum, np.cumprod, np.nancumsum,
np.nancumprod) and ufunc.outer.

accumulate follows NumPy's PyUFunc_Accumulate (ufunc_object.c,
BSD-3-Clause): the accumulator dtype is NumPy's (taken from NumPy itself,
once per ufunc and dtypes, on a 2-element probe), the input is cast to it
first (NumPy's iterator copies a cast input), and the loop runs as NumPy
runs it, one call per row (vp_accumulate). outer is NumPy's reshape of the
first operand (trailing axes of extent 1) followed by a normal ufunc call.
A handler returns _NOPE for arguments it does not cover: NumPy then runs
the call in the eager fallback, with its own errors.
"""

import warnings

import numpy as np
from numpy.lib.array_utils import normalize_axis_index

from . import _join
from . import _ndarray as _nd
from . import _viproc

_NOPE = _nd._NOPE
ndarray = _nd.ndarray

# (ufunc, input dtype, dtype=) -> the accumulator dtype, or None if NumPy's
# accumulation of these does not use one uniform loop (R, R) -> R
_acc_dtypes = {}


def _acc_dtype(ufunc, dtype, req):
    key = (ufunc, dtype, req)
    r = _acc_dtypes.get(key, False)
    if r is not False:
        return r
    r = None
    if ufunc.nin == 2 and ufunc.nout == 1 and ufunc.signature is None:
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                rd = ufunc.accumulate(np.ones(2, dtype), dtype=req).dtype
                # the loop NumPy takes for an input already of the result
                # dtype must be the uniform one (R, R) -> R
                if (ufunc.accumulate(np.ones(2, rd), dtype=req).dtype == rd
                        and ufunc.resolve_dtypes((None, rd, None), reduction=True)
                        == (rd, rd, rd)):
                    r = rd
        except (TypeError, ValueError):
            r = None
    if r is not None and r not in _nd._CODES:
        r = None
    _acc_dtypes[key] = r
    return r


_CALL_KEYWORDS = frozenset(("out", "dtype", "signature", "casting"))
_PY_SCALARS = (bool, int, float, complex)


def call(ufunc, inputs, kwargs):
    """ufunc(*inputs, **kwargs) for what the C fast path leaves out: several
    outputs (divmod, modf, frexp), dtype=, signature=, casting= and an out=
    array of another dtype (NumPy casts into it in its iterator's buffers;
    so does viproc's kernel). The dtypes are NumPy's (ufunc.resolve_dtypes,
    which also raises NumPy's casting errors: then NumPy runs the call);
    inputs of another dtype are cast as NumPy casts them (vp_ufunc_issue,
    vp_ufunc_multi). Returns _NOPE where NumPy has to run it."""
    nin, nout = ufunc.nin, ufunc.nout
    if (ufunc.signature is not None or len(inputs) != nin or not kwargs.keys() <= _CALL_KEYWORDS
            or nin > 3):
        return _NOPE
    out = kwargs.get("out")
    if out is not None:
        out = out if isinstance(out, tuple) else (out,)
        # several outputs into given arrays: not yet (they may alias)
        if nout != 1 or len(out) != 1 or type(out[0]) is not ndarray or out[0]._readonly:
            return _NOPE
    spec = []
    for x in inputs:
        if isinstance(x, (ndarray, np.ndarray, np.generic)):
            spec.append(x.dtype)
        elif type(x) in _PY_SCALARS:
            spec.append(type(x))  # weak (NEP 50)
        else:
            return _NOPE
    spec += [o.dtype for o in out] if out is not None else [None] * nout
    signature = kwargs.get("signature")
    dtype = kwargs.get("dtype")
    if dtype is not None:
        if signature is not None:
            return _NOPE  # NumPy's TypeError
        signature = (None,) * nin + (dtype,) * nout
    resolve_kw = {} if signature is None else {"signature": signature}
    if "casting" in kwargs:
        resolve_kw["casting"] = kwargs["casting"]
    try:
        dtypes = ufunc.resolve_dtypes(tuple(spec), **resolve_kw)
    except (TypeError, ValueError):
        return _NOPE
    codes = tuple(_nd._CODES.get(d) for d in dtypes)
    if None in codes:
        return _NOPE
    ins = []
    for x, d in zip(inputs, dtypes):
        if type(x) in _PY_SCALARS:
            try:
                x = np.asarray(x, d)  # NumPy converts a weak scalar to the loop's dtype
            except (OverflowError, ValueError, TypeError):
                return _NOPE
        if not isinstance(x, ndarray):
            x = _nd.asarray(x)
            if not isinstance(x, ndarray):
                return _NOPE
        ins.append(x)
    if out is None and all(x.ndim == 0 for x in ins):
        return _NOPE  # NumPy's scalars
    # NumPy warns once per cast of complex to real values it sets up (each
    # call); viproc's iterator templates are made once: warn here, quiet there
    discards = sum(a.kind == "c" and b.kind in "iuf" for a, b in
                   [(x.dtype, d) for x, d in zip(ins, dtypes)]
                   + ([(dtypes[-1], out[0].dtype)] if out is not None else []))
    for _ in range(discards):
        warnings.warn("Casting complex values to real discards the imaginary part",
                      np.exceptions.ComplexWarning, stacklevel=3)
    with warnings.catch_warnings():
        if discards:
            warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        if nout == 1:
            r = _viproc.ufunc_typed(ufunc, tuple(ins), codes, None if out is None else out[0])
        else:
            r = _viproc.ufunc_multi(ufunc, tuple(ins), codes)
    return _NOPE if r is None else r


def _accumulate(ufunc, a, axis=0, dtype=None, out=None):
    if type(a) is not ndarray or out is not None or a.ndim == 0 or a.size == 0:
        return _NOPE
    try:
        axis = normalize_axis_index(axis, a.ndim)
        req = None if dtype is None else np.dtype(dtype)
    except (TypeError, ValueError, np.exceptions.AxisError):
        return _NOPE
    rd = _acc_dtype(ufunc, a.dtype, req)
    if rd is None:
        return _NOPE
    if a.dtype != rd:
        a = a.astype(rd)  # order K, as NumPy's iterator copies
    r = _viproc.accumulate(ufunc, a, axis)
    return _NOPE if r is None else r


def accumulate(ufunc, inputs, kwargs):
    """ufunc.accumulate(...) of a viproc array, or _NOPE."""
    if len(inputs) != 1:
        return _NOPE
    try:
        return _accumulate(ufunc, *inputs, **kwargs)
    except TypeError:
        return _NOPE


def outer(ufunc, inputs, kwargs):
    """ufunc.outer(a, b) as NumPy computes it: a reshaped with b.ndim
    trailing axes of extent 1 (a view), then ufunc(a, b)."""
    if len(inputs) != 2 or ufunc.nin != 2 or ufunc.signature is not None:
        return _NOPE
    a, b = (x if isinstance(x, (ndarray, np.ndarray)) else np.asarray(x) for x in inputs)
    return ufunc(a.reshape(a.shape + (1,) * b.ndim), b, **kwargs)

def _cum(ufunc):
    def cum(a, axis=None, dtype=None, out=None):
        if type(a) is not ndarray or out is not None:
            return _NOPE
        if axis is None:
            a = a.ravel()  # PyArray_CheckAxis: C order
            axis = 0
        return _accumulate(ufunc, a, axis, dtype)
    return cum


def _nancum(ufunc, value):
    cum = _cum(ufunc)

    def nancum(a, axis=None, dtype=None, out=None):
        if type(a) is not ndarray or out is not None:
            return _NOPE
        if a.dtype.kind in "fc":  # _replace_nan: NaNs by 0 (sum) or 1 (prod)
            a = np.where(np.isnan(a), value, a)
        return cum(a, axis, dtype)
    return nancum


def _small_ready(func, handler):
    """Small ready data runs in NumPy (_join._small_ready); out= goes to the
    handler, which leaves it to the fallback."""
    small = _join._small_ready(func, handler)

    def h(*args, **kwargs):
        if kwargs.get("out") is not None or (len(args) > 3 and args[3] is not None):
            return _NOPE
        return small(*args, **kwargs)
    return h


ACCUMULATING = {f: _small_ready(f, h) for f, h in {
    np.cumsum: _cum(np.add), np.cumprod: _cum(np.multiply),
    np.nancumsum: _nancum(np.add, 0), np.nancumprod: _nancum(np.multiply, 1),
}.items()}
