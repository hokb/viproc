"""ufunc calls and methods the C fast path does not take (hokb/viproc#25,
step 5): ufuncs with several outputs, dtype=, signature=, casting=, out= of
another dtype; ufunc.accumulate (np.cumsum, np.cumprod, np.nancumsum,
np.nancumprod), ufunc.reduceat, ufunc.at and ufunc.outer.

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

# (ufunc, input dtype, dtype=, method) -> the accumulator dtype of
# accumulate/reduceat, or None if NumPy's does not use one uniform loop
# (R, R) -> R
_acc_dtypes = {}


def _acc_dtype(ufunc, dtype, req, method="accumulate"):
    key = (ufunc, dtype, req, method)
    r = _acc_dtypes.get(key, False)
    if r is not False:
        return r
    r = None
    if ufunc.nin == 2 and ufunc.nout == 1 and ufunc.signature is None:
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                run = getattr(ufunc, method)
                args = ([0],) if method == "reduceat" else ()
                rd = run(np.ones(2, dtype), *args, dtype=req).dtype
                # the loop NumPy takes for an input already of the result
                # dtype must be the uniform one (R, R) -> R
                if (run(np.ones(2, rd), *args, dtype=req).dtype == rd
                        and ufunc.resolve_dtypes((None, rd, None), reduction=True)
                        == (rd, rd, rd)):
                    r = rd
        except (TypeError, ValueError):
            r = None
    if r is not None and r not in _nd._CODES:
        r = None
    _acc_dtypes[key] = r
    return r


_CALL_KEYWORDS = frozenset(("out", "dtype", "signature", "casting", "where"))
_PY_SCALARS = (bool, int, float, complex)


def call(ufunc, inputs, kwargs):
    """ufunc(*inputs, **kwargs) for what the C fast path leaves out: several
    outputs (divmod, modf, frexp), dtype=, signature=, casting=, where= (a
    bool mask: NumPy's masked loop, vp_ufunc_issue_masked) and an out=
    array of another dtype (NumPy casts into it in its iterator's buffers;
    so does viproc's kernel). The dtypes are NumPy's (ufunc.resolve_dtypes,
    which also raises NumPy's casting errors: then NumPy runs the call);
    inputs of another dtype are cast as NumPy casts them (vp_ufunc_issue,
    vp_ufunc_multi). Returns _NOPE where NumPy has to run it."""
    nin, nout = ufunc.nin, ufunc.nout
    if (ufunc.signature is not None or len(inputs) != nin or not kwargs.keys() <= _CALL_KEYWORDS
            or nin > 3):
        return _NOPE
    where = kwargs.get("where", True)
    if where is not True:
        # _wheremask_converter: a bool array (a safe cast); several outputs: not yet
        if nout != 1:
            return _NOPE
        if not isinstance(where, ndarray):
            where = np.asarray(where)
            if where.dtype != np.bool_:
                return _NOPE
            where = _nd.asarray(where)
        if where.dtype != np.bool_:
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
    if out is None and all(x.ndim == 0 for x in ins) and (where is True or where.ndim == 0):
        return _NOPE  # NumPy's scalars
    if where is not True and "out" not in kwargs:
        warnings.warn("'where' used without 'out', expect uninitialized memory in output. "
                      "If this is intentional, use out=None.", UserWarning, stacklevel=3)
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
            r = _viproc.ufunc_typed(ufunc, tuple(ins), codes, None if out is None else out[0],
                                    None if where is True else where)
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

# ufuncs with NumPy indexed loops (generate_umath.py, indexed=): NumPy's
# trivial_at_loop runs a 1-d ufunc.at with them several times faster than
# one loop call per element, which is all viproc can do with NumPy's public
# loops: those stay NumPy's (G3).
_INDEXED = frozenset((np.add, np.subtract, np.multiply, np.floor_divide, np.true_divide,
                      np.maximum, np.minimum, np.fmax, np.fmin))


def at(ufunc, inputs, kwargs):
    """ufunc.at(a, indices[, b]) with one integer index array (or an int) into
    a's first axis and b of a's dtype (a Python scalar counts as int64,
    float64, ... as in NumPy's ufunc_at): one task in place on a
    (vp_ufunc_at). Returns None, as NumPy, or _NOPE."""
    nin = ufunc.nin
    if kwargs or len(inputs) != nin + 1 or ufunc.nout != 1 or ufunc.signature is not None:
        return _NOPE
    a, indices = inputs[0], inputs[1]
    if type(a) is not ndarray or a._readonly or a.ndim == 0:
        return _NOPE
    if isinstance(indices, tuple):
        if len(indices) != 1:
            return _NOPE
        indices = indices[0]
    if type(indices) is int:
        indices = [indices]
    if isinstance(indices, ndarray):
        idx = indices if indices.dtype in _nd._INDEX_DTYPES and indices.ndim == 1 else None
    else:
        x = np.asarray(indices)
        idx = None
        if x.ndim == 1 and x.dtype.kind in "iu" and x.dtype != np.uint64 and x.size:
            idx = _nd._adopt(np.array(x, dtype=np.intp, copy=True))
    if idx is None or idx.size == 0:
        return _NOPE
    if a.ndim == 1 and ufunc in _INDEXED and a.dtype != np.bool_:
        return _NOPE  # NumPy's indexed loop (trivial_at_loop)
    values = None
    if nin == 2:
        b = inputs[2]
        if not isinstance(b, (ndarray, np.ndarray)):
            # PyArray_FromAny: a Python scalar is a strong int64/float64/...
            # here, not a weak one (np.add.at(int8, i, 300) wraps)
            b = np.asarray(b)
        try:
            dtypes = ufunc.resolve_dtypes((a.dtype, b.dtype, a.dtype))
        except (TypeError, ValueError):
            return _NOPE
        if any(d != a.dtype for d in dtypes) or b.dtype != a.dtype:
            return _NOPE  # a cast: NumPy's buffered at (ufunc_at__slow_iter)
        if not isinstance(b, ndarray):
            b = _nd.asarray(b)
            if not isinstance(b, ndarray):
                return _NOPE
        values = b
    return None if _viproc.ufunc_at(ufunc, a, idx, values) else _NOPE


def reduceat(ufunc, inputs, kwargs):
    """ufunc.reduceat(a, indices, axis=0, dtype=None) with host indices: one
    task (vp_reduceat), the input cast first to NumPy's accumulator dtype.
    NumPy raises its errors (indices out of range) in the fallback."""
    if len(inputs) != 2 or not kwargs.keys() <= {"axis", "dtype"}:
        return _NOPE
    a, indices = inputs
    if type(a) is not ndarray or a.ndim == 0 or isinstance(indices, ndarray):
        return _NOPE
    ind = np.asarray(indices)
    if ind.ndim != 1 or ind.size == 0 or ind.dtype.kind not in "iu":
        return _NOPE
    try:
        axis = normalize_axis_index(kwargs.get("axis", 0), a.ndim)
        req = kwargs.get("dtype")
        req = None if req is None else np.dtype(req)
    except (TypeError, ValueError, np.exceptions.AxisError):
        return _NOPE
    if a.size == 0:
        return _NOPE
    n = a.shape[axis]
    if ind.min() < 0 or ind.max() >= n:
        return _NOPE  # NumPy's IndexError
    ind = ind.astype(np.int64).tobytes()  # host data, checked: the task's copy
    rd = _acc_dtype(ufunc, a.dtype, req, "reduceat")
    if rd is None:
        return _NOPE
    if a.dtype != rd:
        a = a.astype(rd)
    r = _viproc.reduceat(ufunc, a, axis, ind, _reduceat_strides(a, axis, len(ind) // 8))
    return _NOPE if r is None else r


def _reduceat_strides(a, axis, m):
    """The strides of the result PyUFunc_Reduceat allocates: 1-d in C order
    (no iterator), else as its iterator allocates it in order K over the
    operands (out: none yet, arr: no stride along `axis`, the intp indices:
    only along `axis`): the axis order of NumPy's
    npyiter_find_best_axis_ordering, a stable insertion sort by |stride|
    in which C order wins where no operand decides."""
    nd = a.ndim
    shape = list(a.shape)
    shape[axis] = m
    if nd == 1:
        return None
    ops = [[0 if ax == axis or shape[ax] == 1 else a.strides[ax] for ax in range(nd)],
           [8 if ax == axis and m != 1 else 0 for ax in range(nd)]]
    perm = list(range(nd - 1, -1, -1))  # innermost first (reversed C order)
    for i0 in range(1, nd):
        ipos = i0
        j0 = perm[i0]
        for i1 in range(i0 - 1, -1, -1):
            j1 = perm[i1]
            ambig, swap = True, False
            for st in ops:
                if st[j0] != 0 and st[j1] != 0:
                    if abs(st[j1]) <= abs(st[j0]):
                        swap = False
                    elif ambig:
                        swap = True
                    ambig = False
            if not ambig:
                if swap:
                    ipos = i1
                else:
                    break
        if ipos != i0:
            perm.insert(ipos, perm.pop(i0))
    strides = [0] * nd
    step = a.itemsize
    for ax in perm:
        strides[ax] = step
        step *= shape[ax]
    return strides


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
