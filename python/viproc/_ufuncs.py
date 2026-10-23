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

import math
import operator
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


def _typed_inputs(ufunc, inputs, out, kwargs):
    """NumPy's loop dtypes for ufunc(*inputs, out=out, **kwargs)
    (ufunc.resolve_dtypes with dtype=, signature=, casting=) and the inputs
    as viproc arrays (weak Python scalars converted to their loop dtype, as
    NumPy converts them): (dtypes, dtype codes, inputs), or _NOPE."""
    nin, nout = ufunc.nin, ufunc.nout
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
    return dtypes, codes, ins


def _complex_warnings(ins, dtypes, out_casts):
    """NumPy warns once per cast of complex to real values it sets up (each
    call); viproc's iterator templates are made once: warn here (the caller
    silences the cast's own warning). Returns the number of warnings."""
    discards = sum(a.kind == "c" and b.kind in "iuf" for a, b in
                   [(x.dtype, d) for x, d in zip(ins, dtypes)] + out_casts)
    for _ in range(discards):
        warnings.warn("Casting complex values to real discards the imaginary part",
                      np.exceptions.ComplexWarning, stacklevel=4)
    return discards


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
    r = _typed_inputs(ufunc, inputs, out, kwargs)
    if r is _NOPE:
        return _NOPE
    dtypes, codes, ins = r
    if out is None and all(x.ndim == 0 for x in ins) and (where is True or where.ndim == 0):
        return _NOPE  # NumPy's scalars
    if where is not True and "out" not in kwargs:
        warnings.warn("'where' used without 'out', expect uninitialized memory in output. "
                      "If this is intentional, use out=None.", UserWarning, stacklevel=3)
    discards = _complex_warnings(ins, dtypes, [(dtypes[-1], out[0].dtype)] if out else [])
    with warnings.catch_warnings():
        if discards:
            warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        if nout == 1:
            r = _viproc.ufunc_typed(ufunc, tuple(ins), codes, None if out is None else out[0],
                                    None if where is True else where)
        else:
            r = _viproc.ufunc_multi(ufunc, tuple(ins), codes)
    return _NOPE if r is None else r


# ufunc -> its parsed core signature: (labels, fixed sizes (-1 if free),
# flexible flags, per operand the label index of each core axis)
_signatures = {}


def _core_signature(ufunc):
    sig = _signatures.get(ufunc)
    if sig is None:
        labels, fixed, flexible, ops = [], [], [], []
        ins, outs = ufunc.signature.replace(" ", "").split("->")
        for part in (ins, outs):
            for group in part[1:-1].split("),("):
                axes = []
                for name in filter(None, group.split(",")):
                    flex = name.endswith("?")
                    name = name.rstrip("?")
                    if name not in labels:
                        labels.append(name)
                        fixed.append(int(name) if name.isdigit() else -1)
                        flexible.append(flex)
                    axes.append(labels.index(name))
                ops.append(axes)
        sig = _signatures[ufunc] = (labels, fixed, flexible, ops)
    return sig


# (ufunc, signature, axes, axis, invalid, flags, out_order, label_sizes,
# given_work, (dtype, shape) per input) -> the arguments of _viproc.gufunc
# and the operands' axis remaps for a call on viproc arrays without keywords
# other than a signature tuple, axes= and axis= (all of them depend only on
# the key). numpy.fft passes axes= on every call: without the plan each call
# planned again (~70 us on the main thread, more than an FFT of 100 points).
_gufunc_plans = {}
_PLAN_KEYWORDS = frozenset(("signature", "axes", "axis"))


def _plan_key(ufunc, inputs, kwargs, label_sizes, invalid, flags, given_work, out_order):
    """The _gufunc_plans key of a call, or None if it has none."""
    if not kwargs.keys() <= _PLAN_KEYWORDS:
        return None
    sig = kwargs.get("signature")
    if sig is not None and type(sig) is not tuple:
        return None
    # viproc arrays, and NumPy arrays or scalars (imported per call, as
    # _typed_inputs imports them); not Python scalars (NEP 50: weak)
    if any(type(x) is not ndarray and not isinstance(x, (np.ndarray, np.generic))
           for x in inputs):
        return None
    axes = kwargs.get("axes")
    try:
        if axes is not None:
            axes = tuple(tuple(e) if isinstance(e, (list, tuple)) else e for e in axes)
        key = (ufunc, sig, axes, kwargs.get("axis"), invalid, flags, out_order,
               None if label_sizes is None else tuple(sorted(label_sizes.items())),
               given_work) + tuple((type(x) is ndarray, x.dtype, x.shape) for x in inputs)
        hash(key)
    except TypeError:
        return None
    return key

_GUFUNC_KEYWORDS = frozenset(("out", "dtype", "signature", "casting", "axes", "axis"))


def _remaps(ufunc, ops, ncore, axes, axis, bcast):
    """Per operand the axis order with its core axes last (NumPy's
    REMAP_AXIS from _parse_axes_arg / _parse_axis_arg), or None to keep it.
    `ops` are the operands (None: an output to allocate). Raises where
    NumPy raises (the caller then leaves the call to NumPy)."""
    nin, nop = ufunc.nin, len(ops)
    if axis is not None:
        if len(_core_signature(ufunc)[0]) != 1:
            raise TypeError
        axis = operator.index(axis)
        axes = [axis if n == 1 else () for n in ncore]
    if not isinstance(axes, list) or (len(axes) != nop and (
            len(axes) != nin or any(ncore[nin:]))):
        raise ValueError
    remaps = [None] * nop
    for i, entry in enumerate(axes):
        nd = ops[i].ndim if ops[i] is not None else bcast + ncore[i]
        if not isinstance(entry, tuple):
            if ncore[i] != 1:
                raise TypeError
            entry = (entry,)
        if len(entry) != ncore[i]:
            raise ValueError
        core = [normalize_axis_index(operator.index(a), nd) for a in entry]
        if len(set(core)) != len(core):
            raise ValueError
        order = [a for a in range(nd) if a not in core] + core
        if order != list(range(nd)):
            remaps[i] = order
    return remaps


def gufunc(ufunc, inputs, kwargs, label_sizes=None, invalid=None, flags=0,
           given_work=None, out_order=()):
    """A generalized ufunc call (ufunc.signature set: matmul with casts or
    out=, vecdot, matvec, vecmat, ...) on NumPy's gufunc loop (vp_gufunc):
    NumPy's dtypes (ufunc.resolve_dtypes), inputs of another dtype cast as
    NumPy's iterator copies them, missing flexible dimensions (matmul's 1-d
    operands), axes=/axis= as views moving the core axes last, out= of the
    loop's dtype. Returns _NOPE where NumPy has to run it (keepdims=, out=
    of another dtype or overlapping an input, NumPy's errors).

    For numpy.linalg's wrappers (_linalg.py): `label_sizes` gives the sizes
    of labels only outputs have (svd's `p`), `invalid` the LinAlgError
    message for the loop's FP invalid flag, `flags` VP_GUFUNC_CHECK_FINITE
    (NumPy's _assert_finite of the first input) and VP_GUFUNC_SERIAL,
    `out_order` the outputs' axis order (numpy.fft: empty_like's order K),
    `given_work` the
    op's cost for the dispatch policy."""
    key = _plan_key(ufunc, inputs, kwargs, label_sizes, invalid, flags, given_work, out_order)
    plan = None if key is None else _gufunc_plans.get(key)
    if plan is not None:  # the same call as before: all decisions taken
        codes, core_n, core_axes, sizes, work, discards, remaps = plan
        for _ in range(discards):
            warnings.warn("Casting complex values to real discards the imaginary part",
                          np.exceptions.ComplexWarning, stacklevel=3)
        nin = len(inputs)
        inputs = [x if type(x) is ndarray else _nd.asarray(x) for x in inputs]
        if remaps is not None:
            inputs = [x if m is None else x.transpose(m) for x, m in zip(inputs, remaps)]
        if discards:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
                r = _viproc.gufunc(ufunc, tuple(inputs), None, codes, core_n, core_axes, sizes,
                                   flags, out_order, invalid, work)
        else:
            r = _viproc.gufunc(ufunc, tuple(inputs), None, codes, core_n, core_axes, sizes,
                               flags, out_order, invalid, work)
        if r is None:
            return _NOPE
        res = []
        for i, x in enumerate(r):
            m = None if remaps is None else remaps[nin + i]
            if m is not None:
                x = x.transpose(np.argsort(m).tolist())
            res.append(_nd._scalar_result(x) if x.ndim == 0 else x)
        return res[0] if len(res) == 1 else tuple(res)
    nin, nout = ufunc.nin, ufunc.nout
    if len(inputs) != nin or not kwargs.keys() <= _GUFUNC_KEYWORDS or nin + nout > 8:
        return _NOPE
    if "axes" in kwargs and "axis" in kwargs:
        return _NOPE
    out = kwargs.get("out")
    if out is not None:
        out = out if isinstance(out, tuple) else (out,)
        if all(o is None for o in out):
            out = None
        elif len(out) != nout or any(type(o) is not ndarray or o._readonly for o in out):
            return _NOPE
    labels, fixed, flexible, core = _core_signature(ufunc)
    r = _typed_inputs(ufunc, inputs, out, kwargs)
    if r is _NOPE:
        return _NOPE
    dtypes, codes, ins = r
    ops = ins + (list(out) if out is not None else [None] * nout)
    # _validate_num_dims: operands with too few axes drop flexible ones
    ncore = [len(c) for c in core]
    missing = set()
    for i, x in enumerate(ops):
        if x is None or x.ndim >= ncore[i]:
            continue
        for label in core[i]:
            if flexible[label] and label not in missing:
                missing.add(label)
                ncore = [n - c.count(label) for n, c in zip(ncore, core)]
                if x.ndim == ncore[i]:
                    break
        if x.ndim < ncore[i]:
            return _NOPE  # NumPy's ValueError
    bcast = max(x.ndim - n for x, n in zip(ops, ncore) if x is not None)
    if "axes" in kwargs or "axis" in kwargs:
        if missing:
            return _NOPE
        try:
            remaps = _remaps(ufunc, ops, ncore, kwargs.get("axes"), kwargs.get("axis"), bcast)
        except (TypeError, ValueError, np.exceptions.AxisError):
            return _NOPE
        ops = [x if m is None or x is None else x.transpose(m) for x, m in zip(ops, remaps)]
    else:
        remaps = [None] * len(ops)
    sizes = [1 if label in missing else f for label, f in enumerate(fixed)]
    for name, n in (label_sizes or {}).items():
        sizes[labels.index(name)] = n
    core_axes = tuple(-1 - label if label in missing else label for c in core for label in c)
    # rough cost for the dispatch policy: the loop's index space per batch
    # element (vecdot: n multiply-adds; matmul: n*k*m, which BLAS runs about
    # 16x faster per element, as vp_matmul counts it)
    batch = 1
    for x, n in zip(ops, ncore):
        if x is not None:
            batch = max(batch, math.prod(x.shape[:x.ndim - n]))
    dims = {}
    for x, c in zip(ops, core):
        if x is not None:
            present = [label for label in c if label not in missing]
            dims.update(zip(present, x.shape[x.ndim - len(present):]))
    work = batch * math.prod(dims.values())
    work = max(work // 16 if ufunc is np.matmul else work, 1) if given_work is None else given_work
    core_n = tuple(len(c) for c in core)
    discards = _complex_warnings(ins, dtypes, [])
    with warnings.catch_warnings():
        if discards:
            warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        r = _viproc.gufunc(ufunc, tuple(ops[:nin]), None if out is None else tuple(ops[nin:]),
                           codes, core_n, core_axes, tuple(sizes), flags, out_order, invalid,
                           work)
    if r is None:
        return _NOPE
    if key is not None and out is None:
        if len(_gufunc_plans) >= 256:
            _gufunc_plans.clear()
        _gufunc_plans[key] = (codes, core_n, core_axes, tuple(sizes), work, discards,
                              None if all(m is None for m in remaps) else tuple(remaps))
    if out is not None:
        return out[0] if nout == 1 else out
    res = []
    for x, m in zip(r, remaps[nin:]):
        if m is not None:
            x = x.transpose(np.argsort(m).tolist())
        res.append(_nd._scalar_result(x) if x.ndim == 0 else x)
    return res[0] if nout == 1 else tuple(res)


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
