"""Float arithmetic with plain 0-d viproc arrays (float_arith in _viproc.c):
computed in C where no FP flag arises, else on NumPy's path. Values, types,
warnings and errors as NumPy's, under every errstate."""

import itertools
import operator
import warnings

import numpy as np
import pytest

import viproc

VALUES = [0.0, -0.0, 1.0, -2.5, 7.25, 1e20, 1e-40, 1e-200, 1e-310, 3e38, 1e308,
          np.inf, -np.inf, np.nan]
OPS = [operator.add, operator.sub, operator.mul, operator.truediv,
       np.add, np.subtract, np.multiply, np.divide]
FORMS = [("arr", "sc"), ("sc", "arr"), ("arr", "py"), ("py", "arr"), ("arr", "arr")]


def _run(f, a, b):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            return f(a, b), None, [str(x.message) for x in w]
        except Exception as e:  # noqa: BLE001 - NumPy's errors are compared
            return None, (type(e).__name__, str(e)), [str(x.message) for x in w]


def _make(v, kind, dt):
    if kind == "arr":
        return np.array(v, dt)
    return dt(v) if kind == "sc" else float(v)


@pytest.mark.parametrize("mode", ["warn", "raise", "ignore"])
@pytest.mark.parametrize("dt", [np.float64, np.float32])
def test_float_arith_of_0d_arrays_is_numpys(mode, dt):
    for x, y in itertools.product(VALUES, VALUES):
        for kinds in FORMS:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                a, b = _make(x, kinds[0], dt), _make(y, kinds[1], dt)
                va = viproc.asarray(a) if kinds[0] == "arr" else a
                vb = viproc.asarray(b) if kinds[1] == "arr" else b
            for f in OPS:
                with np.errstate(all=mode):
                    ref = _run(f, a, b)
                    got = _run(f, va, vb)
                assert got[1] == ref[1] and got[2] == ref[2], (f, x, y, kinds, ref, got)
                assert type(got[0]) is type(ref[0]), (f, x, y, kinds, ref, got)
                if ref[0] is not None:
                    assert np.asarray(got[0]).tobytes() == np.asarray(ref[0]).tobytes()


def test_mixed_precision_0d():
    """float32 0-d with float64 operands computes in float64, as NumPy."""
    a32 = np.array(1.1, np.float32)
    v32 = viproc.asarray(a32)
    for other in (np.float64(0.3), np.array(0.3), 0.3):
        vo = viproc.asarray(other) if isinstance(other, np.ndarray) else other
        for f in OPS:
            ref, got = f(a32, other), f(v32, vo)
            assert type(got) is type(ref) and np.asarray(got).tobytes() == np.asarray(ref).tobytes()
