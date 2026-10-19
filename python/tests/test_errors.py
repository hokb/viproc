"""Asynchronous error reporting (docs/design/error-handling.md): an op that
fails on any thread is reported at the next call into viproc, with NumPy's
exception type (as a subclass), its message, the op and where it was issued."""

import sys
import time

import numpy as np
import pytest

import viproc

NEG = "Integers to negative integer powers are not allowed"


def _operands(n, dtype="i8"):
    base = np.arange(n, dtype=dtype) % 7
    exp = np.full(n, 2, dtype=dtype)
    exp[-1] = -1
    return base, exp


def _line():
    return sys._getframe(1).f_lineno


@pytest.mark.parametrize("n", [5, 100_000])  # run directly / as a task
@pytest.mark.parametrize("dtype", ["i4", "i8"])
def test_integer_power_with_negative_exponent_raises_numpys_value_error(n, dtype):
    # NumPy's loop raises through the Python API and returns success; the
    # kernel catches the error on whatever thread ran it.
    base, exp = _operands(n, dtype)
    with pytest.raises(ValueError, match=NEG):
        np.power(base, exp)
    x, y = viproc.asarray(base), viproc.asarray(exp)
    for f in (lambda: np.power(x, y), lambda: viproc.power(x, y), lambda: x ** y,
              lambda: x ** -1, lambda: np.power(x, exp[::-1])):
        with pytest.raises(viproc.AsyncValueError, match=NEG) as e:
            np.asarray(f())
        assert isinstance(e.value, ValueError)
        assert "in numpy.power, issued at " in str(e.value) and __file__ in str(e.value)
    assert np.array_equal(np.asarray(x + 1), base + 1)  # nothing left behind
    ok = np.abs(exp)
    assert np.array_equal(np.asarray(x ** viproc.asarray(ok)), base ** ok)


def test_integer_power_stays_asynchronous():
    import viproc._ndarray as nd
    base, exp = _operands(100_000)
    ok = np.abs(exp)
    x, y = viproc.asarray(base), viproc.asarray(ok)
    f0 = nd.fallback_calls
    r = [x ** y, np.power(x, y), x ** 3]
    assert nd.fallback_calls == f0
    for a, b in zip(r, [base ** ok, base ** ok, base ** 3]):
        assert np.array_equal(np.asarray(a), b)


def test_error_is_raised_at_the_next_call_into_viproc_and_reported_once():
    base, exp = _operands(100_000)
    x, y = viproc.asarray(base), viproc.asarray(exp)
    line = _line() + 1
    r = x ** y  # fails on a worker; nobody syncs on r
    with pytest.raises(viproc.AsyncValueError, match=NEG) as e:
        for _ in range(2000):  # some later, unrelated call raises it
            time.sleep(0.001)
            _ = x + 1
    assert f"{__file__}:{line} in " in str(e.value)
    assert np.array_equal(np.asarray(x + 1), base + 1)  # reported once only
    with pytest.raises(ValueError, match=NEG):
        np.asarray(r)  # r itself stays failed


def test_wait_all_raises_a_failed_op_and_sync_points_do_not_report_twice():
    base, exp = _operands(100_000)
    x, y = viproc.asarray(base), viproc.asarray(exp)
    x ** y
    with pytest.raises(viproc.AsyncValueError, match=NEG):
        viproc.wait_all()
    viproc.wait_all()  # nothing left
    r = x ** y
    with pytest.raises(ValueError, match=NEG):
        np.asarray(r)  # the sync point reports it ...
    viproc.wait_all()  # ... and the store does not report it again


def test_consumers_inherit_the_error_of_the_op_that_failed():
    base, exp = _operands(100_000)
    x, y = viproc.asarray(base), viproc.asarray(exp)
    line = _line() + 1
    p = x ** y
    s = ((p + 1) * 2).sum()
    with pytest.raises(viproc.AsyncValueError, match=NEG) as e:
        float(s)
    assert f"in numpy.power, issued at {__file__}:{line} in " in str(e.value)
    viproc.wait_all()  # only the op that failed was reported, once


def test_the_earliest_failed_op_is_reported_and_the_others_counted():
    n = 100_000
    v = viproc.asarray(np.arange(float(n)))
    base, exp = _operands(n)
    x = viproc.asarray(base)
    # Everything below waits for `slow`: all three ops fail after the last
    # one was issued, in any order.
    slow = np.sin(np.sin(viproc.asarray(np.ones(10_000_000))))
    zero = (slow[:n] * 0).astype(np.int64)
    idx = viproc.asarray(np.where(np.arange(n) == 7, 2 * n, 1)) + zero
    y = viproc.asarray(exp) + zero
    line = _line() + 1
    v[idx]  # IndexError, issued first
    x ** y
    x ** y
    with pytest.raises(viproc.AsyncIndexError, match=f"index {2 * n} is out of bounds") as e:
        viproc.wait_all()
    assert isinstance(e.value, IndexError) and f"{__file__}:{line} in " in str(e.value)
    assert "[2 more failed ops not reported]" in str(e.value)


@pytest.mark.parametrize("n", [5, 100_000])
def test_an_array_written_by_a_failed_op_fails_until_it_is_overwritten(n):
    base, exp = _operands(n)
    a, y = viproc.asarray(base), viproc.asarray(exp)
    a **= y  # NumPy raises here, having written a prefix; viproc fails `a`
    with pytest.raises(viproc.AsyncValueError, match=NEG):
        viproc.wait_all()
    with pytest.raises(ValueError, match=NEG):
        np.asarray(a)
    with pytest.raises(ValueError, match=NEG):
        np.asarray(a + 1)  # computed from it
    a[...] = 7  # writes every element: valid data again
    assert np.array_equal(np.asarray(a), np.full(n, 7))


def test_failing_ops_on_many_threads_leave_no_python_error_behind():
    base, exp = _operands(50_000)
    x, y, ok = viproc.asarray(base), viproc.asarray(exp), viproc.asarray(np.abs(exp))
    bad = [x ** y for _ in range(40)]
    good = [x ** ok for _ in range(40)]
    with pytest.raises(ValueError, match=NEG):
        viproc.wait_all()
    for b in bad:
        with pytest.raises(ValueError, match=NEG):
            np.asarray(b)
    for g in good:
        assert np.array_equal(np.asarray(g), base ** np.abs(exp))


def test_restarting_the_runtime_raises_an_error_nobody_reported():
    base, exp = _operands(100_000)
    x, y = viproc.asarray(base), viproc.asarray(exp)
    x ** y
    try:
        with pytest.raises(viproc.AsyncValueError, match=NEG):
            viproc.init()
    finally:
        if not viproc._viproc.initialized():
            viproc.init()
    assert np.array_equal(np.asarray(viproc.asarray(base) + 1), base + 1)
