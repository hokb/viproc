"""Floating-point errors (docs/design/error-handling.md, E8): asynchronous ops
report divide-by-zero, overflow, underflow and invalid as NumPy does, under
the np.errstate of the code that issued them: the same warnings (text,
category, order, the issuing line), FloatingPointError under 'raise'
(AsyncFloatingPointError), errstate's call, log and print modes."""

import io
import warnings

import numpy as np
import pytest

import viproc

N = 10_000  # above sync_below: the ops run as tasks
a = np.resize(np.array([0.0, 1e308, -1.0, 1.0, 1e-300]), N)


def warnings_of(f, *args):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = f(*args)
        if not np.isscalar(r):
            np.asarray(r)  # sync
        viproc.wait_all()  # reports what is still queued
    return [(w_.category, str(w_.message)) for w_ in w]


CASES = {
    "divide": lambda x: x / 0,
    "multiply": lambda x: x * 10,
    "sqrt": lambda x: np.sqrt(x),
    "log": lambda x: np.log(x),
    "exp": lambda x: np.exp(x),
    "underflow": lambda x: x * 1e-300,
    "sum": lambda x: np.sum(x * 10),
    "reduce": lambda x: np.add.reduce(x),
    "mean": lambda x: x.mean(),
    "cumsum": lambda x: np.cumsum(x),
    "cast": lambda x: (x * np.inf).astype(np.int32),
    "matmul": lambda x: x.reshape(100, 100) @ x.reshape(100, 100),
    "integer divide": lambda x: x.astype(np.int64) // 0,
    "integer remainder": lambda x: x.astype(np.int64) % 0,
    "small (direct)": lambda x: x[:4] / 0,
    "float32": lambda x: x.astype(np.float32) * 1e30,
}


@pytest.mark.parametrize("errstate", [{}, {"all": "warn"}, {"divide": "ignore"},
                                      {"all": "ignore"}, {"under": "warn", "over": "ignore"}])
@pytest.mark.parametrize("case", CASES)
def test_warnings_like_numpy(case, errstate):
    f = CASES[case]
    with np.errstate(**errstate):
        expected = warnings_of(f, a)
        got = warnings_of(f, viproc.asarray(a))
    assert got == expected


@pytest.mark.parametrize("case", ["divide", "multiply", "sum", "cast", "matmul"])
def test_raise_like_numpy(case):
    f = CASES[case]
    with np.errstate(all="raise"):
        with pytest.raises(FloatingPointError) as expected:
            f(a)
        with pytest.raises(FloatingPointError) as got:
            r = f(viproc.asarray(a))
            np.asarray(r)
            viproc.wait_all()
    assert isinstance(got.value, viproc.AsyncFloatingPointError)
    assert str(got.value).startswith(str(expected.value))


def test_warn_before_raise_in_numpys_order():
    with np.errstate(divide="warn", invalid="raise"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(FloatingPointError, match="invalid value encountered in divide"):
                np.asarray(viproc.asarray(a) / 0)
    assert [str(x.message) for x in w] == ["divide by zero encountered in divide"]


def test_errstate_is_the_one_of_the_issue():
    v = viproc.asarray(a)
    with np.errstate(all="ignore"):
        r = v / 0  # issued here, completes later
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        np.asarray(r)
        viproc.wait_all()
    assert w == []


def test_call_log_and_print_modes(capfd):
    calls = []

    class Log:
        def __init__(self):
            self.lines = []

        def write(self, s):
            self.lines.append(s)

    for x in (a, viproc.asarray(a)):
        with np.errstate(all="call", call=lambda *args: calls.append(args)):
            np.asarray(x / 0)
            viproc.wait_all()
    assert calls[:2] == calls[2:] == [("divide by zero", 9), ("invalid value", 9)]
    logs = []
    for x in (a, viproc.asarray(a)):
        log = Log()
        with np.errstate(all="log", call=log):
            np.asarray(x * 10)
            viproc.wait_all()
        logs.append(log.lines)
    assert logs[0] == logs[1] == ["Warning: overflow encountered in multiply\n"]
    capfd.readouterr()
    for x in (a, viproc.asarray(a)):
        with np.errstate(all="print"):
            np.asarray(x * 10)
            viproc.wait_all()
    err = capfd.readouterr().err
    assert err == "Warning: overflow encountered in multiply\n" * 2


def test_warnings_point_at_the_issuing_line_and_warn_once_there():
    v = viproc.asarray(a)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("default")
        for _ in range(3):
            r = v / 0; line = __import__("sys")._getframe().f_lineno  # noqa: E702
            np.asarray(r)
        viproc.wait_all()
    assert [str(x.message) for x in w] == ["divide by zero encountered in divide",
                                           "invalid value encountered in divide"]
    assert all(x.filename == __file__ and x.lineno == line for x in w)


def test_warning_filter_error_raises():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning, match="divide by zero encountered in divide"):
            np.asarray(viproc.asarray(a) / 0)
            viproc.wait_all()
