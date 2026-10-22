"""Checks that an op takes viproc's asynchronous path (hokb/viproc#25).

A bit-identity test alone also passes when an op falls back to NumPy, syncs
its inputs or runs on the main thread. assert_async() reads the runtime's
counters around one call instead: no fallback, no sync point, exactly the
expected number of tasks, none of them run on the main thread, and no
NumPy array as a result. Use it inside deterministic(), on data above sync_below
(SIZE elements), on pending inputs (pending()), then compare the result
with NumPy."""

import contextlib

import numpy as np

import viproc

SYNC_BELOW = 2048  # the default threshold, set explicitly (VIPROC_SYNC_BELOW)
SIZE = 5000  # elements: above SYNC_BELOW


@contextlib.contextmanager
def deterministic():
    """The default threshold without the adaptive policy, which could switch
    an issue site to inline execution."""
    viproc.init(sync_below=SYNC_BELOW, adaptive=False)
    try:
        yield
    finally:
        viproc.init()


def pending(v, steps=20):
    """v behind a chain of exact copies (np.positive): most likely still
    pending when the next op is issued, so a sync on it shows in the counters
    (and the op is issued on a pending input, as in a program running ahead)."""
    for _ in range(steps):
        v = np.positive(v)
    return v


def counters():
    s = viproc.offload_stats()
    s["fallbacks"] = viproc._ndarray.fallback_calls
    return s


def assert_async(fn, *args, ops=1, small_steps=False, syncs=0, min_tasks=1, **kwargs):
    """Runs fn(*args, **kwargs) and checks that it took the asynchronous path:
    `ops` ops issued as tasks, none run inline, no fallback to NumPy, no sync
    point, no NumPy array as a result. Returns the result.

    syncs: the most blocking sync points allowed (a composition that decides
    something from a small result, as NumPy's code does: allclose's bool, the
    warning for an all-NaN slice). A sync on a result that happened to be
    finished does not count, hence "at most".

    small_steps: the op is a composition with small intermediate results (the
    count of mean, the scalars of var): such a step takes the direct path, or
    NumPy's scalar math (no op at all), when its input happens to be finished
    already, or runs inline as a task (an in-place step on a buffer still being
    read). Then `ops` is the most ops the composition issues (tasks and direct
    ops together) and at least `min_tasks` tasks must leave the main thread
    (0 where the op itself may run directly once its inputs are ready)."""
    before = counters()  # the inputs may be pending: a sync on them counts
    r = fn(*args, **kwargs)
    viproc.wait_all()  # not a sync point of its own (not counted)
    after = counters()
    delta = {k: after[k] - before[k] for k in before}
    results = r if isinstance(r, tuple) else () if r is None else (r,)  # None: a statement
    # Arrays are viproc arrays, never an np.ndarray (that would be a sync
    # through __array__). Where NumPy returns a scalar, viproc returns NumPy's
    # scalar if the data happened to be there already, else a pending 0-d
    # array; metadata (flags, device) are plain Python values.
    assert not any(isinstance(x, np.ndarray) for x in results), [type(x) for x in results]
    assert delta["fallbacks"] == 0, f"fell back to NumPy: {delta}"
    assert delta["syncs"] <= syncs, f"synced an array: {delta}"
    if small_steps:
        assert delta["tasks"] - delta["inline"] >= min_tasks, (
            f"no task left the main thread: {delta}")
        assert delta["tasks"] + delta["direct"] <= ops, f"expected at most {ops} ops: {delta}"
    else:
        assert delta["inline"] == 0, f"ran inline on the main thread: {delta}"
        assert delta["direct"] == 0, f"ran directly on the main thread: {delta}"
        assert delta["tasks"] == ops, f"issued {delta['tasks']} tasks, expected {ops}: {delta}"
    return r
