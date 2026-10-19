# Error handling in viproc

This document collects the design decisions about errors: where they arise,
how a failed op is recorded, when and how the error reaches Python, and what
the user sees. CLAUDE.md summarizes the execution model; this document is the
reference for everything error-related and records *why* each decision was
taken. Issue: hokb/viproc#18.

Status labels as in `memory-management.md` (Decided, Implemented, Proposed).

## Summary

| # | Decision | Status |
|---|----------|--------|
| E1 | Issue-time errors are raised at the call; run-time errors may arise on any thread | Implemented |
| E2 | A failed op stores its error on its task and outputs; consumers inherit it | Implemented |
| E3 | Asynchronous reporting: every call into viproc reads the runtime's error store | Implemented |
| E4 | NumPy's exception types, as subclasses (`AsyncValueError`, `AsyncIndexError`) | Implemented |
| E5 | Where: the exact issue site and the NumPy function, not the full call stack | Implemented |
| E6 | Loops that raise through the Python API are watched on the thread that runs them | Implemented |
| E7 | Accepted differences from NumPy | Decided |
| E8 | Floating-point errors (`np.errstate`) through the same channel | Proposed |

## E1 — Two kinds of errors

**Issue-time errors** do not depend on element values: broadcasting and
shape errors, dtypes NumPy cannot resolve, invalid arguments (`axis`,
`order`, ...). The main thread knows everything they depend on when it
issues the op (shapes are a stage of their own and known at issue in phase
1), so they are raised right there, at the call, with NumPy's own error:
viproc lets NumPy raise them (dtype resolution, a zero-stride probe for
views, the fallback). Nothing is asynchronous about them.

**Run-time errors** depend on element values and arise while a kernel runs,
on whatever thread runs it (a worker, or the main thread for inline and
direct ops): an index out of bounds (gather, scatter), an integer power with
a negative exponent (E6), later linalg errors (`LinAlgError` from the FP
flags of `_umath_linalg`) and errors requested through `np.errstate` (E8).
Data-dependent result shapes (`nonzero`, boolean masks) still sync and run
in NumPy (CLAUDE.md, "Eventing"), so their errors are issue-time errors.

Kernels report a run-time error as a message `"TypeName: text"` when the
Python exception type is known (`"IndexError: index 12000 is out of bounds
..."`, `"ValueError: Integers to negative integer powers are not allowed."`),
else as plain text.

## E2 — Recording a failed op

`TaskError` (`src/runtime/task.hpp`): the message, the issue site, the op
(`Kernel::name()`) and the op's issue number (`id`). It is created once, by
the thread that ran the kernel, and shared (`TaskErrorPtr`, a
`shared_ptr<const TaskError>`) by:

- the failed task (`Task::error()`) and, through `Storage::last_error()`,
  every array it writes, new outputs and in/out arrays alike;
- every consumer: a task whose input failed skips its kernel and completes
  with the same error (`upstream_error_`), so `((x ** y) + 1).sum()` fails
  with the error of `**`, naming where `**` was issued;
- the runtime's error store (E3), for the op that failed itself only.

A sync point on any of these arrays raises the error (`vp_array_wait`,
`vp_array_prepare_write`), every time: their data is not NumPy's.

## E3 — Asynchronous reporting

The thread that runs a failing kernel reports the error to the runtime
(`ErrorReports::report`, a mutex-protected list and an atomic flag) **before
the task completes**: a sync point that sees the task completed finds the
report too, so it can mark it as reported (`forget`) and nothing is raised
twice.

The bridge reads the store **at every call into viproc** from Python
(`check_async_errors` in `_viproc.c`: in `check_runtime`, which every C
entry point calls, and in `fast_ufunc`, which every operator and ufunc call
goes through). While no op failed this is one atomic load
(`vp_error_pending`), measured at ~25 instructions per op with callgrind
(`x * y`, 8 elements: ~5,400 instructions; about 0.5%). After a failure it
takes the store (`vp_take_error`) and raises:

- the error of the **earliest issued** failed op (the one sequential NumPy
  would have raised first), with the number of further failures not reported
  (`"[2 more failed ops not reported]"`); all reports are cleared then;
- once: a later call raises nothing until another op fails;
- deterministically at `viproc.wait_all()` (after waiting) and at shutdown:
  `viproc.shutdown()` / `viproc.init()` raise an error nobody reported, and
  at interpreter exit Python prints it ("Exception ignored in atexit
  callback"); the exit status stays 0.

Which call raises a worker's error depends on timing (the first call after
the failure); that the error is raised does not: the next sync point on a
result of the failed op, `wait_all` or shutdown raises it at the latest.

The bridge must never swallow such an error: Python code in `_ndarray.py`
that catches `TypeError`/`ValueError`/`IndexError` around viproc calls to
fall back to NumPy re-raises `AsyncError` first.

## E4 — Exception types

`viproc.AsyncError` (a `RuntimeError`) is the base. Where the kernel names
the type, viproc raises a subclass of both `AsyncError` and NumPy's type:
`AsyncIndexError` (`IndexError`) and `AsyncValueError` (`ValueError`). User
code catching NumPy's exception (`except ValueError:`) keeps working; code
that wants to know that the error came from an earlier op catches
`AsyncError`. The message is NumPy's, followed by the op and where it was
issued:

```
viproc.AsyncValueError: Integers to negative integer powers are not allowed.
    (in numpy.power, issued at prog.py:12 in step)
```

## E5 — Where the op was issued, not the full call stack

The Python traceback of the raised exception shows the call that raised it,
which for an asynchronous error is a later one. The message therefore names
the op that failed:

- **the issue site**: file, line and function of the first stack frame
  outside viproc and NumPy (the user's code), captured by the main thread
  at issue (`issue_site()` in `_viproc.c`), cached per (code object,
  bytecode offset) and interned once, so every task carries a pointer
  (`Task::issue_site()`, no per-op cost);
- **the op**: the NumPy function the task runs (`numpy.power`,
  `numpy.add.reduce`, `numpy.matmul`, `numpy.where`, `astype`, `indexing
  (a[index])`, ...), from `Kernel::name()`, taken only when the op fails.

Both travel with the task to whichever thread runs it and come back in the
`TaskError`.

**The full main-thread call stack is not captured.** Walking the Python
frames and keeping them (or formatting them) on every op would cost far more
than the whole per-op budget: the site lookup is ~100 instructions because
it is cached per bytecode position, but a stack depends on all callers and
cannot be cached; keeping frame objects alive would also keep their locals
alive (arrays, memory) until the op completes. The issue site plus the op
name identify the failing instruction exactly. An opt-in debug mode that
records the stack per op (for example `VIPROC_TRACEBACK=1`) would fit the
same channel; not implemented.

## E6 — Loops that raise through the Python API

NumPy's signed integer `power` loops are legacy loops: for a negative
exponent they call `npy_gil_error(PyExc_ValueError, ...)`, which takes the
GIL (`PyGILState_Ensure`), sets the exception on the calling thread's Python
thread state, releases the GIL and returns; the loop then reports success
(0). NumPy itself checks `PyErr_Occurred()` after the call. These loops do
not set `requires_pyapi`, so viproc used to run them on workers: the error
was lost (a worker had no thread state: `PyGILState_Ensure` created a
temporary one and `PyGILState_Release` deleted it, error included) and the
op returned a result; on the main thread the error leaked (`SystemError`).
The NumPy bug (undocumented; a draft issue is in the project files,
`numpy-issue-int-power/`) is the loop's contract: a strided loop must return
-1 on failure.

viproc watches these loops (`UfuncLoop::raises_through_pyapi`, set at
resolution for `power` with a signed integer exponent) on the thread that
runs them (`UfuncLoop::call_watched`):

- The thread needs a Python thread state that outlives the call. The main
  thread has one; a worker creates its own on first use
  (`PyThreadState_New`, no GIL needed) and keeps it (thread-local, deleted
  when the worker exits: the runtime is destroyed with the GIL released, so
  the worker can take it then). `npy_gil_error`'s `PyGILState_Ensure` finds
  it, and the error stays on it.
- Before and after the call the thread reads its own thread state's
  current exception (`PyThreadState::current_exception`, CPython ≥ 3.12).
  No other thread touches that state, so this needs no GIL.
- Only if the loop raised (the error path, where `npy_gil_error` took the
  GIL already) the thread takes the GIL, fetches and clears the exception
  and fails the op with `"ValueError: <NumPy's message>"` (E1–E3).

Cost on success: a thread-local load and two pointer reads per loop call
(per op, or per buffer of NumPy's iterator). `int ** int` keeps running
asynchronously and in parallel.

**Rejected alternatives** (both were tried in hokb/viproc#17):

- *Scanning the exponents for negative values before the call.* An extra
  pass over the data on every integer power: 3–10% slower on large data,
  against G3; and with `out=` / `**=` it wrote nothing where NumPy writes the
  elements before the failing one.
- *Running integer power with an array exponent synchronously in NumPy*
  (eager fallback). NumPy-exact, but `int ** int`-array lost all
  parallelism: it ran on the main thread, serial, and was a sync point for
  its inputs.
- *Taking the GIL after every call to check `PyErr_Occurred()`.* The main
  thread holds the GIL while it runs ahead; a worker gets it only at the
  switch interval (5 ms).

## E7 — Accepted differences from NumPy

Asynchronous execution cannot raise at the call that failed, because that
call returned long ago. Deliberately, and only for run-time errors (E1):

- **When:** the error is raised at a later call into viproc (E3), with
  NumPy's type (as a subclass) and message. Small ready ops run directly on
  the main thread report the same way, at the next call.
- **Ops issued after the failing one still run:** the main thread ran
  ahead, so later ops (in NumPy never reached) were issued and run if they do
  not depend on the failed one; their side effects (in-place writes to other
  arrays) happen.
- **Arrays written by the failed op are failed:** an in/out array (`out=`,
  `a **= b`) raises at every sync point afterwards (and so does everything
  computed from it), where NumPy keeps the elements its loop wrote before
  the error. An op that overwrites all of it (`a[...] = 0`, `np.add(x, 1,
  out=a)`) gives it valid data again: an in/out whose old data the op does
  not read (`Kernel::writes_all`, the whole buffer) does not pass its error
  on (`Runtime::issue`, step 1).
- **Exit status:** an error reported only at interpreter exit is printed,
  but the process exits with status 0.

## E8 — Floating-point errors (proposed)

NumPy reports divide-by-zero, overflow, underflow and invalid as warnings,
or raises `FloatingPointError` under `np.errstate(...='raise')`. Integer
division by zero is a warning too. Workers already record the FP flags per
task (`Task::fp_flags`); the plan: capture the `errstate` mode at issue (it
is thread-local in NumPy and belongs to the issuing code), and at reporting
time either warn (NumPy's `RuntimeWarning` text) or raise
`FloatingPointError`, through the store of E3, with the issue site of E5.
Linalg's `LinAlgError` (set through the invalid flag in NumPy's
`errstate(invalid='call')` callback) uses the same path.

## Lesson: why #17 first proposed a solution against G3

The first fix in #17 scanned the exponents before every integer power call.
It was correct and passed the tests, but it added a pass over the data to
a kernel: 3–10% on large data, against G3 ("never slower than NumPy"). The
second fix ran the op synchronously in NumPy: exact, but it gave up
parallelism for every `int ** int`-array to handle a case that almost never
happens. How it happened:

1. **The fix was judged by correctness only.** The change touched a kernel
   but was not treated as a performance change, so neither the benchmark
   suite nor a before/after timing was run before pushing; the cost came to
   light only when the maintainer asked.
2. **The error channel was missing.** viproc had no way for a loop that
   raises through Python to report from a worker, so both fixes worked around
   the gap (prevent the error, or keep it off the workers) instead of
   closing it.
3. **A rare case was paid for on the common path.** The failure needs an
   invalid element; the scan cost every valid call.

Rules that follow (also in CLAUDE.md, "Conventions"):

- Every change to a kernel, the issue path or the bridge's per-op code is a
  performance change: measure it against `main` before pushing (callgrind
  instruction counts for per-op costs, timings for data-sized costs) and
  state the numbers in the PR.
- Never add a pass over the data (a scan, check or copy) to detect a rare
  condition. Detect errors where they occur, in the kernel, and report them
  through the asynchronous channel (E3).
- A fallback to eager NumPy is not a fix for a supported op: it costs the
  op's parallelism on every call.
