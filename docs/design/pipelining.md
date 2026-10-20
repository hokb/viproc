# Stage pipelining in viproc

This document describes the full stage pipelining scheme of the ILNumerics
Accelerator and how viproc gets there. CLAUDE.md summarizes the execution
model; this document is the reference for array stages, partial
completion and data-dependent sizes.

Status labels as in `memory-management.md`: **Decided**, **Implemented**,
**Proposed**.

## Summary

| # | Decision | Status |
|---|----------|--------|
| S1 | Six stages per array, each reached exactly once, in increasing order | Decided; stages 1, 2, 4, 5 exist (`TaskState`) |
| S2 | The main thread guarantees only `Created`; tasks advance the rest | Decided, not implemented (today: `SizeCompleted` at issue) |
| S3 | Tasks react to every stage of their inputs, not only `Completed` | Decided, not implemented |
| S4 | Pending size inputs make a task, whatever the size | Decided, not implemented (today: sync) |
| S5 | Properties of an array block until it reached their stage | Decided, not implemented |

## S1: stages

An array (with the task producing it) reaches each of six stages **exactly
once**, in **monotonically increasing order**: no stage is skipped, none is
repeated, none is reached before the previous one. A stage with nothing to
do is still reached (and signalled to consumers) right after the previous
one:

1. **Created**: the array exists. Nothing else is known: no size, no shape,
   no storage, no elements. Subsequent array operations in user code accept
   an array in any stage, including **created**, as a regular input: they
   are issued at once and register for the stage events they need.
2. **Size completed**: shape (size) known.
3. **Device selected**: where the elements will live and be computed
   (phase 1: always the CPU, reached right after size completed).
4. **Allocated**: element memory exists.
5. **Completed**: the elements are computed; the array is `ready`.
6. **Disposed**: the last reference is gone, the memory is released.

`TaskState` (`src/runtime/task.hpp`) has `Created`, `SizeCompleted`,
`DeviceSelected`, `Allocated` and `Completed`; `Disposed` is implicit today
(reference counting releases the buffer, D2/D11 in `memory-management.md`).
Today a task may skip states (`DeviceSelected` is unused, the direct path
sets none); the scheme requires every stage to be reached.

## S2, S3: who advances the stages

**Full scheme (Accelerator).** The main thread guarantees only stage 1: it
creates the task and the output arrays and returns. Everything after that
happens in the task, driven by stage events of its inputs:

- When all inputs reached **size completed**, the task computes its own
  output size(s) (broadcasting, reduction axes, ...) and signals **size
  completed** to the consumers of its outputs, which react the same way.
  Sizes propagate down a chain of pending ops before any element is
  computed.
- **Device selected** and **allocated** follow in the task (allocation is
  therefore practically always done by tasks, not by the main thread; D11
  already does this for element memory).
- **Completed** when the kernel ran.

Only the small-array shortcut (direct path, `Runtime::issue_direct`) has the
main thread do all stages itself, since there is nothing to overlap.

What it buys:
- **The main thread runs further ahead.** An unknown size no longer forces a
  sync; only a branch condition that depends on it does (`if x.shape[0] >
  n:`, `len(x)` in a loop bound).
- **Cost decisions move into tasks.** Device selection and memory copies
  (future devices, copies between them) happen when sizes are known, in the
  task, hidden behind other work.

**Today.** Every asynchronous op is a NumPy function taken as a whole: its
result shape is computed on the main thread at issue, so every task reaches
`SizeCompleted` at issue (`runtime.cpp`, `issue`), and consumers only
register for `Completed`. Ops whose result size depends on data
(`nonzero`, boolean indexing, `unique`) fall back to eager NumPy after a
sync.

## S4: pending size inputs

A creator whose size argument is a pending viproc array (`np.zeros(n)`,
`np.arange(k)`, `np.empty((n, m))` with pending `n`) issues a task, whatever
the size will be. Its output is created empty (stage 1 only). The task waits
for the size input's **completed** (the size is that input's *data*),
computes the shape, advances its output to **size completed**, allocates and
fills. Today such calls sync on the size argument.

This is the first op with an output that is not size-completed at issue, so
it needs S3 for every consumer: an op issued on such an array computes its
own output shape in its task, on the size-completed event of its inputs,
instead of at issue.

## S5: access to properties of an incomplete array

Whoever reads a property of an array needs the array to have reached the
stage that provides it, or blocks until it has:

| property | needs stage |
|---|---|
| `shape`, `size`, `ndim`, `len()`, strides/layout | size completed |
| element memory (export, fallback) | allocated + completed (writers) |
| element values (sync points) | completed |

The wait is per stage (a stage event the waiter registers for, as
`wait_completed` does for `Completed`), not a general lock on the array.

Open points for the implementation:
- **Thread rules.** `Array` and `Storage` are main-thread only today, and the
  layout lives there. With S2, a task writes the layout of its outputs:
  publish it with the stage (release on advance, acquire on read), so the
  main thread reads it without a lock once it saw the stage.
- **Issue path cost (G3).** A consumer of a size-pending input computes its
  shape in the task; consumers of size-completed inputs keep computing it
  at issue (no extra work on the common path).
- **dtype** stays known at issue (dtype resolution does not depend on data).
- **Data-dependent results** (`nonzero`, boolean indexing, `unique`) become
  tasks too, once S3 exists: their size is known when their kernel ran.
