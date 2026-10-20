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
| S6 | `Disposed` is never observed; checked by sanitizer builds, debug assertions and counters | Proposed |

## S1: stages

An array (with the task producing it) reaches each of six stages **exactly
once**, in **monotonically increasing order**: no stage is skipped, none is
repeated, none is reached before the previous one. A stage with nothing to
do is still reached (and signalled to consumers) right after the previous
one. This is an implementation guarantee, not a convention: handlers of a
stage may rely on being called for it, and a consumer waiting for a stage
that is skipped (or a handler run twice) can deadlock.


1. **Created**: the array exists. Nothing else is known: no size, no shape,
   no storage, no elements. Subsequent array operations in user code accept
   an array in any stage, including **created**, as a regular input: they
   are issued at once and register for the stage events they need.
2. **Size completed**: size known: `ndim`, extents, offset (not the
   strides).
3. **Device selected**: where the elements will live and be computed
   (phase 1: always the CPU, reached right after size completed).
4. **Allocated**: element memory exists, and with it the strides.
5. **Completed**: the elements are computed; the array is `ready`.
6. **Disposed**: the last reference is gone, the memory is released. An
   array in this stage is **never observed**: observing it needs a
   reference, and reference counting disposes it only when none is left
   (S6).

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
  completed** to the consumers of its outputs, which **may** react the same
  way. Sizes propagate down a chain of pending ops before any element is
  computed, as far as the tasks on it support partial eventing.
- **Device selected** and **allocated** follow in the task (allocation is
  therefore practically always done by tasks, not by the main thread; D11
  already does this for element memory).
- **Completed** when the kernel ran.

Tasks may or may not support partial eventing. A task that does not (linalg,
fft, NumPy loops taken as a whole) ignores the intermediate events of its
inputs and reacts only when the last input reaches **completed**. It still
must still signal every stage to its consumers, exactly once and in this
order (S1): **size completed**, **device selected**, **allocated**,
**completed**, typically one right after the other when it runs. Its
consumers see the same sequence of events either way.

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

Every stage, once reached, opens access to the properties it guarantees.
Since stages are reached in strict order (S1), an array in stage S allows
access to the properties of stages 1 to S, and to nothing beyond them.
Whoever (the main thread or a consuming task) reads a property of an array
needs the array to have reached its stage, or blocks until it has:

| stage | opens access to | access allowed in this stage |
|---|---|---|
| 1 created | the array object (as an input of later ops, S1) | the array object |
| 2 size completed | size: `ndim`, extents (`shape`, `size`, `len()`), offset | + size |
| 3 device selected | device selection | + device |
| 4 allocated | storage memory (addresses, not values) and **strides** | + memory, strides |
| 5 completed | element values (sync points, exports, fallbacks) | everything |

Strides belong to **allocated**, not to size completed: they describe the
memory, which the task chooses when it allocates. A consumer in stage size
completed may read the size only, never the strides.

The wait is per stage (a stage event the waiter registers for, as
`wait_completed` does for `Completed`; individual stage locks), not a
general lock on the array.

Open points for the implementation:
- **Thread rules.** `Array` and `Storage` are main-thread only today, and the
  layout lives there. With S2, a task writes the size and the strides of its
  outputs: publish each with its stage (release on advance, acquire on
  read), so the main thread reads it without a lock once it saw the stage.
- **Strides at issue today.** Today the main thread fixes the result layout
  at issue (NumPy's layout, `Layout::keep_order`; views and copy-on-write
  derive from it). With strides in **allocated**, ops that need an input's
  strides (views, the result layout of order K, reduction plans, iterator
  templates) wait for that input's allocated stage, in their task.
- **Issue path cost (G3).** A consumer of a size-pending input computes its
  shape in the task; consumers of size-completed inputs keep computing it
  at issue (no extra work on the common path).
- **dtype** stays known at issue (dtype resolution does not depend on data).
- **Data-dependent results** (`nonzero`, boolean indexing, `unique`) become
  tasks too, once S3 exists: their size is known when their kernel ran.

## S6: checking that `Disposed` is never observed

By construction, nothing can observe a disposed array, task or buffer: an
observer needs a reference, and the last reference is gone. A bug breaks
this as a use after release (a raw pointer kept past its reference, a
release too many). Pools make such bugs silent: a released `Task`, `Buffer`
or `ArrayBlock` goes back to `FixedPool` and is handed out again, so a stale
access reads a live, unrelated object instead of crashing. Proposed checks,
none of them in Release builds (G3):

1. **Sanitizer builds.** With AddressSanitizer, `FixedPool` passes every
   allocation to `operator new`/`delete` (as the element pool already does
   with `VIPROC_ELEMENT_POOL=OFF`), or poisons blocks on its free lists
   (`ASAN_POISON_MEMORY_REGION`). Every access to a disposed object is then
   an ASan error. A CI job runs ctest (C++ tests and pytest) on such a
   build.
2. **Debug assertions.** Debug builds give `Task`, `Buffer` and
   `Array`/`Storage` a stage (or magic) field that the destructor sets to
   `Disposed`; accessors and stage transitions assert it is not, and
   `FixedPool` fills released blocks with a pattern so a stale read fails
   the assertion. The existing forward-only assertion of task states
   (`task.cpp`) becomes "exactly the next stage" once every stage is
   reached (S1).
3. **Counters.** Debug builds count created and disposed objects per type;
   tests check at runtime shutdown that both match (each object disposed
   exactly once, nothing leaked); a second dispose fails assertion 2.

