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
| S1 | Six stages per array, each reached exactly once, in increasing order | Implemented for tasks (`TaskState`, stages 1-5; the direct path has no task) |
| S2 | The main thread guarantees only `Created`; tasks advance the rest | Decided, not implemented (today: `SizeCompleted` at issue) |
| S3 | Tasks react to every stage of their inputs, not only `Completed` | Signals implemented (three per registration, `Task::add_consumer`); no task reacts to SizeCompleted/Allocated yet |
| S4 | Pending size inputs make a task, whatever the size | Decided, not implemented (today: sync) |
| S5 | An array in stage S gives access to the properties of stages 1..S; handlers run only on their stage signal (no locks) | Decided, not implemented |
| S6 | `Disposed` is never observed; checked by sanitizer builds, debug assertions, counters and versions | Proposed |

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
Every task reaches each of them in order: `SizeCompleted` at issue,
`DeviceSelected` (the CPU) and `Allocated` in `Task::execute` right before
the kernel, `Completed` after it; `Task::advance` asserts in debug builds
that a state is the next one. The direct path has no task and sets none.
Stage pipelining belongs to optimization level O2 (`goals.md`) where it
can change results (partial results starting their consumers); stages
that change nothing (S1, S4) hold at every level.

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

**Signals (implemented).** A consumer registered with a producer
(`Task::add_consumer`) receives **exactly three signals** from it, each
exactly once and in this order: **size completed**, **allocated**,
**completed** (`Signal`). Device selected is a state of its own (S1) but
not signalled while the CPU is the only device. The protocol is the same
whatever the producer's state at registration, so the hybrid model below
needs no special case:

- Ops that know their size early (today: every op, at issue) signal size
  completed long before their elements exist; allocated follows right
  before the kernel, completed after it.
- Ops that only know their size when they run (a NumPy function taken as a
  whole, a data-dependent result; O2, not implemented yet) give all three
  in quick succession on the thread that ran them.

Each registration is a node in the producer's consumer list with one
signal word (`ConsumerNode::signals`: the last signal requested, the last
delivered). Two threads can find a signal due: the producer advancing
(it stores its state, then walks the list, both seq_cst) and the main
thread registering (it pushes the node, then loads the producer's state,
both seq_cst), so at least one of them sees the other. Whoever raises the
request while nothing is being delivered delivers, up to whatever is
requested by the time it is done; the other one only raises the request.
No lock, no signal lost, none twice, none out of order. Stages the
producer passed before the node was published are delivered by the
registering thread; a size known at registration is delivered before the
node is published (no atomic: counted with the issue guard). The consumer
counts outstanding signals per stage (`sized_pending_`,
`allocated_pending_`, `pending_`); the last completed signal makes it
runnable, and the thread that delivered it runs it as a continuation or
hands it to the pool's queue (small ones stay local, large ones go to
the pool, as before). Reactions to the size completed and allocated
signals of all inputs are the place for S2/S4 (a task computing its own
size, allocating early); nothing reacts to them yet. Debug builds assert
that a task's states advance one at a time and that a task runs only after
every input's three signals. Cost (callgrind, 1e4-element chains and
fan-outs): main thread unchanged within noise, worker +46 to +50
instructions per task (two CAS per consumer for allocated).

**Today.** Every asynchronous op is a NumPy function taken as a whole: its
result shape is computed on the main thread at issue, so every task reaches
`SizeCompleted` at issue (`runtime.cpp`, `issue`); consumers receive the
three signals, but only completed triggers anything. Ops whose result size
depends on data (`nonzero`, boolean indexing, `unique`) fall back to eager
NumPy after a sync.

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

**Enforced by construction, not by locks.** Fine-grained locks per stage or
property are hard to implement and to maintain. Instead:

- **Stage handlers.** Every function that works on an array in a stage (a
  task's size-completed handler computing its output size, its allocation,
  its kernel) declares the stage it requires, as an attribute of the
  function (a template parameter or tag type in C++). It reads only
  properties of that stage or lower ones.
- **Called on the signal only.** A handler is called only as the direct
  consequence of the stage event it requires being signalled (S2, S3), so
  the stage is reached when it runs: no lock, no check of the state on the
  hot path. Keep this structural (stage handlers are reachable only from
  the event dispatch) so that refactoring and CI tooling can verify it.
- **Checks.** Debug builds assert the required stage at the entry of each
  handler. A CI check (a static analysis over the handlers, e.g. a
  clang-tidy rule or typed accessors that take a stage token) reports any
  access to a property of a higher stage inside a handler as an error.

Blocking remains only where the main thread waits at a sync point or a
metadata query: it waits for the stage event it needs (as `wait_completed`
does for `Completed`), not on a lock of the array.

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
4. **Versions (generation counters).** Every pooled object (`Task`,
   `Buffer`, `ArrayBlock`) carries an integer that `FixedPool` increments
   with each allocation of its block. A holder of a non-owning reference (a
   raw pointer, a callback entry, an operand) stores the version next to it
   and checks it on access: a changed version means the object was disposed
   and its block reused, and the access fails at once instead of reading an
   unrelated live object. Unlike 1 and 2, this catches reuse within the pool
   precisely and is cheap (one store per allocation, one compare per
   checked access), so it could stay enabled in more builds than the
   sanitizer.

