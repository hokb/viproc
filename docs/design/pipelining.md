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
| S2 | The main thread guarantees only `Created`; tasks advance the rest | Implemented for element-wise ufuncs at O2 (`Runtime::issue_deferred`, size and layout steps in the task); other ops and O1: `SizeCompleted` at issue |
| S3 | Tasks react to every stage of their inputs, not only `Completed` | Signals implemented (three per registration, `Task::add_consumer`); no task reacts to SizeCompleted/Allocated yet |
| S4 | Pending size inputs make a task, whatever the size | Implemented at O2 for `zeros`, `ones`, `empty`, `full`, `arange(stop)` (module functions) and data-dependent sizes (`x[mask]`, `flatnonzero`, 1-d `nonzero`); O1 and the rest: sync |
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
size, allocating early); deferred tasks react to them (below). Debug builds assert
that a task's states advance one at a time and that a task runs only after
every input's three signals. Cost (callgrind, 1e4-element chains and
fan-outs): main thread unchanged within noise, worker +46 to +50
instructions per task (two CAS per consumer for allocated).

**Deferred sizing (runtime, implemented).** `Runtime::issue_deferred`
issues a task whose outputs the main thread creates without a size: dtype
only, a buffer without size or memory (`CountableArray::unsized`), the
metadata pointing to the task. The task then takes the stages itself
(`Task::configures`):

- **Size step:** when the last input's size completed signal arrives, the
  task calls `Kernel::size` (output shapes from the input layouts), writes
  the shapes and signals size completed. An error there (shapes that do not
  broadcast) is recorded as the op's error (`size_error`, an
  `ErrorRef` in the runtime's table) and inherited by every consumer, as a
  run-time error: issue-time errors of deferred ops become asynchronous.
  An input's size error is taken over without calling the kernel.
- **Layout step:** when the last input's allocated signal arrives (and the
  task's own size is set), `Kernel::layout` fixes the strides (the shapes
  stay; the main thread may read them since size completed), the buffers
  get their sizes, and the task signals device selected and allocated. The
  physical block is still taken when the kernel runs (D11: the dying
  input's block or the pool's).
- **Run:** completed signals of all inputs, as before.

Steps never nest: a signal that makes a step due only records the task on
the delivering thread's worklist (`Task::step_due`); a worker runs the list
after its task (`Runtime::due_steps`), the main thread hands it to the pool
as step jobs (a task in the pool's queue with the `Job::Steps` bit; the
same task may also sit there for its run, `Job::Run`, which takes
precedence). The steps of one task are serialized by a combining word
(`Task::run_steps`), and `execute` finishes any step not yet run before the
kernel. Metadata queries of the main thread wait for the stage they need
(`Runtime::wait_stage`: shape at size completed, strides at allocated);
an op issued on an unsized input on the regular path still waits for its
allocated stage (it needs the layout; `Array::layout` waits, so the main
thread never reads an incomplete one). A step job can wait in the queue
behind running kernels (all workers busy): a metadata query that finds its
stage not reached first runs the step jobs the main thread queued itself
(`Runtime::help_steps`, `main_steps_`; the combining word makes the queued
job a no-op), and the steps they make due; it waits only if that does not
reach the stage. Found on a 2-core CI runner: one worker in a 10 ms kernel
held every metadata query (and an `astype` issued on a deferred output)
until the kernel was done (`test_deferred_size_wait_helps`). The layout step needs the inputs'
strides, not their memory: an input sized at issue counts as laid out at
registration, so a deferred op behind regular ones is laid out at once,
not when they run (else a regular op's allocated signal, right before its
kernel, would hold the main thread at the next metadata query). A chain of 50 deferred ops is sized
while its first producer still runs (`test_deferred_sizes_run_ahead`).
Cost at O1 (callgrind, chains and fan-outs of 1e4 elements, 1 worker):
main thread unchanged within noise, worker +25 to +44 instructions per
task over the signals alone (one more check per run).

**Element-wise ufuncs at O2 (implemented).** The bridge's ufunc fast path
issues an element-wise ufunc with one output and no casts through
`vp_ufunc_issue_deferred` when an input is pending or the result is larger
than the direct path's threshold: the main thread reads no input size. The
shared element-wise kernel sizes (NumPy's broadcasting and message) and
lays out (NumPy's result layout, `numpy::elementwise_layout`) the result;
it runs NumPy's loop in one call where NumPy makes one (dense operands:
identical results), else per row (strided, broadcast, reversed: NumPy's
iterator would need the GIL to plan; values may round differently). A
0-d result is known to be NumPy's scalar without its size: its inputs were
all 0-d (unsized ones marked so). `x.shape`, `x.ndim`, `x.size` wait for
the size only and raise the op's error if it could not size the array (as
a sync point, every time); `offload_stats()["size_waits"]` counts the
metadata queries that blocked. `out=`, several outputs and the other ops
take the regular path (casts: see "Casts at O2"). Cost at O1 (callgrind, 1 worker): main
thread +10 to +30 instructions per op (the layout check of every
`Array::layout` call), direct path +12.

**Ranks at issue.** Extents may be unknown at issue, the number of
dimensions never is: element-wise results have the largest input rank,
creators the shape's length, `x[mask]` `x.ndim - mask.ndim + 1`,
`flatnonzero` 1, reductions the input's rank less the removed axes. The
caller passes it to `issue_deferred` (stored in the Metadata, checked
against the size step in debug builds); `Runtime::rank` / `vp_array_rank`
read it without a wait, `x.ndim` no longer waits, and ops that need only
the rank at issue (axis normalization) are issued without one. ILNumerics'
Accelerator does the same split: `Segment.SetSize` computes "number and
length of dimensions, but no strides, yet" when all input sizes are
there, with one size rule per function category (`broadcasting_size`,
`aggregating_size`).

**Reductions at O2 (implemented).** `sum`, `prod`, `max`, `min`, `any`,
`all` (methods and module functions with `axis`/`keepdims`, the C path)
of pending or larger data go through `vp_reduce_deferred`: the axes are
normalized against the rank at issue, the loop is resolved for the input
dtype with unknown fixed strides (`UfuncLoop::kAnyStride`), and the shared
kernel (`make_deferred_reduce_kernel`, cached per ufunc, dtype, rank, axes,
keepdims and identity) sizes the result (the aggregation rule), lays it out
C-contiguously (NumPy's fast-path result) and plans the loop calls when it
runs (`plan_reduction` on the input's final layout). One reduced axis after
coalescing gives NumPy's loop calls (identical results, tested for C, F and
strided inputs); several that do not coalesce are visited one by one where
NumPy's iterator would buffer (floating-point sums may round differently,
O2). A reduced axis of extent 0 without an identity fails the size step
with NumPy's ValueError (asynchronously). Reductions that cast (int32
sums accumulate in int64), `dtype=`, `initial=`, `out=`, `where=` take the
regular path. Cost at O1 (callgrind, `x.sum(axis=0)` of 100 elements):
unchanged (8,060 vs. 7,820 instructions per call, noise).

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
fills. At O1 such calls sync on the size argument.

**Implemented at O2** for `viproc.zeros`, `ones`, `empty`, `full` (a
scalar fill value; order C or F) and `arange(stop)`, with sizes that are
pending 0-d int32/int64 viproc arrays (in a shape tuple next to ints):
`vp_create_deferred` issues the task through `Runtime::issue_deferred`
with a kernel that sizes from data (`Kernel::sizes_from_data`,
`size_from_data`): its size step does not run on its inputs' size signals
but in its run, once the sizes are computed; the three signals then
follow each other. A negative extent fails the op with NumPy's ValueError
(`AsyncValueError` at the next metadata query or sync point); `arange` of
a negative stop is empty, as NumPy's. Consumers issued on the result are
deferred too (element-wise ufuncs) or wait at their issue for its layout
(the others). Ready sizes and NumPy's own `np.zeros` (which takes the size
through `operator.index`) still sync.

A running configuring task finishes its steps before its kernel, and on a
worker also the steps they make due (`Runtime::configure_now`): otherwise a
consumer's layout step would wait in the worker's list until the kernel is
done (a view of `np.sin(x)` of 8e6 elements waited 244 ms). They are its
consumers' steps, whose kernels cannot run before this one anyway. Handed
to the pool instead, they waited for a free worker: with one worker (2
cores) a main thread waiting for a consumer's layout (`astype` of
`sin(sin(x))[:4] * 0`) slept through both kernels, since `help_steps` runs
only the step jobs the main thread queued itself.

**Data-dependent result sizes at O2.** The same mechanism serves ops whose
result size is their input's *data*: `x[mask]` with a pending bool mask
over x's leading axes (`vp_mask_select`: the True positions counted in the
size step, the subarrays copied in C order in the run; NumPy's IndexError
for mismatched shapes, asynchronously), `np.flatnonzero(a)` and
`np.nonzero(a)` of a pending 1-d `a` (`vp_flatnonzero`). Their size step
is one pass over the mask (count), the run a second one (copy): the size
is signalled before the elements, and consumers issued meanwhile size
themselves on it. At O1 they sync on the mask. n-d `nonzero` (NumPy's
index arrays are column views of one array), `argwhere`, `unique` and
boolean masks in tuples still sync.

**Views at O2.** Basic indexing (`x[i]`, `x[a:b:c]`, `None`, `...`) and
transposes (`x.T`, `x.transpose(...)`, `np.transpose`) of an array its task
still sizes are views by a rule (`vp_array_view_deferred`,
`Runtime::view_deferred`, `ViewRule` in `task.hpp`): the bridge parses the
key against the rank only (slices as `PySlice_Unpack` gives them, indices
as given) and the runtime keeps the steps (keep, slice, index, new axis) or
the permutation. The view's metadata has its rank, its dtype and the base's
producer as its task; nothing else is known. Its shape follows from the
base's shape, its strides and offset from the base's layout, each derived
once (`std::call_once` per stage, the shape never written again after the
first stage) by whoever needs it first: a task's size or layout step
reading the view as an input, or a main-thread wait on it (`wait_stage`).
A view of a view derives its base first. An index out of bounds is the
view's own error (NumPy's IndexError message, recorded once with the
view's issue site and op name `indexing`): its consumers inherit it,
sync points raise it (`AsyncIndexError`). A **reshape of a 1-d** such
array (`a.reshape`, `np.reshape`, any order) is a view by a rule too
(`vp_array_reshape_deferred`): a 1-d array always reshapes without a copy
(`_attempt_nocopy_reshape`), so the decision NumPy makes from the strides,
view or copy, is known at issue; the unknown dimension and NumPy's size
mismatch ValueError follow at the size step, the strides from
`reshape_view` at the layout step. An n-d reshape still waits for the
base's layout: whether it is a view (aliasing, `base`) depends on the
strides, and a view cannot leave its alias group later. Keys that consume more axes
than the rank, non-basic keys and keys giving a 0-d result (NumPy's
scalar) take the regular path and wait for the base's layout. Index
assignment into such an array still waits. Cost at O1 (callgrind, `a[1:3]`
and `a[5]` of 100 elements): ~5 instructions per subscript, the level test
(2,216 -> 2,227 per pair).

**Casts at O2 (implemented).** An element-wise ufunc whose loop casts an
input (`int32 + float64`, `bool * float32`, integer `divide`) is issued
deferred too: the shared element-wise kernel takes the loop's dtypes
(`UfuncLoop::dtypes`) for the result layout and, in the task, casts every
input of another dtype row by row, in chunks of 8,192 elements, into
per-thread buffers before the loop call (a broadcast input once per row).
`astype` (orders K, C, F; not complex to real, which warns per call) is a
deferred task of its own (`DeferredCastKernel`, kernel per (dtype, order)
cached in the runtime): its size step copies the shape, its layout step is
`numpy::astype_layout` (`PyArray_NewLikeArray`'s order; all strides 0 for
an empty array, as NumPy), its run casts along the output's fastest axis.
The element casts are NumPy's: NumPy 2 leaves the legacy
`ArrFuncs->cast` table empty, so a cast is a ranged, buffered `NpyIter`
template per (from, to) over 8,192 contiguous elements, copied per thread
from the template's pool and pointed at the data
(`NpyIter_ResetBasePointers`, no GIL), the converted elements copied out
of its buffer (`numpy::cast_elements`). Every reset of a buffered iterator
copies its current range from its current base pointers: a range change
(a short last chunk) is therefore made on pointers to a static block of
zeros (no FP flags), never on the pointers of an earlier call, which may
be freed. FP flags of the casts (NaN to int: "invalid value encountered in
cast") are reported as NumPy's. Cost at O1 (callgrind, `x * y` of 8
elements, `z.astype(np.float32)` of 100): unchanged within the noise
(< 10 instructions per op).

**Not yet:** matmul, gathers and the other regular-path ops
read their inputs' layouts at issue, so one issued on an array a task
still sizes (S2/S4) waits at its issue for that array's layout (for an
S4 array: until its size input is computed). Making them deferred
(`Kernel::size`/`layout` per kernel, the NumPy iterator templates planned
without the GIL) is the next step.

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

