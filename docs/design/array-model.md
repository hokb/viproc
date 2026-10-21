# Array data model

This document describes the target data model of a viproc array: which
objects make up an array, who creates, reads, writes and releases them, and
which checks keep the data consistent when the main thread only issues ops
and workers compute them. It builds on the execution model in CLAUDE.md,
`memory-management.md` (ownership, pools, copy-on-write, NumPy interop) and
`pipelining.md` (stages, tasks advancing them; open in PR #23).

Status labels as in `memory-management.md`: **Decided**, **Implemented**,
**Proposed**. The whole model is **Proposed**; most of it maps onto
existing code (see "Mapping to the current code").

## Summary

| # | Decision | Status |
|---|----------|--------|
| A1 | An array is four objects: Array, Storage (size descriptor + task), AliasGroup (shared by a base and its views), CountableArray (one data version, one handle per device) | Proposed |
| A2 | The size descriptor (SD) is a flat, pooled block fixed to its Storage; operands reference it instead of copying it | Proposed |
| A3 | One task per NumPy op (the Accelerator's *awaiter*); the main thread always creates the Array and an empty Storage, the task configures the op's outputs | Decided (pipelining S2) |
| A4 | Copy-on-write is register renaming: the AliasGroup switches to a new CountableArray; readers keep the version they captured | Implemented (as `Storage` / `Buffer`) |
| A5 | The Array, not the Storage, holds the AliasGroup | Proposed |
| A6 | An array is not a NumPy `ndarray` (rejected alternative) | Decided |

## A1: objects

```
Python name ──► Array A ──► Storage S_A ── SD, Task
                   │
                   └──────► AliasGroup G ── current ──► CountableArray V ── handle[dev0], handle[dev1], …
                   ┌──────►      ▲                       (one data version)   async_reads, host_views
Python name ──► Array B ──► Storage S_B ── SD, Task
               (view of A)

Task T ── operands: (Storage, CountableArray captured at issue), …
```

- **Array** (A): the Python-visible object. Main thread only. Holds its
  Storage and its AliasGroup.
- **Storage** (S): the layout of one array or view: its size descriptor
  (SD) and the task of the op that configures it. One per array and per
  view. Created empty by the main thread; complete once its task reached
  *allocated*, immutable after that.
- **AliasGroup** (G): the node a base array and all its views share: the
  current data version (`current`) and the last task writing it
  (`lastWriter`, which decides `ready` / `pending`). Main thread only. A
  write through any view changes only G, so the base and all other views
  see it without viproc knowing the views. In register-renaming terms G is
  the architectural register.
- **CountableArray** (V): one data version, the "memory": one handle per
  device, each empty or a valid copy of the same values (the physical
  register). `async_reads` and `host_views` live here, because readers read
  a version and all views share the version. Which device a reader uses is
  decided later, by its task (phase 2: cost functions); a copy to another
  device fills a handle of the *same* version and changes no values.
- **Task** (T): one per op invocation, the Accelerator's *awaiter*. Each
  operand holds the Storage (for the SD) and the CountableArray the op was
  issued against (all device copies of that version). A task never reaches
  its data through Storage → AliasGroup: after a copy-on-write, G points to
  a newer version, the task must still read its own.

### Mapping to the current code

| this document | current code | difference |
|---|---|---|
| Array + Storage | `Array` with its `Layout` | SD as a pooled block with the task of its op (A2, A3) |
| AliasGroup | `Storage` | same role: current buffer, producer |
| CountableArray | `Buffer` | one handle per device instead of one pointer |
| Task | `Task` | operands reference the SD instead of copying a `Layout` |

## A2: the size descriptor

After ILNumerics' buffer size descriptor (`ILNumerics.ONAL`,
`StorageLayer/Size/Size.cs`): one flat block
`{ndim, dtype, flags | numel | offset | lengths[ndim] | strides[ndim]}`,
packed (strides right after the `ndim` lengths).

- **Fixed and pooled with its Storage.** A block for NumPy's maximum rank
  (`NPY_MAXDIMS` = 64: 3 + 128 fields, ~1 KB) costs no allocation and no
  cache traffic beyond the `3 + 2·ndim` fields in use; only the pool blocks
  get larger (more pages and TLB entries per live array).
- **Referenced, not copied, by operands.** Today every operand carries a
  copy of the `Layout` (it outlives the array). With pipelining a copy at
  issue is impossible anyway: an input that is only *created* has no SD
  yet, the consumer reads it after the input's stage event.
- **Strides in bytes**, as NumPy (`view(dtype)`, `real`/`imag`,
  `as_strided` give strides that are no multiple of the item size).
- **Published by stage, without a lock.** The task writes the lengths
  (*size completed*) and the strides (*allocated*) with plain stores, then
  stores the stage with `memory_order_release`; a thread that read the
  stage with `memory_order_acquire` sees the fields complete (as
  `TaskState` today). Before *allocated* only the header and the lengths
  may be read or copied.

## A3: tasks configure their outputs

The main thread **always** creates the Array and its Storage (empty,
stage *created*), and the AliasGroup and CountableArray of a new output
(handles empty). The task of the op does everything else on its outputs:
size, device, memory, strides, values (`pipelining.md`, S2). An in-place
write creates no new Storage: the writing task becomes `G.lastWriter`.

## Python assignments

NumPy has reference semantics and so does a viproc array: a view writes
into its base. There are no lazy-copy views as in ILNumerics; copies happen
where NumPy copies, and invisibly as copy-on-write for asynchronous readers.

| Python | semantics | effect on the data model |
|---|---|---|
| `b = a` | second name, same object | none (Python refcount of A) |
| `b = a[1:]`, `a.T`, `a.reshape(...)` as a view | reference to the same data | new Array and Storage (own SD), G shared |
| `b = a.copy()`, `np.array(a)` | value | new Array, Storage, G, V; a copy task reads V |
| `a = a + 1` | new result, name rebound | new objects; the old A dies, the result may take over its block (elision) |
| `a += 1`, `np.add(a, 1, out=a)` | mutation | in/out op on G; all views see the result |
| `a[k] = x` | partial mutation | see "Mutation" |
| `s = a.sum()`, then `s += 1` | value (NumPy scalars are immutable) | ready: NumPy scalar; pending: 0-d array marked `_scalar`, `+=` rebinds |

## Issuing an op: checks per argument

Notation: `S` the argument's Storage, `G` its AliasGroup, `V = G.current`.

**Input only**
1. SD not yet at the stage the op needs? Register on that stage of
   `S.task` (pipelining).
2. `G.lastWriter` pending? Register on its *completed*: the op becomes a
   continuation.
3. **Always** (pending or ready) `V.async_reads++`; the operand holds V. A
   writer issued later must also protect readers that have not started.

**In/out**
1. `G.lastWriter` pending? Register on it (write after write). Then
   `G.lastWriter := T`.
2. Other readers? `V.async_reads > 0`, `V.host_views > 0`, or an input of
   T reads G through another view (`a += a[::-1]`). The op's own reads are
   not counted.
3. Yes: copy-on-write. No: in place.

**New output**: Array, empty Storage (task T), new G with an empty V. T
fills the SD, picks the device, allocates (or takes over the block of a
dying input), computes.

### A4: copy-on-write as register renaming

`G.current := V'`, a new CountableArray with empty handles. T reads V and
writes V'; older readers keep V through their operands until they finish.

- **Without a copy** when A is the only view of G, covers all of it and T
  writes every element.
- **Otherwise with a copy:** T copies V into V' right before its kernel;
  if nobody reads V any more by then, V' takes over V's block instead.
- Never attach a new AliasGroup to A: its views would stay on the old one
  and miss the write.

Example: T1 reads `a` (operand holds `(S_a, V)`, `V.async_reads == 1`);
`b = a[1:]` (new Array and Storage, G shared); `b[0] = -1`: `async_reads > 0`
gives copy-on-write, `G.current := V'`, the write task copies V into V' and
writes -1. `a` and `b` reach their data through G and see V'; T1 reads V
through its operand and sees the old values.

## Ops that change no elements

`.T`, `swapaxes`, `reshape` as a view, basic indexing: new Array, new
Storage, G shared. Nothing is read: no `async_reads++`, no wait on
`G.lastWriter`.

- The view's SD is computed on the main thread when A is *allocated*
  (strides known), else by a task once A reaches *allocated*.
- **Open: `reshape`.** Whether NumPy returns a view or a copy depends on
  the strides, and so does whether the result shares G. The main thread
  must know that at issue, because later ops find their dependencies
  through G. Proposal: keep predicting strides at issue (NumPy's layout
  rules are deterministic), even when memory comes at *allocated*; where
  that is impossible, `reshape` waits for A's *allocated*.

## Mutation (`a[key] = value`, ILNumerics' SetRange)

The target is A or a view; the procedure is the same, since everything
shared lives in G.

Checks (main thread, at issue):
1. **Read-only** target (`broadcast_to`, part of an export): NumPy's error,
   at once.
2. **Key and value:** pending viproc arrays become dependencies (callbacks,
   `async_reads++` on their versions). The value is cast to the target
   dtype. The key kind picks the kernel: basic → assign, integer arrays →
   scatter, boolean mask → masked assignment.
3. **Write after write:** `G.lastWriter` pending? Register; then
   `G.lastWriter := T`.
4. **Write after read:** `V.async_reads > 0` or `V.host_views > 0`?
   Copy-on-write.
5. **Overlap:** value or key read G (`a[1:] = a[:-1]`)? Copy-on-write too;
   T reads the old V.
6. **Shortcut:** everything ready, no readers, small target: NumPy's own
   slot writes on the main thread, no task.

Actions:
- **Only G changes:** `lastWriter`, and `current` on copy-on-write. The
  Arrays and Storages of A and all views stay as they are; their SDs
  (offsets, strides) stay valid in the new version, since copy-on-write
  keeps the layout.
- **All views become pending automatically:** readiness is read from
  `G.lastWriter`, not from the Storage.
- **Copy-on-write for a partial write always copies:** T copies the whole
  old version before its kernel, then writes its part. No copy only when
  the target is the only view, covers everything and every element is
  written (`a[...] = x`); scatter and masks never write everything.
- **Run-time errors** (index out of bounds) are reported asynchronously
  (`error-handling.md`); G keeps the error, also after a later overwrite.

## Lifecycle

Everything is created by the main thread when it issues an op. Workers
create nothing; they fill in (SD, memory, values) and release. What a
worker may release has an atomic count; what only the main thread touches
has a plain one.

| part | created | access | held by | released: who, when |
|---|---|---|---|---|
| Array A | main thread, at issue: op output, view, import, creation function | main thread only (Python object, GIL) | Python names and containers | main thread, when the Python refcount reaches 0; drops S and G |
| Storage S | main thread, with A, empty (*created*), from the pool; SD fixed to it | write: T in stage order (lengths at *size completed*, strides at *allocated*), the main thread only where it knows the SD at issue (today: always); read: main thread and tasks holding S as an operand, after the stage signal | A; task operands (inputs and outputs); S holds its task until *completed* | any thread, with the last reference (atomic): A is dead and no operand holds S; back to the pool |
| AliasGroup G | main thread, with the base array (output, import, copy); views share it | main thread only: sets and reads `current`, `lastWriter` (whether the task completed: its atomic state) | A and all its views (Arrays, not Storages) | main thread, when the last array of the group dies; drops `current` and `lastWriter` (the latter as soon as it sees it completed) |
| CountableArray V | main thread, for a new output or on copy-on-write; handles empty | handles: the writing task allocates right before its kernel, device copies fill further handles (atomic slots), on the task's thread; `async_reads`: main thread ++, reading task −−; `host_views`: main thread | G (`current`), operands of reading and writing tasks, NumPy exports | any thread, with the last reference (atomic): G points to a newer version or is dead, and no operand or export holds V; device memory to that device's pool, foreign memory (imported ndarray) through the main thread |
| Task T | main thread, at issue, one per op, from the pool (the direct path for small ready data creates none) | main thread: issue, callback registration, waits at sync points; workers (or the main thread inline): advance stages, run the kernel | callbacks of its inputs (as consumer), the Storages of its outputs, G as `lastWriter`, the thread running it | any thread, with the last reference; on *completed* T releases its operands (`async_reads` −−) and callbacks, S and G drop T once they see it completed; back to the pool |

**Cycles.** S ↔ T (T holds S as an output, S holds T as its task) breaks
on *completed*: T drops its operands, S drops T.

### A5: the Array holds the AliasGroup

A Storage can die on a worker (the last reference from an operand). If it
held G, a task's operand would keep G alive, and with it `G.current` = V,
after A died: V would always have one reference more than the task's, and
the task could never take over a dying input's block (run-time elision,
`memory-management.md` D11). G would also need an atomic count although no
worker needs G. An atomic count is not free: an uncontended `lock xadd` is a
full barrier on x86 (~5-10 ns), and a count that both the main thread and a
worker change moves its cache line between cores (~40-100 ns per transfer
on one socket, read-for-ownership under MESI), on the main thread, which
issues every op and is the bottleneck. So A holds S and G, G stays a
main-thread object, and atomic counts stay on S, V and T, which workers do
release.

## Sync points

- Values (`print`, `float`, export): wait for *completed* of
  `G.lastWriter`.
- Metadata (`shape`): wait for *size completed* of `S.task`.

## A6: rejected alternative, an array as a NumPy `ndarray` plus a task

Each of these rules it out:

1. An `ndarray` needs a shape, strides and a data pointer when it is
   created: no stage *created* (pipelining S1/S4), and element memory
   taken at issue on the main thread (D10, superseded because dependent
   chains got 2-4x slower than NumPy).
2. Workers cannot touch it: its references are Python refcounts (GIL).
3. NumPy views have their own data pointers; copy-on-write would have to
   find and patch every view. Readiness, producer and `async_reads` need a
   shared node anyway.
4. Handing the `ndarray` to the user is not effortless but unsafe: NumPy
   calls on it bypass viproc and race with the workers (D7, and the
   decision against an `ndarray` subclass, D9).
5. One host pointer, no device copies.

Handing data to NumPy is copy-free already (D7: read-only exports counted
as readers, adoption of fresh memory). NumPy's custom allocator (NEP 49,
`PyDataMem_SetHandler`) would only let NumPy allocate fallback results from
viproc's element pool; it is set per context and applies to every NumPy
allocation there, so it would have to be set around viproc's own fallback
calls only. Not needed for this model.
