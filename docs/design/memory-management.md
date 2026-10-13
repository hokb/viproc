# Memory management in viproc

This document collects the design decisions about memory in viproc: who owns
what, when memory is released, how it is reused, and how viproc memory is
shared with NumPy. CLAUDE.md summarizes the execution model; this document
is the reference for everything memory-related and records *why* each
decision was taken.

Status labels:

- **Decided** — agreed, not implemented yet.
- **Implemented** — in the code (file references given).
- **Proposed** — worked out, waiting for a decision.

## Summary

| # | Decision | Status |
|---|----------|--------|
| D1 | No dedicated (temporary) array types; one `viproc.ndarray` type | Decided |
| D2 | Deterministic memory management through reference counting | Decided; implemented for objects |
| D3 | Own memory pool for element memory in the viproc layer | Decided |
| D4 | Pools for the per-op runtime objects (Task, Buffer, ArrayBlock) | Implemented |
| D5 | Reuse the buffer of a temporary input for the output | Proposed |
| D6 | Ownership: Python owns objects, the runtime owns element memory | Implemented |
| D7 | Copy-free conversion between viproc arrays and NumPy arrays | Decided (rule proposed) |
| D8 | Write-after-read safety by copy-on-write, not by locking | Implemented |

## Background: the ILNumerics model

ILNumerics (.NET; checked against the API documentation of the NuGet
packages `ILNumerics.Computing` / `ILNumerics.Core` 7.4.62) manages memory
with dedicated array types:

- `RetArray<T>`: return type of all numerical functions, released after its
  first use. The implicit "persistence cast" to `Array<T>` keeps the inner
  storage ("The inner storage is kept and used for the new array"), i.e.
  converts without a copy.
- `Array<T>`: local array, lives until the end of the enclosing
  `using (Scope.Enter(...))` block; clones are lazy copy-on-write.
- `InArray<T>` / `OutArray<T>`: immutable input and output parameter types.
- `Scope`: "no memory is left as garbage after the scope block is left".
  Released buffers return to a memory pool, so an expression like
  `a + c * 2 / abs(sin(c) * -b / log(a))` needs about twice the memory of
  one array.

The purpose is deterministic release on a garbage-collected runtime: the
.NET GC frees memory at an unknown later time, so without these types large
temporaries would pile up until the next collection and could not be
reused.

## D1 — No dedicated (temporary) array types

**Decision.** viproc has one array type, `viproc.ndarray`. There is no
temporary/return type, no input/output parameter type and no scope API.

**Why.**

- CPython's reference counting already is deterministic. A temporary such
  as `sin(c)` in `a + c * 2 / abs(sin(c))` is released the moment the
  consuming op drops its reference; locals are released when the function
  returns. That is exactly what `RetArray` and `Scope` provide on .NET,
  without any type or scope in user code.
- Dedicated types would rule out running unchanged NumPy programs
  (`python -m viproc script.py`), a goal of the Python bridge.
- What the type system tells ILNumerics ("this value is used once") can be
  read from the reference count at runtime where it matters (D5).

**Limits of determinism in CPython** (rare for arrays, accepted): reference
cycles (released by the cycle collector), frames kept alive by a traceback,
`_` in the interactive shell. Alternative Python implementations without
reference counting (PyPy) are out of scope.

## D2 — Deterministic memory management

**Decision.** Every object and every buffer is released when its last
reference goes, never by a collector or at an unknown later time.

| Object | Counted by | Released |
|--------|-----------|----------|
| `viproc.ndarray` (Python object) | CPython refcount | when the program drops it |
| `Array`, `Storage` (runtime, main thread only) | `Ref<T>`, plain int (`src/runtime/ref.hpp`) | with the last Python object / view referring to them |
| `Task` | `std::shared_ptr` (atomic) | after completion, when no array names it as producer |
| `Buffer` (element memory) | `std::shared_ptr` (atomic) | when no array and no issued task (operand) refers to it |

Asynchrony is part of the rule: a Python object can die while issued tasks
still read its data. The tasks hold the `Buffer` through their operands, so
the memory lives exactly until the last reader completes, then it is
released on that thread. Cycles between arrays and tasks are broken on
`completed` (CLAUDE.md, "Reference counting").

Python objects whose memory a task still uses (wrapped NumPy arrays) are
released on the main thread only: a worker dropping the last reference
parks them in a graveyard drained at the next C ABI call
(`src/capi/capi.cpp`).

## D3 — Own memory pool for element memory

**Decision.** Element memory comes from a pool in the viproc layer. Buffers
released by D2 go back to the pool and are reused by later ops of the same
size class, instead of being returned to the system allocator.

**Why.** Fresh element memory costs more than the computation for large
arrays. Above the mmap threshold of glibc's malloc (128 KiB by default)
every allocation maps new pages: one page fault per 4 KiB page on first
write, and `munmap` on release. Page faults and mmap/munmap also take the
process-wide memory-map lock in the kernel, so they serialize workers and
limit strong scaling for large arrays. Measured, chain
`((x + 1) * 2) + 3` on float64 (Release, 1 worker):

| Elements | NumPy, new temporaries | NumPy, reused buffer (`out=`) | viproc (current, no pool) |
|---------:|-----------------------:|------------------------------:|--------------------------:|
| 1e5 | 0.48 ms | 0.06 ms | 0.30 ms |
| 1e6 | 3.8 ms  | 1.4 ms  | 2.8 ms  |
| 1e7 | 55 ms   | 23 ms   | 110 ms  |

**Design (to be implemented in `src/runtime/`):**

- **Where it plugs in.** `Buffer::ensure_allocated()` takes memory from the
  pool; `Buffer::~Buffer()` returns it. Buffers wrapping foreign memory
  (NumPy arrays) are not pooled. Element memory up to 128 bytes stays
  inline in the `Buffer` object (already implemented, `buffer.hpp`).
- **Threads.** Memory is allocated by the producing task (any worker or the
  main thread) and released by whichever thread drops the last reference.
  The pool is therefore thread-safe: one free list per size class with its
  own lock (an allocation of this size costs far more than an uncontended
  lock). A per-thread cache in front is an option if measurements show
  contention.
- **Size classes.** Geometric, two classes per power of two
  (2^k and 1.5 · 2^k), so the waste is at most ~33 %. Requests are rounded
  up to their class; the buffer remembers its class.
- **Alignment.** 64 bytes, as today (SIMD loops, no false sharing between
  buffers).
- **Large blocks.** Allocated with the transparent-huge-page hint
  (`madvise(MADV_HUGEPAGE)` on Linux) above a threshold (e.g. 2 MiB), which
  cuts the number of page faults by 512x.
- **Bound.** The pool keeps at most a configured number of cached bytes
  (default: a fraction of physical memory; `VIPROC_POOL_LIMIT` to
  override). Beyond it, released memory goes back to the system,
  largest/oldest classes first. `viproc.trim()` (and the runtime shutdown)
  releases everything cached.
- **Determinism is kept.** Release into the pool happens at the same moment
  as before; only *where* the memory goes changes. Peak memory equals the
  live data plus the pool bound.
- **Debugging.** A build option disables the pool (direct allocation), so
  AddressSanitizer keeps catching use-after-free in element memory.

**Relation to the look-ahead limit.** The limit of 1000 active tasks also
bounds how much element memory pending ops can hold. A memory-based limit
(bytes held by pending outputs) is the natural next step once the pool
knows the totals.

## D4 — Pools for the per-op runtime objects

**Implemented** (`src/runtime/pool.hpp`). `Task`, `Buffer` objects and
`ArrayBlock`s (an `Array` and its `Storage` in one allocation) come from
fixed-size pools: per-thread free lists without atomics for the allocating
thread; other threads return blocks through a lock-free stack, which an
allocating thread takes over as a whole (exchange, so no ABA). The pools
only grow, up to the peak number of live objects.

Per op this leaves one pooled `Task`, one pooled `Buffer`, one pooled
`ArrayBlock` and the Python object; together with D2's other measures the
per-op cost dropped from ~7,100 to ~5,400 instructions (CLAUDE.md, "Python
bridge").

## D5 — Reuse the buffer of a temporary input (proposed)

The runtime counterpart of ILNumerics' persistence cast, without a type:
when an input of an element-wise op is a temporary, the op writes its
result into that input's buffer instead of a new one.

An input counts as temporary when, at issue time on the main thread:

1. its Python object is referenced only by the interpreter's value stack
   (refcount 1) and the call comes straight from the interpreter (C code
   may hold borrowed references; NumPy's own temporary elision checks the
   call chain for the same reason),
2. its `Array` and `Storage` have no other references (no views),
3. no issued task still reads its buffer (`async_reads == 0`; a pending
   *producer* is fine, the op is its consumer anyway),
4. dtype, shape and layout equal the output's (element-wise ops may read
   and write the same element in one loop call, as NumPy does in place).

Then the op is issued with the input as an in/out operand (no new buffer,
no copy-on-write since nothing else reads it). This halves the memory of
expression chains and avoids most pool traffic for them. NumPy does the
same above 256 KiB; in viproc it is a local issue-time decision, like
everything else.

## D6 — Ownership

**Implemented.**

- **Python objects** (`viproc.ndarray`, `numpy.ndarray`) are owned by
  Python and its reference counting.
- **Element memory** is owned by the runtime (`Buffer`), whether viproc
  allocated it or adopted it from a NumPy array. Adopted memory stays
  alive through a reference to its ndarray held by the `Buffer`.
- **`Array` / `Storage`** are runtime objects reachable only from the main
  thread; workers see `Task`s and `Buffer`s only.
- **The C ABI** hands out `Array*` as `vp_array*`, one reference each;
  bridges never see C++ ownership types.

## D7 — Copy-free conversion to and from NumPy

**Decided:** conversion between viproc arrays and NumPy arrays copies no
element data, in both directions. **Proposed:** the rule below that keeps
results identical to NumPy.

Today `asarray` copies NumPy input and `__array__` / `numpy()` return a
copy; fallback results are already wrapped without a copy.

NumPy loops need no protection: the runtime calls a loop only when all its
inputs are `completed` and its outputs are allocated, and writes in place
only without pending readers (D8). The risk is Python code holding a NumPy
array on the same memory and accessing it outside the task protocol:

```python
b = np.asarray(x)   # copy-free: b views the viproc buffer
y = x * 2           # issued asynchronously, reads the buffer later
b[0] = 5            # main thread writes now -> y may see 5 (NumPy: no)
```

```python
x = np.ones(5)
v = viproc.asarray(x)   # adopted without a copy
x[0] = 7                # user still holds x -> pending tasks on v see it
```

"A function no longer uses an array it returned" does not hold for views
(`a[1:]`, `a.T`, `reshape`, `df.values` into pandas' internal blocks,
`np.frombuffer`, memmaps, cached arrays). Python's reference count tells
exactly whether anybody else can still touch the memory: an outside
holder, or a view, which references its base.

**Proposed rule** (local, per storage):

- A storage that shares memory with a NumPy array keeps a reference to that
  ndarray object.
- **Export** (`__array__`, `numpy()`): wait for the producer and for
  `async_reads == 0`, then return a writable ndarray view of the buffer.
- **Import** (`asarray`, fallback results): adopt the ndarray's memory.
- **Per op touching such a storage:** if the ndarray is referenced outside
  viproc (refcount above viproc's own), the op runs synchronously: it waits
  for its inputs, runs inline and writes in place (no copy-on-write
  switch). Exactly NumPy semantics, views included. Once the outside
  references are gone, ops on the storage are asynchronous again.
- Cost: one refcount check per op, only for storages shared with NumPy.
  Short-lived exports (`print(x)`, `np.asarray(x)` inside a library call)
  cost nothing afterwards; an ndarray kept next to a viproc array costs
  asynchrony for that array, never correctness.
- Out of reach: C code keeping a raw data pointer without holding a
  reference, which breaks Python's rules anyway.

## D8 — Write-after-read safety: copy-on-write

**Implemented** (`Runtime::issue`, CLAUDE.md "Reference counting"). Each
buffer counts its issued, not yet completed readers (`async_reads`). An op
writing an array in place (in-place operators, `out=`, `__setitem__`)
writes the current buffer only if nobody still reads it and no input of the
op overlaps it through another view; otherwise the storage switches to a
new buffer and the old one lives on through its readers' references.
Single-view storages are *consolidated* (the op writes a fresh contiguous
buffer, no copy); storages with views get a copy task first. Decided at
issue time on the main thread, so layouts never change after they were
communicated. Known difference to NumPy: strides/flags of an array may
change after an in-place op (values never do).

## Open questions

- Pool bound default (fraction of RAM vs. fixed) and the eviction order.
- Per-thread caches in front of the element pool: only if contention shows
  up in `strong_scaling.py` with large arrays.
- Huge-page threshold and Windows equivalent (large pages need a
  privilege; probably plain `VirtualAlloc` reuse only).
- Memory-based look-ahead limit (bytes held by pending outputs) instead of,
  or in addition to, 1000 active tasks.
- D5's interpreter check: how to tell "called from the interpreter" cheaply
  on CPython 3.12+ (NumPy inspects the C call chain; a cheaper criterion
  may exist for the number-protocol slots).
