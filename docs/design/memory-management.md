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
| D3 | Own memory pool for element memory in the viproc layer | Implemented |
| D4 | Pools for the per-op runtime objects (Task, Buffer, ArrayBlock) | Implemented |
| D5 | Reuse the buffer of a temporary input for the output | Implemented |
| D6 | Ownership: Python owns objects, the runtime owns element memory | Implemented |
| D7 | Array conversions viproc ↔ ndarray: copy-free export, copy-free import of fresh memory, eager fallbacks with NumPy's semantics | Implemented |
| D8 | Write-after-read safety by copy-on-write, not by locking | Implemented |
| D9 | `viproc.ndarray` is a wrapper, not a subclass of `numpy.ndarray` | Decided, implemented |
| D10 | Access model: pool allocations on the main thread only, releases from any thread | Superseded by D11 for element memory; holds for D4 |
| D11 | Element memory as late as possible: the task producing a buffer takes its block when it runs, from a dying input or the pool | Implemented |

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
released by D2 go back to the pool and are reused by later ops instead of
being returned to the system allocator. A request is served by a cached
block of **equal or larger** size, as long as the excess stays below a
threshold (12.5 %, by quantized bins); otherwise a new block is allocated.

**Why: fresh memory costs page faults and zeroing.** A fresh block from
the OS is mapped lazily: the first write to each 4 KiB page faults, and the
kernel zeroes the page (a security requirement) before handing it over.
Measured (C, `out[i] = in[i] + 1`, 80 MB): fresh block 40 ms (19,532
faults), with `MADV_HUGEPAGE` 25 ms (625 faults), reused block 14 ms;
zeroing alone is about one memory pass (~8 ms), the rest is fault handling.
Huge pages remove most faults but not the zeroing; only reuse removes both.
Faults also take the kernel's per-process memory-map lock, so they
serialize workers.

Measuring this needs realistic patterns. glibc reuses blocks itself when
only one is alive at a time (its mmap threshold adapts up to 32 MB), so a
single-op loop shows no difference. With several live temporaries it trims
and re-faults memory on every iteration (Python, `((x + 1) * 2) + 3`,
float64):

| Elements | NumPy, temporaries | NumPy, `out=` reuse | viproc, no pool |
|---------:|-------------------:|--------------------:|----------------:|
| 1e5 | 0.61 ms (358 faults)   | 0.08 ms (0) | 0.75 ms (554)    |
| 1e6 | 3.8 ms (1,319 faults)  | 1.3 ms (0)  | 8.3 ms (5,827)   |
| 1e7 | 58 ms (1,875 faults)   | 18 ms (0)   | 108 ms (58,596)  |

(At 1e5 the reused buffer is also cache-warm; a LIFO pool has the same
advantage.)

**Design** (implemented: `src/runtime/element_pool.*`, `buffer.*`):

- **Where it plugs in.** The task producing a buffer takes its block when
  it runs, on whatever thread runs it (D11; ops run directly by the main
  thread take it there); `Buffer::~Buffer()` returns it, on whatever
  thread. Foreign memory (NumPy arrays) is never
  pooled. Element memory up to 128 bytes stays inline in the `Buffer`
  (implemented).
- **Matching: equal or larger, bounded excess — by quantized bins** (as
  ILNumerics.ONAL's `MemoryPool`). Requests are rounded up to a bin
  length: 8 bins per power of two (bin lengths `(8 + s) · 2^(k-3)`,
  s = 0..7), i.e. a block is at most 12.5 % larger than requested. All
  blocks of a bin have the same capacity, so lookup is O(1) (bit-scan of
  the size, index into an array of free lists) and any cached block of the
  bin fits. 12.5 % is the excess threshold; coarser bins (e.g. 4 per power
  of two, 25 %) are a configuration choice, not a different design.
- **LIFO per bin.** The most recently released block is reused first: it is
  most likely still in cache and its pages are mapped.
- **Capacity vs. size.** A `Buffer` keeps its logical size (`nbytes()`,
  what the layout covers) and the capacity of its block. Everything about
  the array's data uses the logical size (copy-on-write's "covers the
  whole buffer" test, `CopyKernel`); only the pool sees the capacity.
- **Not zeroed.** Pooled blocks keep old contents; op outputs overwrite
  every element. Creation functions that need zeros (`zeros`, once they
  allocate from the pool) clear explicitly (ONAL: `New(length, init:
  true)`).
- **Alignment.** 64 bytes below 64 KiB, 4 KiB (page, as ONAL's native
  pool) from 64 KiB, 2 MiB from 2 MiB, so large blocks share no pages and
  huge pages can back all of them.
- **Large blocks.** `MADV_HUGEPAGE` from 2 MiB (Linux; a hint). Windows:
  blocks from 2 MiB come from `VirtualAlloc` (page-aligned, lazily
  committed like on Linux). Large pages (`MEM_LARGE_PAGES`) are opt-in,
  `VIPROC_LARGE_PAGES=1`: used when the process holds the "Lock pages in
  memory" privilege (`SeLockMemoryPrivilege`, enabled once at first use)
  and the block is a whole number of large pages (always from 16 MiB on).
  Not by default: Windows commits and zeroes a large-page block inside
  `VirtualAlloc` (measured when the main thread still allocated at issue), ~30 ms per
  24 MB block (i7-12700K, Windows 11). Every pool miss stalled the
  run-ahead (`test_main_thread_runs_ahead`: 60 ops issued in ~100 ms
  instead of < 1 ms, 3 misses). Linux's huge pages fault in on the
  worker that first writes the block instead.
- **Bound.** At most a configured number of cached bytes (counted in
  capacity; default: half the physical memory, as ONAL; `VIPROC_POOL_LIMIT`
  — bytes, `K`/`M`/`G` suffix — to override; `viproc.set_pool_limit()`).
  A release that would exceed the bound returns its block to the system
  (the simplest policy; evicting older cached blocks instead is an open
  question). `viproc.trim()` and runtime shutdown empty the pool.
- **Process-wide.** One pool per process (`ElementPool::instance()`, never
  destroyed: blocks may be released at exit). It is trimmed when a runtime
  shuts down.
- **Out of memory.** If allocating a fresh block fails, the pool empties
  itself and retries once, then `std::bad_alloc` propagates. Not done yet:
  waiting for issued tasks (their completion releases buffers) before
  giving up, and raising `MemoryError` at the issuing op.
- **Statistics.** `viproc.pool_stats()`: hits, misses, cached bytes,
  releases dropped by the bound, limit (C ABI: `vp_pool_get_stats`).
- **Determinism is kept.** Release into the pool happens at the same moment
  as before; only *where* the memory goes changes. Peak memory: live data +
  outputs of running tasks (D11) + the pool bound.
- **Debugging.** `-DVIPROC_ELEMENT_POOL=OFF` sets the default bound to 0
  (no caching), so AddressSanitizer keeps catching use-after-free in
  element memory.

**Measured after implementation** (Release, same machine): the chain
`((x + 1) * 2) + 3`, 1 worker: 1e5 0.90 → 0.095 ms, 1e6 9.9 → 2.2 ms,
1e7 117 → 40 ms, page faults per iteration 554 / 5,827 / 58,596 → 0
(NumPy with temporaries: 0.61 / 3.8 / 58 ms). `strong_scaling.py`
(4 workers, 8 chains of length 5): 1e6 238 → 203 ms, 1e7 4.10 → 3.29 s.

**Taken from ILNumerics.ONAL** (`MemoryLayer/MemoryPool.cs`, MIT, read for
design only): quantized bins with 8 steps per power of two, one LIFO stack
per bin, rounding up on allocation, a byte bound of half the available
memory, page-aligned native blocks, optional clearing on request, and OOM
recovery by emptying the pool and retrying. **Not taken:** its growth of
the bound by 10 % on every overflowing release (the bound drifts upward;
viproc keeps a fixed bound) and its eviction order (alternately smallest
and largest bins; viproc prefers the least recently released blocks, which
are least likely to be reused soon). ONAL also contains a disabled attempt
to block the issuing thread when allocated bytes exceed the bound; the
comment notes it failed for memory-heavy algorithms whose pressure does not
come from pending work, and leaves memory pressure to OOM handling. A
memory-based look-ahead limit in viproc (below) must avoid that trap.

**Relation to the look-ahead limit.** The limit of 1000 active tasks also
bounds the memory pending ops hold. A memory-based limit (bytes reserved by
pending outputs, known exactly to the main thread, D10) may complement it;
it must count only memory held by *pending* outputs, never live data, or it
blocks programs that simply use a lot of memory (see ONAL's experience
above).

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

## D5 — Reuse the buffer of a temporary input

**Implemented** (`python/viproc/_viproc.c`: `elision_candidate`,
`is_temporary`; C ABI `vp_array_reusable_bytes`). The runtime counterpart
of ILNumerics' persistence cast, without a type: when an operand of an
element-wise **operator** (`+ - * / // % << >> & | ^ @`, unary `- + ~`) is
a temporary, the op writes its result into that operand's buffer instead of
a new one, and returns the operand's object. Function calls (`np.add(t, 1)`),
in-place operators and comparisons are not elided (as in NumPy).

An operand qualifies when, at issue time on the main thread:

1. **dtype and shape equal the output's** (no broadcasting into it; element
   -wise loops read and write the same element in one call, as NumPy does
   in place);
2. **the runtime may reuse its buffer** (`vp_array_reusable_bytes`): the
   `Array` has no other reference and its `Storage` no other array (no
   views), the buffer is pool memory (so never memory shared with NumPy),
   no issued task still reads it (`async_reads == 0`; a pending *producer*
   is fine, the op is its consumer), no read-only NumPy export of it is
   alive (`host_views == 0`, D7), and the array covers the buffer
   contiguously from offset 0;
3. **its Python object is a temporary**, referenced only by the
   interpreter's value stack:
   - CPython ≥ 3.14: `PyUnstable_Object_IsUniqueReferencedTemporary`
     (exact; mode `"exact"`).
   - CPython 3.12/3.13 on glibc and on Windows: reference count 1 **and**
     the operator was applied by the bytecode interpreter itself, not by C
     code that may hold a borrowed reference to an object owned elsewhere
     (mode `"backtrace"`). The return addresses after viproc's frames
     (`backtrace()` / `RtlCaptureStackBackTrace()`; module of an address
     via `dladdr()` / `GetModuleHandleExW()`) must start with a chain
     recorded once by running every operator on a probe object, entered
     two ways: module code and a function called from C. The frames both
     runs share are the runtime's operator entry and the interpreter's
     call site; below that they differ. That common prefix is the chain.
     Its length depends on how CPython was compiled: `PyNumber_Negative`
     tail-calls the slot (1 frame: the interpreter's call site);
     `PyNumber_Add` is 2 frames in Ubuntu's build, 3 in the `python:3.13`
     Docker image, where `binary_op1` is not inlined. A fixed count of two
     frames, used before, stopped inside the runtime there, so `sum()`
     matched and elided (caught by the test suite in that image). A third
     calibration run applies the operators through `operator.add` & Co.
     (C callers); a chain one of them matches would not reach the
     interpreter and is dropped. Recorded addresses work with the
     BOLT/PGO layout of distribution builds, where the interpreter's code
     lies outside its symbol's range. Stricter than NumPy: builtins in the
     runtime (`sum()`, `operator.add`) do not elide either. Only for
     buffers ≥ 256 KiB (a backtrace costs ~1–2 µs), as NumPy.
     Each binary operator is calibrated in three forms, `p + p`,
     `p + 1.0` and `1.0 + p`: they take different paths through the
     runtime's dispatch (`binary_op1`), and a PGO build (e.g. GitHub's
     `actions/setup-python` CPython, `--enable-optimizations`) gives
     each path its own call site; calibrated with `p + p` only, `x + 1.0`
     never elided there. `_viproc.elision_chains()` lists the calibrated
     chain lengths (39: 12 binary operators x 3 forms, 3 unary).
   - Otherwise (other C libraries, macOS with CPython < 3.14): off (mode
     `"off"`).
   `viproc._viproc.elision_mode()` reports the mode, `_viproc.elided()`
   counts elided ops.

**Views.** Rule 2 works on the storage, not on the Python object, so views
are covered: a temporary view of a named array (`x[::2] + 1`) shares its
storage with `x` and is never reused; neither is a temporary whose storage
still has another view (e.g. one kept in a list). A view that is the only
array left on its storage and covers its buffer contiguously from offset 0
(`(x * 2)[:] + 1`) is reused, correctly: nothing else can see that buffer.
Offset or strided views are not reused (conservative). Buffers exported to
NumPy (D7) fail rule 2: a live export counts as a reader. Read-only viproc
arrays (D7, "Eager fallbacks") are never reused either.

Then the op is issued with the operand as its `out=` array: an in/out op
writing in place. That needed one change to D8's accounting: an op does not
count its reads of the buffer it writes in place in `async_reads`
(`Operand::counted_read`); every later writer waits for it as the storage's
producer anyway. Before, a pending in-place op made the next in-place op on
the same array copy-on-write, and blocked the next elision.

**Measured** (Release, `((x + 1) * 2) + 3`, 1 worker; before = pool only):
1e5 0.118 → 0.067 ms, 1e6 2.0 → 1.45 ms, 1e7 39 → 17 ms (NumPy 0.40 /
3.6 / 60 ms); pool memory held after the loop 1e7: 266 → 88 MiB.
`strong_scaling.py` 1e7 (4 workers, 8 chains of length 5): 2.96 → 1.98 s,
7.5x NumPy (above 4x because NumPy pays page faults for its temporaries
and viproc writes in place, saving memory bandwidth).

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

## D7 — Array conversions: viproc ↔ ndarray

**Implemented** (`numpy::export_ndarray`, `Runtime::prepare_host_write`,
C ABI `vp_array_export`, `vp_array_prepare_write`, `vp_array_view`;
`python/viproc/_ndarray.py`: `_Host`). Exports to NumPy copy no element data;
imports copy no element data where the memory is provably nobody else's.
Values stay identical to NumPy in every case.

### Is NumPy only ever called by viproc?

No, and that decides the design. viproc controls every access to a viproc
array: NumPy loops run only inside tasks (inputs completed, outputs
allocated, writes in place only without pending readers, D8), or eagerly on
the main thread after a sync. But once a plain `numpy.ndarray` exists on
the same memory, anyone can call NumPy on it directly: user code
(`b[0] = 5`, `np.add(b, 1, out=b)`), libraries, C extensions. Those calls
do not pass through viproc (`numpy.ndarray` is NumPy's own type), and they
run on the main thread *while viproc's workers may still read or write the
same buffer*. Synchronous with respect to the main thread is not
synchronous with respect to the workers. So a copy-free conversion needs a
rule for that outside access.

### Export: read-only view that counts as a reader

`np.asarray(v)`, `v.__array__()` and `v.numpy()` wait for `v`'s producer,
then return a **read-only** NumPy view of the buffer (shape, strides and
offset of `v`; views of views work). Its base object holds the buffer and
counts as **one reader** of it (`Buffer::host_views`, main thread only)
until the last NumPy view on it is gone. `Buffer::has_readers()` is
`host_views > 0 || async_reads > 0`; copy-on-write (D8) and temporary
elision (D5) test that. Host views are kept apart from `async_reads`
because a task's readers are waited for (a worker finishes them), host
views are not (only the user can drop them). Hence:

- Nobody writes through the export (read-only), so concurrent viproc
  readers are safe.
- A later in-place write to `v` sees a pending reader and copies-on-write
  (D8): `v` moves to a new buffer, the export keeps the values it showed.
  These are exactly the semantics of the former copy, paid only when `v`
  is written while an export is alive.
- Temporary elision (D5) never reuses an exported buffer (rule 2).
- Short-lived exports (`print(v)`, `float(v)`, a NumPy function on `v`)
  cost nothing afterwards.
- `np.array(v)`, `__array__(copy=True)` and `v.numpy().copy()` give a
  writable copy; `__array__(copy=False)` is allowed now, `dtype=` changes
  cast (one copy).
- Inside the bridge, NumPy code that viproc runs on its own behalf gets
  *writable* views instead (next section).

Measured (1e7 float64): `np.asarray(v)` 18.8 ms → 0.001 ms; `np.sum(v)`
(eager fallback) 25 ms → 3.5 ms.

### Eager fallbacks: NumPy code on viproc memory, NumPy's semantics

Everything viproc does not run asynchronously (reductions, `np.*`
functions, advanced indexing, `ufunc.at`, mutating functions such as
`np.put`, …) runs eagerly in NumPy on the main thread (`_Host` in
`_ndarray.py`). The goal is NumPy's semantics exactly, including identity,
aliasing, writability and in-place writes, so the fallback hands NumPy the
viproc memory itself, not copies:

1. **Exclusive, writable export per viproc argument**
   (`vp_array_export(a, writable=1)` → `Runtime::prepare_host_write`): wait
   for the array's producer; if read-only exports of the buffer are alive
   (`host_views > 0`), copy-on-write first (a copy task, then wait) so
   those exports keep their values; finally wait until no issued task
   still reads the buffer (`Buffer::wait_task_readers`). From then on no
   worker touches the buffer: the main thread issues nothing while NumPy
   runs. A writable export is not counted as a reader. Read-only viproc
   arrays get a read-only export (NumPy then raises its own error).
2. **NumPy runs** on these views: `out=`, in-place functions (`np.put`,
   `np.place`, `np.putmask`, `np.fill_diagonal`, `np.copyto(where=)`,
   `np.add.at`), overlapping assignments (`x[idx] = x[::-1][idx]`) and
   NumPy's own overlap checks behave exactly as on an ndarray, because
   they are NumPy's.
3. **Results are mapped back** to what NumPy would have returned:

   | NumPy result | viproc result |
   |--------------|---------------|
   | a viproc argument's export itself (`np.atleast_1d(x)`, `out=`) | that viproc array (`r is x`, as in NumPy) |
   | shares memory with an export (`reshape`, `.T`, `ravel`, `split`, `diagonal`, `broadcast_to`, `sliding_window_view`, `real`/`imag`, …) | a viproc **view** of the same storage (`vp_array_view`, dtype views included), read-only if NumPy made it read-only |
   | a NumPy argument or a view of one | returned unchanged (a plain ndarray, as NumPy returns it) |
   | fresh memory (owns its data, no base) | adopted without a copy |
   | anything else (view viproc cannot express: unaligned, non-native byte order, unsupported dtype) | copied |

   Views are recognized by memory overlap (`np.may_share_memory`), not by
   the base chain: NumPy code builds views through other objects
   (`as_strided`'s `DummyArray`).
4. **The exports are made read-only** after the call, in case NumPy code
   kept a reference to one (a cache, an iterator): it can still read, but
   no longer write behind viproc's back.

**Read-only viproc arrays** (`_readonly`): views that NumPy returns
read-only (`broadcast_to`, `sliding_window_view`, …) stay read-only in
viproc. Writes raise NumPy's `ValueError("assignment destination is
read-only")` / `output array is read-only`; operators fall back to NumPy
for them, which raises the same error.

Remaining differences to NumPy:

- `np.broadcast_arrays` results: NumPy warns (`DeprecationWarning`) on the
  first write and then writes; viproc maps them as writable views without
  that warning.
- An export that NumPy code keeps beyond the call is read-only afterwards,
  where NumPy's would stay writable (step 4).
- Strides/flags after copy-on-write (D8) can differ; values never do.

### Import: adopt only memory nobody else holds

"A function no longer uses an array it returned" does not hold for views
(`a[1:]`, `a.T`, `df.values`, `np.frombuffer`, memmaps) or for functions
returning an argument (`np.atleast_1d(a)`), and a user's own array may be
written at any time. Adopted memory is therefore limited to arrays that
provably have no other user:

| Path | Conversion |
|------|-----------|
| Creation functions (`viproc.zeros`, `ones`, `arange`, …) | NumPy creates, viproc adopts (no copy) |
| Results of eager fallbacks | mapped as above: viproc argument, viproc view, NumPy argument unchanged, fresh memory adopted, otherwise copied |
| `viproc.asarray(nd)`, NumPy operand in a viproc op (`v + nd`) | copy: the caller still holds `nd` and may write it |
| `viproc.asarray(nd, dtype=other)` | one copy (the cast) |
| Unsupported dtypes / layouts | stay NumPy arrays (eager NumPy) |

**Considered, not taken:** sharing a user-held ndarray and running every
op on that storage synchronously while the ndarray is referenced outside
viproc (refcount check per op). Correct, but `x = viproc.asarray(a)` with
`a` still in scope (the common case) would lose all asynchrony for `x`;
one copy at import is cheaper than that. Writable exports *to the user*
with the same rule were rejected for the same reason and because
copy-on-write would have to be replaced by input copies for such storages.
Writable exports exist only for the duration of an eager fallback, while
the main thread issues nothing (above).

## D8 — Write-after-read safety: copy-on-write

**Implemented** (`Runtime::issue`, CLAUDE.md "Reference counting"). Each
buffer counts its issued, not yet completed readers (`async_reads`). An op
writing an array in place (in-place operators, `out=`, `__setitem__`)
writes the current buffer only if nobody still reads it and no input of the
op overlaps it through another view; otherwise the storage switches to a
new buffer and the old one lives on through its readers' references.
Single-view storages are *consolidated* (the op writes a fresh buffer with
the array's layout, no copy); storages with views get a copy task first. An op's reads
of the buffer it writes in place are not counted (D5): later writers wait
for it as the producer anyway. Decided at
issue time on the main thread, so layouts never change after they were
communicated. Arrays keep their strides, as in NumPy: until reductions ran
asynchronously, consolidation switched to a C-contiguous layout, which made
later sums of F-order arrays round differently from NumPy (the element
order of a pairwise sum follows the layout).

## D9 — A wrapper, not a subclass of `numpy.ndarray`

**Decided and implemented** (`python/viproc/_viproc.c`, `_ndarray.py`).
`viproc.ndarray` is its own type implementing NumPy's array protocols
(`__array_ufunc__`, NEP 13; `__array_function__`, NEP 18; `__array__`)
plus all Python operators. It does not derive from `numpy.ndarray`.

**Why not a subclass.** A subclass would pass `isinstance(x, np.ndarray)`
and `PyArray_Check` in library C code, and libraries would take it without
conversion. But that C code then reads and writes `PyArray_DATA(x)`
directly, bypassing every sync point:

1. **No sync point.** C code would read a pending array's memory before its
   producer completed (wrong values, silently), and writes would race with
   issued readers.
2. **Memory protection does not fit.** Bohrium's npbackend subclasses
   `ndarray` and protects memory (mprotect + signal handler) until a flush.
   In viproc, workers write the memory while the main thread must not read
   it; `mprotect` applies to the whole process, not per thread. It would
   need two mappings of every buffer (protected for the ndarray, writable
   for workers), syscalls per array and state change, page granularity and
   TLB shootdowns on all worker cores, which defeats small and medium
   arrays.
3. **Copy-on-write would break.** An ndarray has a fixed data pointer that
   C code may keep; viproc switches a storage to a new buffer when an
   in-place write meets pending readers (D8), and consolidates layouts.
4. **Unsafe by default.** A subclass inherits hundreds of C methods that
   work on the memory directly (`.sum`, `.copy`, `.tobytes`, `.fill`,
   indexing, pickling); each would need an override, and every one missed
   returns wrong values silently. With a wrapper, anything not implemented
   raises or goes through `__array__`, a clean sync point.
5. **Views created in C.** NumPy creates subclass views itself (slicing,
   `reshape`, `.T`) without telling viproc, which would lose the mapping
   onto shared storages.

**Consequences.**

- Library code that requires a real `numpy.ndarray` calls `__array__`
  (directly or through `np.asarray`): one sync point, and with D7 no copy.
- `isinstance(x, np.ndarray)` is false outside code that runs with
  `numpy` mapped to `viproc` (`python -m viproc`, planned), where
  `np.ndarray` is `viproc.ndarray`.
- Pure-Python NumPy code in libraries still runs asynchronously on viproc
  arrays passed to it: their `np.sin(x)` and `x * 2` dispatch through the
  protocols.

## D10 — Access model: allocate on the main thread, release anywhere

**Superseded by D11 for element memory** (the object pools of D4 keep this
model). Kept for the record: buffers took their block
when they are created, and only the main thread creates buffers (at
issue); tasks reach `allocated` at issue. viproc enforces one access model
for all its pools (element
memory, D3; runtime objects, D4):

- **Allocations from a pool happen only on the main thread** — the thread
  that issues ops.
- **Releases back into a pool may happen on any thread** — whichever thread
  drops the last reference (a worker finishing the last reader of a
  buffer, the main thread dropping a Python object).

**Pool structure that follows from it.**

- The pool's free index (ordered by capacity, D3) belongs to the main
  thread: lookups, inserts, eviction and the pool bound need no lock and
  no atomics.
- A release on the main thread inserts into the index directly.
- A release on another thread pushes the block onto a lock-free
  multi-producer stack (one CAS). The main thread takes the whole stack
  over with one exchange (no ABA) and merges it into its index — before an
  allocation that misses the index, and at sync points. The same pattern as
  the object pools of D4 (`src/runtime/pool.hpp`).
- Released memory is therefore reusable at the next main-thread pool
  operation, not necessarily at the exact moment of release; *when* it is
  released (D2) does not change.

**Consequences for the execution model.**

- **Element memory is taken at issue time.** In phase 1 every output shape
  is known when the op is issued, so the main thread takes the output's
  block from the pool right there, and the task reaches the `allocated`
  stage at issue (it passes `size_completed` and `allocated` on the main
  thread). Kernels never allocate; `Task::execute()` finds its outputs
  allocated. This replaces the current lazy allocation by the producing
  task (`Buffer::ensure_allocated()` on the worker). The copy-on-write
  paths (consolidated new buffer, copy task) and scalar inputs are issued
  on the main thread already.
- **Peak memory.** A pending output holds its memory from issue on instead
  of from the moment its producer starts. The look-ahead limit (1000 active
  tasks) bounds this; a memory-based look-ahead limit (bytes reserved by
  pending outputs, known exactly to the main thread) becomes the natural
  complement and is cheap to keep, since only the main thread allocates.
- **Interaction with D5.** Reusing a temporary's buffer is decided at issue
  time on the main thread as well; it simply skips the pool.
- **Data-dependent output sizes** (`nonzero`, boolean indexing, `unique`;
  eager NumPy fallbacks in the MVP): when such ops become asynchronous, the
  producer learns its output size only while running and cannot take from
  the pool. It then allocates from the system allocator; the block enters
  the pool on release like any other (releases are allowed on any thread).
  The main thread stays the only thread that ever takes from the pool.
- **Devices (future).** Device selection precedes allocation in the task
  stages; with allocation on the main thread, device selection for an op
  must happen at issue time as well, or device memory gets pools of its
  own with the same access model.

**Why.**

- The allocating side is the hot path (one allocation per op); making it
  single-threaded removes all synchronization from it and keeps every
  pool decision (matching, eviction, the bound, D5) deterministic and in
  one place, in line with "local decisions on the main thread".
- Releases cannot be confined to one thread without delaying them: the
  last reader of a buffer is often a worker. A push onto a lock-free stack
  keeps them immediate and cheap.
- It matches the thread rules already in place: only the main thread
  creates tasks and arrays; workers only advance and complete them.

## D11 — Element memory as late as possible, taken by the producing task

**Implemented** (`Buffer::Buffer(nbytes, Buffer::deferred)`,
`Task::allocate_outputs` in `src/runtime/task.cpp`,
`src/runtime/element_pool.*`). Replaces D10 for element memory.

**Decision.** A buffer a task produces is created *deferred* at issue: the
`Buffer` object exists (consumers capture it, count their reads on it,
the storage refers to it), but its element memory does not. The task takes
the memory right before its kernel runs, on the thread that runs it:

1. **From a dying input** (kernels with `Kernel::in_place_ok()`: NumPy's
   element-wise loops, plain and on NumPy's iterator): an input operand
   whose buffer this task holds the only reference to (no array, export or
   other task; `use_count() == 1`, `async_reads == 1`: only this task's
   read) and whose layout equals the output's (dtype, shape, strides,
   offset). The output takes over the input's block (`take_block`); the
   input operand then reads the output's buffer, same data. This is
   NumPy's temporary elision (D5), decided by the worker at run time
   instead of by the main thread at issue. `x = np.abs(x)` in a loop runs
   in place, in one hot block.
2. **Else from the element pool**, on that thread: the thread's own
   last-released block of the bin if it has one (LIFO, hot in its cache;
   the input it just released is a candidate), else the shared stack.

Inline buffers (≤ 128 bytes) have their memory at once. Ops the main thread
runs directly (`Runtime::issue_direct`) allocate there, right before they
run. `size_completed` is still reached at issue (phase 1: shapes are known),
`allocated` now when the task runs. Copy-on-write buffers (consolidated new
buffer, storage copies) are deferred as well.

**Why.** Measured (4-vCPU VM, Release, one worker, µs per op, `x =
np.abs(x)` repeated; NumPy in brackets): with the main thread allocating at
issue, 100 x 100 9.6 (2.4), 316 x 316 111 (25): the main thread runs up to
1000 ops ahead, so up to 1000 output blocks were reserved at once (80 MB at
100 x 100, 800 MB at 316 x 316), each cold when its task finally wrote it,
while NumPy reuses one hot malloc block. The same chain with `out=x` (no
new memory) ran at 2.4 (1.6) and 17 (24). With D11: 2.3 (2.6) and 19.7
(27.8).

**Pool access model that follows.**

- Any thread may allocate and release. A thread that allocates registers
  its free lists (`ElementPool::Local`) and keeps up to `kKeep` = 2
  released blocks per bin itself (enough for "release the input, take the
  output"); further releases go onto the per-bin lock-free shared stack, so
  a thread that mostly releases (the main thread dropping arrays) does not
  hoard memory other threads then allocate anew. A thread whose own list is
  empty takes the whole shared stack over (exchange, no ABA).
- The list heads are atomic only so that `trim()` (any thread) can take
  every registered thread's lists with one exchange each; the owner pushes
  with a CAS and pops by exchange (uncontended).
- A thread hands its blocks to the shared stacks and unregisters when it
  exits (a `thread_local` guard; `Local` itself stays trivially
  destructible, as in D4).

**No memory needs an address at issue.** NumPy's reduction iterator used
to be built per op at issue (it needs the GIL) over the input's real
address, which pinned a pending input's memory on the main thread. It is
now built once per signature over a dummy address (`ReduceTemplate`, as
the element-wise iterator templates) and the task points its copy at the
buffers when it runs (`NpyIter_ResetBasePointers`); the result is a
regular deferred output in the layout NumPy allocated for the template.
`ensure_allocated` and `take_block` still agree on one block by CAS on
the data pointer (the direct path allocates on the main thread right
before it runs). The alignment check for async `matmul` treats a deferred
buffer as aligned (pool blocks are at least 64-byte aligned).

**Consequences.**

- Peak memory falls: pending outputs hold no memory, only running tasks'
  outputs and live data do. The 1000-task look-ahead limit no longer
  multiplies output sizes; a memory-based look-ahead limit becomes much
  less pressing.
- D5 (main-thread elision of temporaries at issue) stays: it turns the op
  into an in/out op before a task exists, and covers ops run directly.
- Releases remain deterministic (D2): only where a block goes changes.
- Data-dependent output sizes (`nonzero`, boolean indexing, `unique`) fit
  without a special case: the task learns its size and takes its memory
  when it runs, from the same pool.
- Devices (future): the stages `size_completed → device_selected →
  allocated` can now all happen in the task, so the device can be chosen
  when the task runs, from where its inputs are.
- Versus D10: D10 kept the allocating side single-threaded (no atomics on
  the hot path). D11 pays one uncontended CAS per release and an exchange
  per allocation, and gains cache-hot memory, lower peak memory and in-place
  chains; the measurements above decide it.

## Open questions

- Pool bound default (fraction of RAM vs. fixed) and the eviction order.
- `FixedPool` (D4) still lets any thread allocate; tighten it to the D10
  model (main-thread allocation, assertion in debug builds).
- Bin granularity (8 vs. 4 per power of two) and the size range served by
  the pool at all (very small blocks could stay with malloc).
- Per-thread caches in front of the element pool: only if contention shows
  up in `strong_scaling.py` with large arrays.
- Huge-page threshold. Windows large pages are not swappable, need a
  privilege and are expensive to allocate (opt-in, see "Large blocks");
  measure whether they pay off with a warm pool before recommending them,
  or allocate them off the main thread.
- Memory-based look-ahead limit: pending outputs hold no memory since D11;
  check whether in-flight inputs (live data the main thread has dropped
  but tasks still read) still call for one.
- D5's interpreter check: how to tell "called from the interpreter" cheaply
  on CPython 3.12+ (NumPy inspects the C call chain; a cheaper criterion
  may exist for the number-protocol slots).
