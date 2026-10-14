# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**viproc** is a layer that adds **instruction-level parallelism for array
instructions (array ILP)** to regular NumPy programs. The program is
written sequentially (`c = a + b`, `m = max(c)`, `s = sum(c, axis=0)`, …);
viproc executes its array instructions in parallel wherever the data
dependencies allow it. The final computation of every instruction is
delegated to NumPy's own inner loops, including their SIMD paths.

The execution model follows the **ILNumerics Accelerator**: the caller's
thread does not compute. It only *issues* array instructions and returns
immediately. Each instruction runs as soon as its inputs are ready.
Independent instructions run concurrently; dependent ones keep only the
local order needed for correct results. Results must always be identical
to plain sequential NumPy execution.

### How parallelism is found

- **At runtime, by local decisions only.** There is no dependency graph
  (DAG) of the program and no ahead-of-time (AOT) analysis. Every decision
  is local: is this argument array `ready` or `pending`? Do not introduce
  global analysis structures.
- **The main thread runs ahead.** The Python main thread never waits for
  results (except at sync points). It races (far) ahead of the
  computation, and that implicit look-ahead is where the parallelism comes
  from: by the time early results complete, many later ops are already
  issued and waiting on exactly their inputs.

## Execution model

### Arrays

- An array is either **`ready`** (its data is final) or **`pending`** (a
  task will still compute it).
- A pending array references **the task that computes it** (its producer).
  For an in/out array this is the **last** task that writes it.
- Arrays do **not** reference the tasks that consume them as inputs.
- Element memory comes from viproc's element pool and is taken by the main
  thread when it issues the op that creates the array (phase 1: shapes are
  known at issue), so tasks reach `allocated` at issue. Pool allocations
  happen on the main thread only, releases on any thread
  (`docs/design/memory-management.md`, D3 and D10).
- **Views share a storage.** A base array and all its views refer to one
  shared *storage*, which holds the current buffer (data version) and the
  last task writing it. Readiness, the producer and `async_reads` belong
  to the storage: a write through one view makes the base and all other
  views `pending`, and copy-on-write switches all of them together.
  (Conservative: two views of disjoint regions are still ordered.)
- Each storage buffer has a dedicated atomic counter **`async_reads`**: the
  number of issued, not yet completed tasks that read it as an input.
  - The main thread increments it when it issues a task reading the array.
  - The reading task decrements it when it has finished reading (at the
    latest on `completed`), on whatever thread it runs.
  - An op's reads of the buffer it writes in place are not counted: every
    later writer waits for it as the storage's producer anyway.
  - Only the main thread increments, so **`async_reads == 0` seen by the
    main thread stays 0** until the main thread itself issues a new reader
    (same reasoning as `ready`, see "Thread rules").

### Tasks

- Exactly **one task per array op** (per op invocation). It represents and
  communicates the op's progress.
- Task states, in this order:
  `created → size_completed → (device_selected, future) → allocated → completed`
  - `size_completed`: output shape (and dtype) is known.
  - `allocated`: output element memory exists.
  - `completed`: output data is computed; the outputs become `ready`.
- A task passes through each state **at most once and only forward**. It
  may skip states, never go back.
- The task holds references to its input arrays, its output arrays and the
  resolved NumPy loop.
- Tasks, buffers and array blocks come from per-thread memory pools
  (`src/runtime/pool.hpp`), not malloc/free per op.

### Issuing an op (main thread)

1. **Create the task** (state `created`).
2. **Create the output array(s)** as `pending`, referencing the new task.
   For an **in/out argument** the array may already be `pending`; it then
   gets the new task as its producer (the *last* task that computes its
   data). A `ready` in/out array is switched to `pending` here, by the main
   thread.
3. **Register a callback for every `pending` input or in/out array** on
   that array's producer task. The callback informs the new task about the
   producer's stage events (partial completion such as `size_completed`,
   and full completion).
4. **If all inputs are `ready`,** the main thread triggers the task itself
   (see "Dispatch decision").
5. Continue with the next op.

**Look-ahead limit (back-pressure):** at most **1000 active tasks** (issued,
not yet `completed`). When the limit is reached, the main thread blocks
before creating the next task until a quarter of the active tasks
completed (hysteresis; completions wake it only then: resuming after each
completion cost a wake-up and a context switch per op). Fixed constant for
now; keep it in one place so it can become configurable or memory-based
later (1000 in-flight results of 77 KB each are 77 MB, far more than the
L2 caches the same ops would reuse with a shallow look-ahead).

When a producer reaches `completed`, its callbacks run on the completing
thread. The callback that reports the last pending input of a task
triggers that task, and **the thread that completed the last input runs
it** (no hand-off to the pool).

It must run as a **continuation, not nested**: the callback only records
the consumer as the thread's next task; the thread then unwinds its stack
back to its top-level task loop and runs the consumer from there
(trampoline). Never call a consumer task from inside the producer's
callback frame. Nested execution would build deep stacks along dependency
chains and, on errors, hide where the failing op came from.

If one completion unblocks several consumers, the completing thread runs
one of them as its continuation; the others go to idle workers / the
pool's queue.

Each task records its **issue site** (Python file, line and function,
captured by the main thread at issue time). Errors from a task are
reported with that issue site, not with the worker's stack.

### Eventing

- Tasks communicate only through these **multi-stage completion events**.
  Stage events can be used early: once all inputs are `size_completed`, a
  task can compute its own output shape (and go to `size_completed`) and
  allocate (`allocated`) before any input data is computed.
- The whole event system must be **asynchronous and robust**: a callback
  can be registered while the producer advances concurrently on a worker.
  Registration is atomic with respect to the producer's state: if the
  producer already passed the stage, the registering thread learns that
  immediately and handles it itself; no event is lost or delivered twice.
- **Sync points** (reading data, `print`, `float(x)`, handing memory back to
  plain NumPy) wait for `completed` of the array's producer. Querying only
  metadata (`x.shape`, `x.dtype`) waits for `size_completed` only. Because
  shape is a stage of its own, ops with data-dependent result shapes
  (`nonzero`, boolean indexing, `unique`) fit the model too; in the MVP
  they still fall back to eager NumPy.

### Thread rules

- **Creating** tasks and arrays: **main thread only.**
- **Advancing / completing** tasks and arrays: worker threads or the main
  thread.
- Only the main thread switches an array `ready → pending` (when it issues
  an op writing it). Only the array's current producer switches it
  `pending → ready`, on `completed`. A task that is no longer an array's
  last producer (a later in/out op took over) does not make it `ready`.
- Consequence: **once the main thread sees an array `ready`, it stays
  `ready`** until the main thread itself changes it. Checks of `ready`
  on the main thread need no lock; only the `pending` path (callback
  registration on a producer) needs synchronization. The same holds for
  in/out arrays.

### Reference counting

Memory decisions (no temporary array types, deterministic release, element
memory pool, temporary reuse, NumPy interop) are collected in
`docs/design/memory-management.md`; keep it up to date with any change to
ownership, release or reuse of memory.

- Tasks, arrays and array memory (buffers) are **reference counted**.
  Tasks and buffers are shared across threads (`std::shared_ptr`, atomic
  counts); arrays and storages are main-thread only (`Ref<T>`, plain int
  counts, see "Implementation notes").
- References: Python object → array; array → producer task (while
  `pending`); task → its input and output arrays; producer task →
  registered callbacks → consumer tasks.
- Cycles (array → producer → callback → consumer → array) are broken on
  `completed`: the producer releases its callbacks and its inputs, and a
  `ready` array drops its producer reference.
- **Write-after-read safety via `async_reads` (copy-on-write).** Only
  relevant for in/out argument semantics (an op that writes into an
  existing array: in-place operators like `a += b`, `out=`). When the main
  thread issues such a task, it checks the in/out array's storage:
  - No pending readers (`async_reads == 0`) and no input of this op reads
    the storage through a *different* view: write in place.
  - Otherwise the old data is still needed (by pending readers, or by this
    op's own overlapping inputs, e.g. `a += a[::-1]`, where NumPy requires
    all inputs to be read before the output is written). The storage
    switches to a new buffer; the op's inputs read the old one. The old
    buffer stays alive through its readers' references.
    - **Consolidated** (array is the only view of its storage and covers
      all of it): no copy at all. The op reads the old buffer and writes
      the new one, with the **same layout** (as in NumPy, the array keeps
      its strides).
    - **With views or partial writes:** a copy task copies the whole old
      buffer into the new one first; the op then writes its part. Views
      keep their layouts (offsets and strides stay valid in the new
      buffer).
- **Consolidation is decided at issue time, on the main thread.** Layout
  and size, once communicated (`size_completed`), never change. Tasks
  already issued captured (buffer, layout) pairs and are unaffected; only
  ops issued later see the new buffer.
- **Layouts follow NumPy's**, because the order in which a reduction visits
  elements depends on them (pairwise `add`): a new ufunc result gets the
  layout NumPy would allocate (`Layout::keep_order`: order='K', dense, axes
  ordered by the inputs' strides as NumPy's iterator orders them; F-order
  inputs give an F-order result), and copy-on-write keeps the array's
  layout. Importing a non-contiguous NumPy array (`asarray`) copies it
  densely, in its axis order.
- Memory is handed to plain NumPy for *writing* only inside an eager
  fallback, after the buffer was made exclusive (no pending writer, no
  task reader, no live export); exports to the user are read-only and
  count as readers (`Buffer::host_views`, design doc D7).

### Goals and non-goals

- **Goal:** strong scaling of whole array programs: a fixed program gets
  faster with more cores, because independent array instructions overlap.
- **Non-goal:** weak scaling, and data parallelism *inside* a single array
  op (splitting one op across threads). One instruction runs on one
  thread, as one NumPy loop call (or the same sequence of loop calls
  NumPy itself would make).
- **Not a goal yet:** any optimization besides array ILP: own SIMD code,
  kernel fusion, JIT code generation, GPU offloading. SIMD comes from
  NumPy's loops. Do not add these, but do not make design decisions that
  rule them out (keep device and memory handling behind an interface).

### Phase 1 (current): MVP on CPUs

Goal: autonomous, asynchronous execution of array instructions on
multicore CPUs.

- **Compute kernels: official NumPy C code, taken from the installed
  NumPy at runtime.** We do not vendor or compile NumPy. The runtime looks
  up NumPy's ufunc inner loops (with their SIMD dispatch) through the NumPy
  C API and calls them on viproc buffers. No kernels of our own.
- **Async execution / glue layer (ours):** sits between the user ops
  (`add`, `sum`, `max`, …) and the NumPy ndarray layer. It owns the
  arrays and tasks, the completion events between tasks, the worker
  threads, and synchronization when the caller needs a concrete value.
- **Python integration: design adopted from Bohrium's `npbackend`**
  (`bridge/npbackend` in https://github.com/bh107/bohrium): array operations
  are intercepted and forwarded to the runtime instead of being executed
  eagerly; data is synced back to NumPy on access. Implemented with NumPy's
  array protocols instead of an `ndarray` subclass (see "Python bridge").

Also out of scope for phase 1: .NET/ILNumerics bindings and distributed
execution.

## Tech stack

- **C++20** for the runtime, **C11** for the public ABI and for code that
  interfaces with NumPy's C sources. Same languages as NumPy (C) and
  Bohrium (C/C++).
- **CMake ≥ 3.24** with Ninja.
- **CPython ≥ 3.12 and NumPy ≥ 2.0** are runtime and build dependencies
  (NumPy headers via `numpy.get_include()`). Phase 1 always runs inside a
  Python process.
- Tests: CTest. Plain C/C++ test executables for now; pytest once the
  Python bridge exists.
- Formatting: `.clang-format` (LLVM style, 4 spaces, 100 columns).

## Repository layout

```
docs/design/           Design documents (decisions with rationale)
  memory-management.md ownership, release, pools, NumPy interop, COW
docs/benchmarks.md     benchmark results: current breakdown per category and
                       case, progress per change, external machines
include/viproc/        Public C ABI (viproc.h). The only interface bridges may use.
src/runtime/           Runtime core, no Python dependency (library `viproc`)
  layout.*             DType, Layout (shape, byte strides, offset), broadcasting
  small_vector.hpp     inline-storage vectors (shapes/strides, task operands)
  ref.hpp              Ref<T>: non-atomic intrusive refcount (Array, Storage)
  pool.hpp             FixedPool / make_pooled: per-thread pools (Task, Buffer, …)
  offload.*            OffloadPolicy: inline vs. worker for ready tasks
  element_pool.*       ElementPool: process-wide pool for element memory
                       (quantized bins, LIFO, bound, huge pages)
  buffer.*             Buffer: element memory (inline ≤ 128 B, else pool); async_reads
  task.*               Task: stages, consumer callbacks, execute(); Kernel interface
  runtime.*            Array (main-thread view), Runtime: issue protocol,
                       worker pool, trampoline, look-ahead limit, wait
  assign.*             element copy kernel (dst[...] = src)
  index.*              gather / scatter / masked assignment / select (indexing, np.where)
src/kernels/           Kernel adapters using NumPy (library `viproc_numpy`)
  numpy_ufunc.*        UfuncLoop / LoopCache (resolved NumPy loops),
                       element-wise kernel, wrap_ndarray / to_ndarray
  numpy_reduce.*       reduction kernel: NumPy's reduction loop, called in
                       NumPy's blocks (ReducePlan)
src/capi/capi.cpp      Implementation of the C ABI (library `viproc_capi`)
src/viproc_python.h    the only way to include Python.h (MSVC debug fix)
python/viproc/         Python package
  _viproc.c            CPython extension (C only, uses viproc.h): base class
                       _Array with the per-op hot path (__array_ufunc__ for
                       element-wise ufuncs, all operators, dtype-resolution
                       cache, scalars, broadcasting, issue sites)
  _ndarray.py          viproc.ndarray(_Array): fallbacks, __array_function__,
                       indexing, sync points; registers hooks with setup()
  __init__.py          module API (asarray, zeros, …; other names from NumPy)
python/tests/          pytest tests of the bridge (bit-identical to NumPy)
python/benchmarks/     strong_scaling.py, sync_latency.py
  suite.py             benchmark suite: runner, machine detection, score
  suite_cases.py       its cases (plain NumPy programs, width P, variants)
  suite_report.py      its Markdown / HTML reports, multi-run comparison
pyproject.toml         pip build (scikit-build-core; CMake installs the package
                       only for SKBUILD), version from CMakeLists.txt
dev.cmd, dev.sh        one-click dev loop (tools/dev.ps1 on Windows): pull,
                       build into .venv, test
.github/workflows/ci.yml  pip build + pytest, Linux (GCC) and Windows (MSVC)
docker/                images that build, test and run viproc (README.md):
  entrypoint.sh        command line of the Linux and Wine images
  linux/               GCC, CPython (python:<ver>-bookworm)
  windows/             native Windows container: MSVC, entrypoint.ps1
  wine/                Windows build (llvm-mingw) run under Wine on Linux
tests/
  test_numpy_loops.cpp    spike: NumPy loops on a worker without the GIL
  test_runtime.cpp        runtime core with C++ test kernels
  test_numpy_runtime.cpp  end-to-end with NumPy loops, bit-identical to NumPy
```

The build assembles the importable package in `build/python/viproc`
(Python sources copied, extension module built there).

Rules:
- Bridges (Python, later .NET) only talk to the runtime through
  `include/viproc/viproc.h`. No C++ types cross the ABI.
- Include Python via `src/viproc_python.h`, never `<Python.h>` directly:
  it hides MSVC's `_DEBUG`, which would otherwise make Debug builds link
  `python3XX_d.lib` and use the debug ABI. Test executables that embed
  Python start it through `tests/embed_python.h`.
- Never edit vendored third-party code in place. Wrap it, and if a patch is
  unavoidable, keep it as a separate patch file and document why.

## Commands

```bash
./dev.sh  |  dev.cmd      # one click: git pull, pip-build Release into .venv, pytest
                          # (--no-pull/--no-test, -NoPull/-NoTest); the maintainer's
                          # way to try a merged change on Windows
pip install .             # scikit-build-core (pyproject.toml), VIPROC_BUILD_TESTS=OFF
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug   # configure
cmake --build build                                     # build
ctest --test-dir build --output-on-failure              # run tests
clang-format -i <files>                                 # format

# Python bridge
PYTHONPATH=build/python python3 -m pytest -q python/tests
PYTHONPATH=build/python python3 python/benchmarks/strong_scaling.py   # use a Release build
PYTHONPATH=build/python python3 python/benchmarks/suite.py [--quick]   # all cases, score, report
docker/benchmark.sh [--quick]     # (re)build images, suite on Linux + Windows (Wine),
                                  # reports copied to ./reports (old ones deleted first)
```

`ctest` also runs the pytest suite. Tests need `pytest`
(`pip install pytest`).

Windows / Visual Studio: Debug builds work with a regular (release)
Python installation; the extension module is written to
`build/python/viproc` for every configuration (`ctest -C Debug`).

Docker (`docker/README.md`): each image builds this checkout and runs all
test suites by default (`test`, ctest only: no benchmark); `python`,
`pytest`, `benchmark`, `info` make it a command line. The Wine image tests
the Windows build on a Linux host. Benchmark reports: use
`docker/benchmark.sh` / `docker\benchmark.ps1` (they build both images
from the checkout every time, cached; delete old report files; copy the
new ones out with `docker cp <id>:/work/reports/<x>/.`: without the `/.`
an existing target directory gets the copy one level deeper and the old
report stays, which looked like "unchanged results"). Every report header
names its build (`build <time> #<fingerprint>`, `suite.build_stamp`);
check it before comparing runs.

```bash
docker build -f docker/linux/Dockerfile -t viproc:linux .
docker build -f docker/wine/Dockerfile -t viproc:wine .
docker run --rm viproc:wine                         # all suites, Windows build
docker run --rm -v "$PWD:/src" viproc:linux         # test this checkout (incremental)
```

Linux and Windows must behave identically: same results, same features
(temporary elision, element pool, huge/large pages where the OS allows).
Platform-specific code stays behind small functions with one
implementation per platform (`capture_stack`/`module_base` in
`_viproc.c`, `system_allocate` in `element_pool.cpp`), and both builds run
the same tests.

Build with `-DVIPROC_WARNINGS_AS_ERRORS=OFF` only for local experiments;
the default build treats warnings as errors. Always build and run the tests
before committing.

## Architecture (phase 1)

```
Python user code  (viproc.asarray(...) / import viproc as np)
        │
Python bridge     viproc.ndarray, NumPy protocols        ← design from Bohrium npbackend
        │  C ABI (viproc.h)
Runtime           arrays (ready/pending) · tasks · stage events ·
                  worker threads · sync                   ← ours
        │
Kernel adapters   map an instruction onto a NumPy loop
        │  function pointers resolved once, called without the GIL
NumPy C kernels   ufunc inner loops, SIMD                ← installed NumPy
```

See "Execution model" for arrays, tasks, eventing, thread rules and
reference counting.

- **Dispatch decision (when the main thread triggers a task whose inputs
  are all `ready`):** hand it to a worker, or execute it immediately on the
  main thread. Tasks triggered by a callback run on the thread that
  completed their last input, as a continuation (see "Issuing an op").
  The decision is one swappable component, `OffloadPolicy`
  (`src/runtime/offload.*`); `RuntimeOptions::should_offload` overrides it.
  The case to catch: the main thread needs the result right after issuing
  (`if (x*y).any():`, `print(c)`), so there is nothing to overlap and the
  hand-off is pure latency. Hybrid policy, after the ILNumerics
  Accelerator's "scalar shortcut" and "adaptive fork suppression":
  1. **`sync_below` (N_min, default 2048 output elements):** smaller ready
     ops always run on the main thread. Deterministic, one comparison.
     Calibrated with `strong_scaling.py` (4 CPUs, sin/cos chains): inline
     wins up to ~1000 elements, async from ~3000. Such an op takes the
     **direct path** (`Runtime::issue_direct`): no task at all. Its kernel
     runs right in `issue()` and its outputs are created ready; in/outs
     qualify when nothing reads them and no input reads them through
     another view (else copy-on-write: the task path). A kernel error is
     stored on the outputs and raised at their sync point, as with a
     task. Before, such ops built a task and ran it inline: the threshold
     decided *where* an op ran, not whether it paid for the task protocol
     (task, operand capture, read counting, look-ahead slot, completion),
     ~1 µs per op even for scalars. `offload_stats()["direct"]` counts them.
  2. **Adaptive above N_min, per (issue site, log2 size class):** count
     async issues and how many of their results the main thread accessed
     "right away" — a sync point within `window` = 2 issued ops of the
     decision that started the work, **whether the task had finished or
     not** (counting only unfinished ones made fast sites oscillate between
     modes). Consumers issued while it was pending inherit the attribution.
     After `sample` = 8 issues: ≥ 3/4 accessed right away → the site runs
     inline; in inline mode an exploration call still goes async, first
     every 16th call, doubling up to every 1024th while explorations confirm
     inline; ≤ 1/4 switches back. Reversible, unlike the Accelerator.
  3. **Deterministic mode:** `adaptive = false` (`viproc.init(adaptive=False)`
     or `VIPROC_DETERMINISTIC=1`) keeps only rule 1. Results are identical
     in every mode; only where ops run changes.
  Not done yet: per-op cost classes (N_min is in elements, calibrated on
  sin/cos; cheap ops like `add` would warrant a higher N_min).
  **Decided against:** letting a waiting main thread run a queued, not yet
  started task itself. Ready work rarely sits in the queue: a task whose
  inputs complete runs at once as the continuation of the thread that
  completed the last input. Measured (`strong_scaling.py`, 1e4 and 1e6):
  1278 continuations vs. 336 queue pops (chain starts, fan-out), and no
  sync point found its task runnable but queued. In `sync_latency.py` that
  happened only for the async exploration calls, which the policy keeps
  rare.
  `python/benchmarks/sync_latency.py` measures the targeted pattern.
- **Errors** from asynchronous execution are stored on the task and its
  outputs and raised at the next sync point that touches them.

### Notes on the reference projects

- **Bohrium** (bh107/bohrium, last commit 2020): CPython bridge
  (`bridge/npbackend`), a C bridge API (`bridge/c`), core IR
  (`core/bh_ir.cpp`, `bh_instruction.cpp`, `bh_view.cpp`) and backends
  (`ve/openmp`, …). We adopt the **bridge**, not the JIT backends. It
  targets NumPy 1.x and Python ≤ 3.7, so porting to current NumPy 2.x and
  Python 3.12+ is expected work.
- **NumPy** (checked against 2.5.3 installed, 2.6.0.dev0 main): every
  op we need is a ufunc or gufunc, including `numpy.linalg`
  (`numpy.linalg._umath_linalg`: `det`, `inv`, `solve`, `svd`, `eig`, `qr`,
  `cholesky`, `lstsq`, …) and `numpy.fft` (`numpy.fft._pocketfft_umath`:
  `fft`, `ifft`, `rfft_n_even`, `rfft_n_odd`, `irfft`). All of them go
  through the same loop machinery.

### How NumPy handles element types

- **One inner loop per dtype signature.** Each ufunc has a list of loops,
  one per type signature (`np.add.types`: `??->?`, `bb->b`, …, `ii->i`,
  `ll->l`, `ff->f`, `dd->d`, `FF->F`, `DD->D`, …). `int32`, `int64`,
  `float32`, `float64` etc. each have their own C function. They are
  generated from templates (`*.c.src` / `*.dispatch.c.src` with
  `/**begin repeat` blocks, C++ templates and Highway for newer loops in
  `numpy/_core/src/umath/`), so the source exists once and the binary has
  one instance per type.
- **SIMD dispatch per CPU:** `*.dispatch.*` loops are compiled several
  times for different instruction sets (SSE/AVX2/AVX512/NEON/…); the best
  one is selected at import time.
- **Mixed inputs are not separate loops.** `add(int32, float64)` is
  resolved by type promotion to the `dd->d` loop; the `int32` operand is
  cast first. Some combinations are explicit loops (`logical_and`:
  `bb->?`, `fft`: `Dd->D`). `l` and `q` are both `int64` on Linux.
- Linalg gufuncs only have `f`, `d`, `F`, `D` loops (integers are cast to
  `float64`). FFT loops exist for `f`, `d`, `g` and their complex types.

### Calling NumPy kernels

Use NumPy's low-level loop access API (unstable, NumPy-version-specific
capsule name `numpy_1.24_ufunc_call_info`, but works for ufuncs,
reductions and gufuncs alike):

1. `call_info = ufunc._resolve_dtypes_and_context(dtypes, reduction=...)`
   resolves dtypes, applying NumPy's promotion rules.
2. `ufunc._get_strided_loop(call_info, fixed_strides=...)` fills the capsule:
   `strided_loop`, `context`, `auxdata`, `requires_pyapi`,
   `no_floatingpoint_errors`.
3. The loop has the new-style signature
   `int loop(PyArrayMethod_Context *ctx, char *const data[], npy_intp const dims[], npy_intp const strides[], NpyAuxData *aux)`
   and returns 0 on success, -1 on failure.
4. Argument layout: `dims = {outer_count, core dims…}`,
   `strides = {outer stride per operand…, core strides per operand…}`.
   E.g. `det` `(m,m)->()`: `dims = {N, m}`,
   `strides = {in_outer, out_outer, in_row, in_col}`; `fft` `(n),()->(m)`:
   `dims = {N, n, m}`, `strides = {in_outer, fct_outer, out_outer, in_n, out_m}`.

**Why not call the regular NumPy entry (`np.sum`, …) on a worker?** It is
Python C API (argument parsing, ndarray objects, dtype resolution,
`errstate`, warnings) and needs the GIL; NumPy releases the GIL only
inside the inner loop (from ~500 elements). The main thread holds the GIL
while it runs the program ahead, so a worker gets it only at the switch
interval (5 ms). Measured (4 worker threads calling `np.sum`): 1e6
elements: 62 ms serial, 16 ms with an idle main thread, but 148 ms with a
busy main thread; 1e3 elements: 34 ms serial vs. 6858 ms. Hence workers
call the loops directly, without the GIL, and viproc reproduces NumPy's
calling pattern where results depend on it: only floating-point `add`
reductions (pairwise summation; `sum`, `mean`, `var`, `std`). Element-wise
ops, `min`/`max`/`any`/`all`/`prod`, linalg and fft give the same result
however the work is split into loop calls. With free-threaded CPython
(3.13t+, supported by NumPy ≥ 2.1) calling the regular entry on workers
becomes viable as a generic path.

Mirror only the leading fields of the capsule struct (`strided_loop`,
`context`, `auxdata`, `requires_pyapi`, `no_floatingpoint_errors`). The
embedded `PyArrayMethod_Context` behind them changes size between NumPy
versions (extra fields when `NPY_FEATURE_VERSION > NPY_2_3_API_VERSION`).

Rules:
- **Resolve with the GIL, on the caller thread,** when an instruction is
  issued (cache per ufunc and dtype signature). The capsule owns `auxdata`;
  keep it alive until all instructions that use it have finished.
- **Call without the GIL** from worker threads. Never run a loop with
  `requires_pyapi` set (object dtype etc.) asynchronously; fall back to
  eager NumPy instead.
- **Casting:** when promotion casts an operand, the cast is its own
  instruction (cast loops via the same ArrayMethod machinery) or done
  eagerly before the op. Never call a loop on data of the wrong dtype.
- **Floating-point errors:** loops report divide-by-zero, overflow and
  invalid via the thread-local FP status flags (unless
  `no_floatingpoint_errors`). Workers clear the flags before and read them
  after each loop and store them on the instruction; the bridge applies
  `np.errstate` semantics at the next sync point.
- **Reductions** (`sum`, `prod`, `min`, `max`, `any`, `all`, …;
  `src/kernels/numpy_reduce.*`) use the binary loop resolved with
  `reduction=True` and `fixed_strides` = the inner strides, called with
  `args = {acc, in, acc}` (acc stride 0 along reduced axes). NumPy's float
  `add` loop sums each call pairwise, so the blocks decide the rounding;
  viproc calls the loop in exactly NumPy's blocks (`ReducePlan`, built at
  issue time):
  - axes ordered by the input's |stride| (nditer's insertion sort over C
    order; zero strides and extent-1 axes never decide), negative strides
    kept (DONT_NEGATE_STRIDES);
  - adjacent reduced axes coalesced when the input is contiguous across
    them; kept axes not (element-wise accumulation: blocks do not matter);
  - one loop call per outer position over the whole innermost axis;
  - accumulator starts at `ufunc.identity` (add: +0.0, so `sum([-0.0])` is
    0.0, as NumPy); without identity (min/max) at the first element along
    the reduced axes, which the loop then skips (NumPy's first-visit skip).
  `ReducePlan` covers dense inputs (any axis order) without a cast whose
  reduced axes coalesce into one. **Everything else runs on NumPy's own
  iterator** (`make_iter_reduction`, `IterReduceKernel`): casts (int32
  sums accumulate in int64, `dtype=`), `initial=`, non-coalescing reduced
  axes and strided inputs, which NumPy's iterator may copy into buffers
  even without a cast (and a loop on a contiguous buffer can round
  differently from the same loop on strided data). The iterator is built at
  issue time with the GIL exactly as `PyUFunc_ReduceWrapper` builds it (it
  also allocates the result in NumPy's layout); the kernel runs NumPy's
  `reduce_loop` over it without the GIL (`NpyIter_Reset(iter, &errmsg)`).
  Fuzzed against NumPy over shapes, axes, layouts and dtypes
  (`python/tests/test_reductions.py`, incl. signs of zeros and NaNs):
  bit-identical.
- **NumPy's reduction chunking depends on its version** (measured for a
  contiguous 1-d float64 `np.sum`, 100003 elements, NumPy 2.1–2.5):
  **up to 2.2** the loop is called in chunks of `np.getbufsize()` (default
  8192; users can change it, so read it at issue time); **from 2.3 on** in
  one call, independent of the buffer size. `test_numpy_loops` checks the
  rule for the installed version. Axis reductions and strided inputs are
  not measured yet; check them the same way before relying on them.
- **`argmin`/`argmax`** come from NumPy too: the per-dtype functions in
  `PyDataType_GetArrFuncs(descr)->argmax` / `->argmin`
  (`int f(void *data, npy_intp n, npy_intp *index, void *arr)`).
- **Linalg:** the Python wrappers (`np.linalg.inv`, …) do argument checks
  and raise `LinAlgError` via an `errstate(invalid='call')` callback when
  the gufunc sets the invalid flag. We replicate those checks in the
  bridge and map the stored FP flags to `LinAlgError` at sync. Linalg
  loops call BLAS/LAPACK, which may be multithreaded itself; limit BLAS
  threads (e.g. 1 per worker) to avoid oversubscription.
- **FFT:** the Python wrappers in `numpy/fft/_pocketfft.py` handle `n`
  (padding/truncation), `axis`, `norm` (passed as the scalar `fct`
  argument) and the choice between `rfft_n_even`/`rfft_n_odd`. We port that
  logic into the bridge and issue the gufunc as an instruction.
- **Dtype semantics follow NumPy exactly:** promotion, `divide` on integers
  yielding `float64`, overflow wrap-around, NaN handling.
- **Broadcasting and scalars:** NumPy broadcasting rules apply to all
  ops. Scalars are true 0-d arrays (`ndim == 0`, one element), not
  1-element 1-d arrays; Python scalars become 0-d arrays at issue time.
  A broadcast operand is passed to the loop with stride 0.
- The proven calling patterns are in `tests/test_numpy_loops.cpp`; keep
  that test green when changing how loops are called.

## Python bridge

- **Decision: a wrapper class using NumPy's array protocols, not an
  `ndarray` subclass.** `viproc.ndarray` implements `__array_ufunc__`
  (NEP 13) and `__array_function__` (NEP 18) plus the operators. Bohrium's npbackend subclasses `ndarray` and
  protects memory (mprotect + signal handler) to sync when NumPy C code
  touches the data; with the protocols every NumPy entry point either is
  intercepted or calls `__array__`, which is a clean sync point. No
  Bohrium code is copied (license still open); only the design idea
  (intercept, run elsewhere, fall back, sync on access) is reused.
- **Supported asynchronously:** element-wise ufunc calls (`ufunc.signature
  is None`, one output) with NumPy's own dtype resolution
  (`ufunc.resolve_dtypes`, NEP 50 weak Python scalars), incl. casts of
  inputs, `out=` and in-place operators; `astype` (NumPy's cast functions,
  NumPy's result layout); basic indexing as views (no sync); metadata-only
  NumPy functions as views without a sync (`.T`, `transpose`, `swapaxes`,
  `moveaxis`, `squeeze`, `expand_dims`, `flip`, `rot90`, `broadcast_to`,
  `diagonal`, `real`/`imag`, `view(dtype)`, `reshape`/`ravel` where NumPy
  makes a view, else an async copy first, `atleast_*d`, `fliplr`/`flipud`,
  `matrix_transpose`). **No ndarray is built for a view:** NumPy has no
  public function that computes view metadata (`get_view_from_index`,
  `_attempt_nocopy_reshape` are internal), so viproc computes the layout
  itself, after NumPy's rules (as ILNumerics does): basic keys in C
  (`_viproc.basic_view`, slices through CPython's `PySlice_Unpack` /
  `PySlice_AdjustIndices`, as NumPy), `_viproc.transpose`,
  `vp_array_reshape` (`reshape_view` in `layout.cpp`, a port of
  `_attempt_nocopy_reshape`; NULL where NumPy copies), `broadcast_to`,
  `PyArray_Diagonal` in Python. Invalid arguments raise NumPy's own error
  from the same call on a zero-stride probe (`np.broadcast_to` of one
  element, no element memory); `python/tests/test_views.py` fuzzes
  layouts against NumPy. `a[key]` is the C slot `_Array.mp_subscript`:
  ready integer keys (NumPy's scalar) and basic keys (a view) without a
  Python frame, the rest through the hook `_getitem` (`setup()`).
  `a[1:3]` ~1,580 instructions / 0.12 µs (NumPy ~1,520 / 0.10; it was
  ~4,700 with `__getitem__` in Python), `a.T` ~1,410 / 0.11 µs;
  **advanced indexing** as gather / scatter
  tasks (`src/runtime/index.*`, element copies, so the element order does
  not matter): integer arrays (viproc arrays may be pending), lists, ints
  next to arrays, boolean masks (they sync: `nonzero()` gives the shape),
  `x[mask] = scalar` without a sync (masked assignment); NumPy's
  placement of the broadcast index dimensions (adjacent in the key or
  first) and NumPy's result layout (the order of the result's axes),
  taken once per key pattern from NumPy indexing a tiny proxy (extents
  1 or 2, same axis order) and cached, never per op; out-of-bounds
  indices raise `AsyncIndexError` (an `IndexError`) at the sync point;
  scalar indexing returns NumPy's scalar if the array is ready, else a 0-d
  copy (no sync); `np.where(cond, x, y)` as a select task (NumPy's result
  dtype, x and y cast to it, layout order K);
  `__setitem__` with basic keys (assign kernel); **reductions** (NumPy ≥
  2.3): `ufunc.reduce` of reorderable ufuncs (add, multiply, minimum,
  maximum, fmin, fmax, logical and/or/xor, bitwise and/or/xor) and
  `np.sum/prod/min/max/amin/amax/any/all` (also as methods) without
  `out=` and `where=`, with casts and `initial=`; `any`/`all` of non-bool data as `a != 0` first;
  `mean`, `var`, `std` of every viproc dtype composed exactly as NumPy's
  `_methods.py` (sum, in float64 for bool and integers; divide by the
  count in the promoted dtype and cast back; squared deviations, for
  complex data through NumPy's float view of the deviations);
  `argmin`/`argmax` (NumPy's per-dtype functions, one call per row of a
  C-contiguous copy with the axis last, as `PyArray_ArgMax`). Per-call
  preparation is cached per (ufunc, dtype, ndim, axis, keepdims)
  (`_reduce_plans`).
- **0-d results and the scalar shortcut.** Where NumPy returns a scalar
  (full reductions, `a[i, j]`, ufuncs on 0-d operands), viproc returns
  NumPy's scalar if the data is there already (the op ran directly), else
  a pending 0-d array marked `_scalar`: in-place operators on it rebind
  (as on NumPy's immutable scalars; a plain 0-d array keeps NumPy's
  in-place semantics). An operator or ufunc call whose operands are all
  scalars (ready 0-d viproc arrays, Python and NumPy scalars) has nothing
  to run asynchronously: the C hot path converts them to NumPy scalars
  and lets NumPy compute it right there (`scalars_of`; operators as NumPy
  scalar math, ufunc calls as NumPy's ufunc), returning NumPy's scalar,
  as the NumPy program would. Any n-d operand, even of one element, keeps
  the array path and NumPy's output shape. `float()`, `bool()`,
  `format()`, `round()` of a pending 0-d array sync.
- **Fallback (sync + eager NumPy on viproc memory, `_Host`):** everything
  else (`nonzero`, `flatnonzero`, `argwhere`, `count_nonzero` sync on their
  argument and run in NumPy on a read-only export, without the general
  fallback's wait for readers): other reductions (`out=`, `where=`), gufuncs
  (`matmul`), empty advanced-index results, 0-d boolean indices,
  `ufunc.at`, other NumPy functions. NumPy's semantics exactly (design doc
  D7, "Eager fallbacks"): each viproc argument becomes a writable NumPy
  view after `vp_array_prepare_write` (wait for producer and task readers,
  copy-on-write if read-only exports are alive), so `out=`, mutating
  functions (`np.put`, `np.place`, …) and overlap work in place. Results
  are mapped back: an argument returned as is → that viproc array; a view
  of an argument (by memory overlap) → a viproc view of its storage
  (`vp_array_view`, dtype views, read-only if NumPy made it read-only); a
  NumPy argument → unchanged; fresh memory → adopted; else copied.
  `np.copyto(dst, src)` maps to assignment.
- **Read-only viproc arrays** (`_readonly`, e.g. from `broadcast_to`):
  writes raise NumPy's errors; never reused by temporary elision.
- **Sync points:** `np.asarray(x)`, `x.numpy()`, printing, `float(x)`,
  `bool(x)`, scalar indexing, iteration, `x.wait()`, `viproc.wait_all()`.
- **Conversions (design doc D7):** exports (`np.asarray(x)`, `x.numpy()`,
  `__array__`) are copy-free **read-only** NumPy views; each counts as a
  reader of the buffer (`host_views`) while it lives, so later in-place
  writes to `x` copy-on-write and the export keeps its values (copy
  semantics without a copy). Imports copy user-held NumPy arrays
  (`asarray(nd)`, `v + nd`: the user may write `nd` later); fresh memory
  (creation functions, fallback results that own their data and are no
  argument) is adopted without a copy. Never give user code a writable
  view of viproc memory: NumPy calls on an ndarray bypass viproc and race
  with the workers (fallback exports are made read-only after the call).
- **Hot path in C** (`_viproc._Array`): everything that runs once per op —
  `__array_ufunc__` for element-wise calls, `a[key]` for ready integer and
  basic keys, all Python operators (number
  protocol and rich comparisons, with NDArrayOperatorsMixin semantics,
  incl. deferring to `__array_ufunc__ = None`), dtype resolution via
  NumPy's `ufunc.resolve_dtypes` cached per (ufunc, input kinds; Python
  scalars as weak NEP 50 kinds), scalar conversion (`vp_array_from_scalar`,
  NumPy's conversion rules), broadcasting and the issue site. Whatever it
  does not handle goes to the Python fallback, so NumPy semantics (errors
  included) stay NumPy's.
- The issue site is the first stack frame outside viproc and NumPy. It is
  cached per (code object address, bytecode offset), formatted only on a
  miss and interned once with `vp_intern`; tasks keep that pointer. Never
  key dicts by code object: hashing one hashes its whole bytecode
  (~3000 instructions per lookup; it was 20% of the per-op cost).
- Loops are resolved once per (ufunc, input kinds) into a `vp_loop`
  handle (`vp_ufunc_loop`); ops are issued with `vp_ufunc_issue` without
  any lookup. Per-op caches (issue sites, code flags, signatures) are small
  open-addressing C tables keyed by (pointer, int), not dicts: building key
  tuples/ints and two dict lookups cost ~1,700 instructions per op, the
  tables ~100. Each entry holds a reference to its key object. Sites and
  signatures are cleared when the runtime shuts down (handles and interned
  sites belong to it).
- **Temporary elision** (design doc D5): an operator whose operand is a
  temporary (dtype/shape of the output, buffer reusable per
  `vp_array_reusable_bytes`, object only on the interpreter stack) writes
  into that operand. Temporaries are detected exactly on CPython ≥ 3.14;
  on 3.12/3.13 (glibc, Windows) by refcount 1 plus the C stack, which
  must start with a calibrated chain reaching the interpreter's call
  site (≥ 256 KiB only); off elsewhere. The chain is the common prefix of
  two calibration runs, not a fixed frame count: how many frames
  CPython's operator entry has depends on its build.
- Evaluated and rejected: caching/reusing `viproc.ndarray` objects. Their
  allocation and deallocation is ~1% of the per-op instructions.
- One Python thread only: the thread that initialized the runtime.
- Per-op cost (`x * y`, 8 elements, Release, callgrind, op run inline,
  `OPENBLAS_NUM_THREADS=1` so idle BLAS threads do not blur the counts):
  viproc ~5,400 instructions, ~0.65 µs; NumPy ~4,650, ~0.4 µs. History:
  ~10,500 → ~7,100 (C lookup tables instead of dicts, flat loop calls)
  → ~6,600 (array and storage in one block, intrusive non-atomic refs,
  `vp_array*` is the `Array*`, small element memory inline in the Buffer)
  → ~5,800 (operands built in place in the task, no lock to read the
  upstream error, inline `itemsize`) → ~5,400 (pooled Task/Buffer/block
  memory). The bridge front end (~900) is cheaper than NumPy's own dispatch
  and dtype resolution (~2,500); what remains on top is the task protocol
  itself (task object, operand capture, async_reads, slot accounting,
  completion under the task mutex) and the result's release.
- `std::atomic::notify_all` is not free without waiters: libstdc++ bumps a
  process-wide counter (a shared cache line) for atomics that are not
  4-byte futex words. Task completion and `run()` therefore notify only
  when a waiter registered (`Task::waiters_`, `active_waiters_`).
- **Element-wise loops are called exactly as NumPy calls them.** Their
  results depend on the strides of the call: NumPy's SIMD paths (complex
  multiply, float32 exp/sin/arctan2, ...) round differently from their
  scalar fallbacks, even for one element. Where NumPy makes one call
  (`numpy::single_call`: its trivial loop, or an iterator that coalesces
  everything) the shared kernel makes the same call with NumPy's strides;
  everything else (strided, reversed, broadcast, mixed orders, casts) runs
  on NumPy's iterator, set up as `execute_ufunc_loop` sets it up
  (`LoopCache::iter_elementwise`: a template per (loop, layouts, buffer
  size) over a dummy address; per op a finished iterator of that template
  from its pool, else an `NpyIter_Copy`; `NpyIter_ResetBasePointers` on
  the worker; a fresh copy per op allocated its buffers on the worker and
  released them later on the main thread, which kept that memory cold). Casts happen in the
  iterator's buffers, or, for 0-d and short 1-d inputs, as a separate cast
  task first (NumPy's `check_for_trivial_loop` copies those).
- **Measurements live in `docs/benchmarks.md`** (suite scores per change,
  per-category and per-case breakdown, micro benchmarks, external
  machines). Headline (4 CPUs, Release): suite score 43 → 69 over this
  phase, all results bit-identical; `strong_scaling.py` 8 chains vs. NumPy
  with 4 workers: 1e7 ~7.5x, 1e6 ~4.5x, 1e4 ~3x, 1e3 ~0.9x. Small ready
  ops: `x * y` (8 elements) 0.63 µs (NumPy 0.35), ops on scalars as NumPy
  (0.06 µs), `a[1:3]` 0.12 µs (NumPy 0.10), `a.T` 0.11 µs (NumPy 0.08),
  `a[i, j]` 0.20 µs (NumPy 0.07), `x[idx]` (1,000 elements) 2.9 µs (NumPy
  0.7), `a.reshape(-1)` 0.6 µs (NumPy 0.15; Python handler), small
  `np.sum` ~2.8 µs (NumPy 1.5–2; NumPy's own `__array_function__`
  dispatch is most of it).
- "Issue an op, read the result right away" (`sync_latency.py`): 8 elements
  8.1 → 1.4–2.4 µs (NumPy 0.5–0.9), 1e3 5.1 → 1.6–2.7 µs (NumPy 0.8–1.6),
  1e4 ~NumPy (4.3–6.9 vs. 6.1–6.9 µs), 1e5 faster than NumPy, thanks to
  the dispatch policy. What remains at small sizes (~1 µs per op) is the
  issue path itself (task, arrays, bridge), paid even when the op runs
  inline; the hand-off to a worker is no longer part of it.

## Implementation notes (current code)

- `Array` and `Storage` objects are **main-thread only**. Workers see only
  `Task`s and the `Buffer`s captured in their operands. State is derived:
  a storage is `pending` while its producer is set and not completed.
- `Runtime::view(base, layout)` creates a view (metadata only, no task).
  The bridge must map NumPy views of one base onto runtime views of one
  storage; wrapping two NumPy views separately would lose the aliasing.
- After copy-on-write the storage no longer uses the wrapped NumPy memory;
  the bridge must treat the runtime array, not the original ndarray
  memory, as the source of truth.
- Operands capture the **buffer at issue time**, not the array, so a reader
  keeps the data version it was issued against when an in/out op switches
  the array to a new buffer (copy-on-write).
- Tasks the **main thread** runs inline (`should_offload == false`) hand the
  consumers they unblock to the pool, so the main thread keeps issuing.
  Workers run one unblocked consumer as their continuation.
- Worker pool: at most one idle worker spins (50 µs, `pause`) before
  sleeping; `enqueue()` wakes a sleeper only if nobody spins. This avoids a
  wake-up syscall per small task without taking cores from busy threads.
- Per-op allocations: one pooled Task (operands inline up to 4), one
  pooled Buffer (element memory inline up to 128 bytes, larger from the
  element pool: no page faults once warm), one pooled
  `ArrayBlock` (Array and its Storage), the Python object. Shapes/strides
  inline (`SmallVector`), one shared kernel object per resolved loop,
  interned issue sites.
- `Array`/`Storage` use `Ref<T>` (`src/runtime/ref.hpp`): plain int counts,
  legal because only the main thread creates, copies and releases them. An
  `ArrayBlock` holds a base array and its storage; each object is destroyed
  with its last reference, the block when both are gone (views get a block
  of their own). The C ABI hands out `Array*` as `vp_array*`, one reference
  each.
- Pools (`FixedPool`): per-thread free lists without atomics for the
  allocating thread; other threads return blocks through a lock-free stack
  that an allocating thread takes over whole (exchange, so no ABA). Pools
  only grow, to the peak number of live objects.
- Copy-on-write copy tasks are counted as active but do not wait on the
  look-ahead limit: the op needing the copy already holds its slot.
- **GIL:** the bridge must release the GIL while waiting on the runtime
  (`Runtime::wait`, `wait_all`); workers never take it.
- `UfuncLoop`s are owned by a `LoopCache` on the main thread; kernels hold
  raw pointers. Destroy the cache only after `wait_all()`. Kernels never own
  Python objects.
- `wrap_ndarray` borrows the NumPy memory; the bridge keeps the ndarray
  alive while tasks may read it.
- Python objects whose memory a task still uses are released only on the
  main thread with the GIL: a worker dropping the last reference parks
  the object in a graveyard drained at the next C ABI call.
- Not implemented yet: casting into `out=` of another dtype, gufuncs
  (linalg, fft, matmul), FP-flag reporting to
  Python (NumPy warnings / `np.errstate`), `python -m viproc`.

## Conventions

- Language for code, comments, docs and commit messages: **English**.
- C ABI symbols use the `vp_` prefix; C++ code lives in `namespace viproc`.
- C++: RAII, no raw `new`/`delete`, no exceptions across the C ABI (convert
  to error codes at the boundary).
- Thread safety is part of every runtime API. Document which thread may
  call it and which locks it takes.
- Every new instruction needs a test that compares the asynchronous result
  with NumPy executing the same ops eagerly.
- Keep the MVP small: prefer the simplest correct scheduling over
  optimizations until there is a benchmark that shows the need.
- Measure success as strong scaling: benchmark whole array programs
  (fixed problem size) against plain NumPy at 1, 2, 4, … worker threads.
- **Benchmark suite** (`python/benchmarks/suite.py`): the yardstick for
  changes that affect performance; run it before and after. Score =
  100 x geomean over categories of geomean(efficiency), efficiency =
  speedup vs NumPy / ideal, ideal = min(cores, P) for medium and large
  data and 1 (parity) for scalar, tiny and small data; P is the case's
  parallel width; `large` scales with RAM per core. New op paths (casting,
  reductions, int16/float16, advanced indexing) should show up as fewer
  fallbacks and higher category scores. Cases are plain NumPy code and
  must stay bit-identical; add a case for every new kind of program.
  Record every full run that a change is judged by in
  `docs/benchmarks.md` (progress table: commit, build stamp, machine,
  score, categories, size classes). Compare runs only on the same
  machine; `--quick` runs are noisier and not comparable to full runs.
  Single configurations vary by ±25-30% between full runs, the score by a
  few points: judge a change by the cases it targets (same-machine
  before/after, e.g. the previous commit in a git worktree).
  Note: data-dependent functions that sync by design (`nonzero`, ...)
  count as fallbacks (`kmeans_points`: one per cluster and round).

## Licensing

- NumPy: BSD-3-Clause.
- Bohrium: the root `LICENSE` is Apache-2.0, but source file headers
  (including `bridge/npbackend/src/*`) say LGPLv3+. Clarify which applies
  before copying Bohrium code into this repo, and keep its copyright
  headers intact.
- viproc itself has no license file yet.

## MVP scope (decided)

All ops below run asynchronously; anything else falls back to eager NumPy
in the bridge after a sync, the way Bohrium falls back for unsupported
functions.

- **Element-wise unary ufuncs:** all of them (`negative`, `absolute`,
  `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `floor`, `rint`, …).
- **Element-wise binary ufuncs:** all of them (`add`, `subtract`,
  `multiply`, `divide`, `power`, `floor_divide`, `remainder`, `maximum`,
  `minimum`, `arctan2`, `hypot`, bitwise ops, shifts, …), incl. `matmul`.
- **Boolean functions:** comparisons (`less`, `equal`, …), `logical_and`,
  `logical_or`, `logical_xor`, `logical_not`, `isnan`, `isinf`,
  `isfinite`, `signbit`.
- **Reductions:** `sum`, `prod`, `min`, `max`, `any`, `all` (ufunc
  `.reduce`); `mean`, `var`, `std` as compositions of these; `argmin`,
  `argmax` (not ufuncs, need their own kernel path).
- **Linear algebra:** the `numpy.linalg` functions backed by
  `_umath_linalg` gufuncs (`det`, `slogdet`, `inv`, `solve`, `cholesky`,
  `qr`, `svd`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `lstsq`), plus `matmul`/`dot`.
- **FFT:** the `numpy.fft` interface (`fft`, `ifft`, `rfft`, `irfft` and
  the `n`-dimensional variants built on them).
- **Kernels:** from the installed NumPy at runtime (see "Calling NumPy
  kernels").
- **Dtypes:** `bool`, `int32`, `int64`, `float32`, `float64`, `complex64`,
  `complex128`. Object, string, datetime, `float16` and `longdouble` fall
  back to eager NumPy.
- **Shapes:** full NumPy broadcasting; true 0-d scalars.

## Status and next steps

Where development stands (branch `claude/clever-johnson-nqs2hl`, PR
hokb/viproc#1). The maintainer writes in German; answer in German, keep
code, comments, docs and commit messages in English.

- Done in phase 1: runtime (tasks, stage events, trampoline, pools,
  copy-on-write, back-pressure with hysteresis), element memory pool and
  temporary elision, copy-free NumPy interop, eager fallbacks with NumPy
  semantics, Windows parity (Wine image), Docker images, benchmark suite;
  asynchronous element-wise ufuncs (exact NumPy iteration, casts),
  reductions (incl. casts, strided, `mean`/`var`/`std`,
  `argmin`/`argmax`), advanced indexing (gather/scatter/masks), `np.where`,
  views computed in C/Python without any ndarray, `a[key]` in the C slot,
  direct path and scalar shortcut for small ready data, adaptive dispatch.
- Last suite results: `docs/benchmarks.md` (4 CPUs: score 69 with 97
  cases, 68 with 100 cases incl. `kmeans_points`).
- ILNumerics' k-means program (`kmeans_points`, from
  github.com/hokb/decentralized-array-execution-artifacts2026): 3.9x vs.
  NumPy on 4 cores (1203 x 1203, k = 50), bit-identical.
- Next, by expected gain: async `matmul`/gufuncs; scaling analysis on many
  cores; per-op overhead of the remaining Python paths (see open
  questions).

## Open questions

- [x] Spike: NumPy loops from a worker thread without the GIL work for
      element-wise ufuncs (incl. 0-d broadcast), `add.reduce`, linalg
      `det`, `fft` and `argmax`; FP flags are visible on the worker
      (`tests/test_numpy_loops.cpp`, NumPy 2.5.3).
- [x] `should_offload`: hybrid N_min + adaptive per-site policy (see
      "Dispatch decision").
- [ ] Dispatch: op cost classes for N_min (a waiting main thread running a
      queued task itself was decided against, see "Dispatch decision").
- [ ] `matmul`/`dot` and the other gufuncs (linalg, fft) asynchronously:
      the largest remaining fallback (`matmul_batch`).
- [ ] Scaling beyond 4 cores: on a 20-core i7-12700K (8 P + 4 E cores,
      HT) the suite scores 22 with medium/large at 11 (`chains_wide`
      medium 6.2x of ideal 20x, large 1.2–3.9x): find out whether memory
      bandwidth, the main thread's issue rate or worker wake-ups limit it.
- [ ] Remaining per-op overhead on small data: views still handled in
      Python (`reshape` 0.6 µs, `swapaxes` ~1 µs via NumPy's
      `__array_function__`), index parsing for advanced indexing (3–10 µs
      small), direct-path result objects (~0.25 µs over NumPy).
- [x] Per-op overhead of the bridge: hot path moved to C (~35 µs → ~3–7 µs).
- [ ] Evaluate a generic path under free-threaded CPython: workers call the
      regular NumPy entry (no reimplementation of NumPy semantics).
- [ ] Bohrium license (Apache-2.0 vs. LGPLv3, see above) and viproc's own license.
- [ ] Run `docker/windows` (MSVC) on a Windows host; so far the Windows
      code paths are verified with the Wine image (llvm-mingw build).
- [ ] MSVC warning level: `-Wall -Wextra -Wpedantic -Werror` apply to
      GCC/Clang only; decide on `/W4 /WX` once an MSVC build is in CI.
