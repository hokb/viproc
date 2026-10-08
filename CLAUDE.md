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
- A pending array may have no element memory yet; the producer allocates
  it.
- Each array has a dedicated atomic counter **`async_reads`**: the number
  of issued, not yet completed tasks that read the array as an input.
  - The main thread increments it when it issues a task reading the array.
  - The reading task decrements it when it has finished reading (at the
    latest on `completed`), on whatever thread it runs.
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
- TODO: performance, caching / pooling of task objects.

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

- Tasks, arrays and array memory (buffers) are **reference counted**,
  with atomic counts (they are shared across threads).
- References: Python object → array; array → producer task (while
  `pending`); task → its input and output arrays; producer task →
  registered callbacks → consumer tasks.
- Cycles (array → producer → callback → consumer → array) are broken on
  `completed`: the producer releases its callbacks and its inputs, and a
  `ready` array drops its producer reference.
- **Write-after-read safety via `async_reads`.** Arrays do not know their
  pending readers, only how many there are. When the main thread issues a
  task that writes an array (in/out argument, `out=`), it checks
  `async_reads`:
  - `0`: the task may write the existing buffer in place.
  - `> 0`: pending readers still need the old data. The writing task must
    not overwrite that buffer; it writes into a new buffer, and the array
    switches to it. The old buffer stays alive through the readers'
    references and is freed when the last one releases it.
- Sync points that hand memory back to plain NumPy for *writing* also
  wait until `async_reads == 0`.

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
- **Python integration: adopted from Bohrium's `npbackend`**
  (`bridge/npbackend` in https://github.com/bh107/bohrium): an `ndarray`
  subclass whose operations are intercepted and forwarded to the runtime
  instead of being executed eagerly; data is synced back to NumPy on access.

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
include/viproc/   Public C ABI (viproc.h). The only interface bridges may use.
src/runtime/      C++ async execution / glue layer
tests/            CTest test executables
                  test_numpy_loops.cpp: embeds CPython and proves that NumPy
                  loops run bit-identically on a worker thread without the GIL
```

Planned (not created yet):

```
src/kernels/         Adapters that resolve and call NumPy's ufunc loops
python/              Python bridge derived from Bohrium npbackend
```

Rules:
- Bridges (Python, later .NET) only talk to the runtime through
  `include/viproc/viproc.h`. No C++ types cross the ABI.
- Never edit vendored third-party code in place. Wrap it, and if a patch is
  unavoidable, keep it as a separate patch file and document why.

## Commands

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug   # configure
cmake --build build                                     # build
ctest --test-dir build --output-on-failure              # run tests
clang-format -i <files>                                 # format
```

Build with `-DVIPROC_WARNINGS_AS_ERRORS=OFF` only for local experiments;
the default build treats warnings as errors. Always build and run the tests
before committing.

## Architecture (phase 1)

```
Python user code  (import viproc as np / python -m viproc)
        │
Python bridge     ndarray subclass, op interception      ← from Bohrium npbackend
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
  are all `ready`):** `should_offload(task)` decides: `true` → hand it to a
  worker, `false` → execute it immediately on the main thread.
  `should_offload` is a heuristic still to be defined (candidates: element
  count, op cost class, number of pending tasks). Keep it a single,
  swappable function. Tasks triggered by a callback run on the thread that
  completed their last input, as a continuation (see "Issuing an op").
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
- **Reductions** (`sum`, `prod`, `min`, `max`, `any`, `all`, …) use the
  binary loop resolved with `reduction=True`, called with an output stride
  of 0 (`args = {acc, in, acc}`), seeded with the identity or the first
  element. `axis=None` first, `axis=k` and `keepdims` later. NumPy's float
  `add` loop does pairwise summation inside one call, so splitting a
  reduction into chunks changes rounding; call the loop with the same
  blocks NumPy would to stay bit-identical.
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

## Open questions

- [x] Spike: NumPy loops from a worker thread without the GIL work for
      element-wise ufuncs (incl. 0-d broadcast), `add.reduce`, linalg
      `det`, `fft` and `argmax`; FP flags are visible on the worker
      (`tests/test_numpy_loops.cpp`, NumPy 2.5.3).
- [ ] `should_offload` heuristic for tasks whose inputs are all completed.
- [ ] Writer with `async_reads > 0`: new buffer (as described) or wait
      for the readers? — confirm.
- [ ] A completing producer may unblock several consumers at once: the
      completing thread runs one as its continuation; the others go to
      idle workers / the pool queue.
- [ ] Back-pressure: how far may the main thread run ahead (number of
      pending tasks, or memory of pending outputs) before it blocks?
- [ ] Bohrium license (Apache-2.0 vs. LGPLv3, see above) and viproc's own license.
