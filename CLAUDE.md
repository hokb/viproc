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
  (DAG) of the program and no ahead-of-time (AOT) analysis. Each array
  carries its own state (`pending` or `completed`); that per-array state
  is the only dependency bookkeeping. Do not introduce global analysis
  structures.
- **The main thread runs ahead.** The Python main thread never waits for
  results (except at sync points). It races (far) ahead of the
  computation, and that implicit look-ahead is where the parallelism comes
  from: by the time early results complete, many later instructions are
  already issued and waiting on exactly their inputs.

### Issuing an array op (main thread)

For every array op, the main thread only does this, then continues with
the next op:

1. **Create a task** for the op and store its input arrays in it, for
   later use when the task runs.
2. **Allocate the output array** as metadata only (shape, strides, dtype);
   no element memory yet. Mark it `pending` and reference it from the
   task. The element memory is allocated when the task runs.
3. **Register the task** with every input that is still `pending`. The task
   is triggered when the last of them becomes `completed` (completed by
   the preceding tasks that produce them).
4. **If all inputs are already `completed`,** trigger the task right away
   (see "Dispatch decision").

When a task finishes, it marks its output `completed` and triggers the
waiting tasks whose inputs are now all completed.

Consequences:
- The output's shape and dtype must be known at issue time without
  looking at data (broadcasting, `resolve_dtypes`). Ops whose result shape
  depends on data (boolean indexing, `nonzero`, `unique`, …) are sync
  points and fall back to eager NumPy.
- Registering a task on an input and that input completing can happen at
  the same time on different threads; the check "still pending? then
  register" must be atomic per array so no task is lost or triggered twice.

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
  per-array readiness state, the queue of instructions handed to worker
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
Runtime           per-array readiness · local dispatch decision ·
                  worker threads · sync                   ← ours
        │
Kernel adapters   map an instruction onto a NumPy loop
        │  function pointers resolved once, called without the GIL
NumPy C kernels   ufunc inner loops, SIMD                ← installed NumPy
```

Core concepts:
- **Array handle:** opaque runtime handle with shape, byte strides,
  element type, state (`pending` / `completed`) and a reference to its
  buffer. A pending array has no element memory until its producing task
  runs. Readers and writers of a buffer are tracked per handle and buffer.
- **Task:** an op plus references to its input and output arrays and the
  resolved NumPy loop. Issuing one never blocks (except for back-pressure
  if the main thread gets too far ahead).
- **Dispatch decision (per task, at issue time):**
  - If any input array is still `pending`, the task waits on those inputs
    and later runs on a worker thread.
  - If all inputs are `completed`, `should_offload(task)` decides: `true`
    → hand it to a worker, `false` → execute it immediately on the main
    thread. `should_offload` is a heuristic still to be defined
    (candidates: element count, op cost class, current queue length).
    Keep it a single, swappable function.
  - Readiness also covers the output buffer: if pending instructions still
    read or write it (WAR/WAW), the task counts as pending.
- **Dependencies:** read-after-write, write-after-read and write-after-write
  on overlapping buffers, tracked locally on each buffer (no global graph).
  Start conservatively: any instruction touching the same buffer is
  ordered. When a pending instruction finishes, it marks its outputs
  ready and releases the instructions waiting on exactly those buffers.
- **Sync points:** reading data from the caller (printing, converting to a
  scalar, handing memory back to plain NumPy) waits only for the
  instructions that produce that data.
- **Errors** from asynchronous execution are stored on the affected result
  and raised at the next sync point that touches it.

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
- [ ] Back-pressure: how far may the main thread run ahead (number of
      pending tasks, or memory of pending outputs) before it blocks?
- [ ] Bohrium license (Apache-2.0 vs. LGPLv3, see above) and viproc's own license.
