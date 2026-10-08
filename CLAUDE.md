# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**viproc** is a virtual processor: it takes a *serial* stream of array
instructions (`c = a + b`, `m = max(c)`, `s = sum(c, axis=0)`, …) and executes
it autonomously and asynchronously on parallel, heterogeneous hardware. The
program author writes ordinary sequential array code; viproc schedules and
dispatches the work.

The execution model follows the **ILNumerics Accelerator**: the caller's
thread does not compute. It only *issues* array instructions, records their
data dependencies, and returns immediately. Each instruction runs as soon as
the instructions producing its inputs have finished. Independent instructions
run concurrently; dependent ones keep only the local order needed for correct
results. Results must always be identical to plain sequential execution.

### Phase 1 (current): MVP on CPUs

Goal: autonomous, asynchronous execution of array instructions on
multicore CPUs.

- **Compute kernels: official NumPy C code.** We reuse NumPy's
  implementation (ufunc inner loops, reductions, SIMD dispatch, strided
  iteration, `npymath`) instead of writing our own kernels.
- **Async execution / glue layer (ours):** sits between the user ops
  (`add`, `sum`, `max`, …) and the NumPy ndarray layer. It owns the
  instruction queue, dependency tracking, scheduling onto worker threads,
  and synchronization when the caller needs a concrete value.
- **Python integration: adopted from Bohrium's `npbackend`**
  (`bridge/npbackend` in https://github.com/bh107/bohrium): an `ndarray`
  subclass whose operations are intercepted and forwarded to the runtime
  instead of being executed eagerly; data is synced back to NumPy on access.

Out of scope for phase 1: .NET/ILNumerics bindings, GPUs, distributed
execution, kernel fusion and JIT code generation. Do not add them, but do
not make design decisions that rule them out (keep device and memory
handling behind an interface).

## Tech stack

- **C++20** for the runtime, **C11** for the public ABI and for code that
  interfaces with NumPy's C sources. Same languages as NumPy (C) and
  Bohrium (C/C++).
- **CMake ≥ 3.24** with Ninja.
- Tests: CTest. Plain C/C++ test executables for now; pytest once the
  Python bridge exists.
- Formatting: `.clang-format` (LLVM style, 4 spaces, 100 columns).

## Repository layout

```
include/viproc/   Public C ABI (viproc.h). The only interface bridges may use.
src/runtime/      C++ async execution / glue layer
tests/            CTest test executables
```

Planned (not created yet):

```
third_party/numpy/   Pinned NumPy sources (git submodule), kernels only
src/kernels/         Thin adapters that call NumPy's loops on viproc buffers
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
Runtime           instruction queue · dependency tracking ·
                  scheduler · worker threads · sync       ← ours
        │
Kernel adapters   map an instruction onto a NumPy loop
        │
NumPy C kernels   ufunc loops, reductions, SIMD          ← from NumPy
```

Core concepts:
- **Array handle:** opaque runtime handle with shape, byte strides,
  element type and a reference to its buffer. Readers and writers of a
  buffer are tracked per handle and buffer.
- **Instruction:** an opcode plus input and output handles. Issuing one
  never blocks (except for back-pressure when the queue is full).
- **Dependencies:** read-after-write, write-after-read and write-after-write
  on overlapping buffers. Start conservatively: any instruction touching the
  same buffer is ordered.
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
- **NumPy**: ufunc loops are reachable through the ufunc C API
  (`char **args, npy_intp const *dimensions, npy_intp const *steps, void *data`);
  this is the calling convention our kernel adapters target.

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

## Licensing

- NumPy: BSD-3-Clause.
- Bohrium: the root `LICENSE` is Apache-2.0, but source file headers
  (including `bridge/npbackend/src/*`) say LGPLv3+. Clarify which applies
  before copying Bohrium code into this repo, and keep its copyright
  headers intact.
- viproc itself has no license file yet.

## Open questions

- [ ] Which NumPy version to pin, and how much of it to vendor (full
      submodule vs. just `numpy/_core/src/umath`, `npymath`, …)?
- [ ] Do we call NumPy's kernels through the installed NumPy at runtime
      (ufunc C API, needs CPython) or compile its loops into the runtime
      (no CPython dependency, much more build work)?
- [ ] Threading inside a single instruction (split large arrays across
      workers) or only between instructions in the MVP?
- [ ] Which ops are in the MVP set? Proposal: element-wise `add`,
      `subtract`, `multiply`, `divide`, plus reductions `sum` and `max`, on
      `float64` and `int64`.
- [ ] Bohrium license (Apache-2.0 vs. LGPLv3, see above) and viproc's own license.
