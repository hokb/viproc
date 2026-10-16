# viproc

**Array instruction-level parallelism for NumPy programs.** viproc, a virtual processor for
serial array instruction streams, finds the parallel potential between the array instructions
of a regular, sequentially written NumPy program, automatically and without annotations, and
runs it on parallel hardware.

## What viproc does

- **Finds parallel potential automatically, across large parts of a program.** viproc looks
  at the stream of array instructions a NumPy program issues (`c = a + b`, `m = c.max()`,
  `s = c.sum(axis=0)`, ...) and finds which of them can run at the same time, across loops,
  functions and modules. The program is not rewritten and carries no annotations, decorators or
  parallel loops: viproc is switched on from outside (`python -m viproc prog.py`).
- **Puts that potential to work on the available parallel resources, dynamically.** Work is
  handed to compute resources at runtime, as instructions become ready, and small
  instructions whose result is needed right away stay on the calling thread. Today the
  resources are the cores of a multi-core CPU; the execution model is meant for
  heterogeneous hardware.

## How it works

viproc is different in that it avoids the restrictions of existing approaches by not even
attempting to analyse program dependencies globally. Instead, array operations decide their
own scheduling at runtime, in parallel and autonomously. There is no central scheduler.

All array instructions are first freed from the sequential order the program text gives
them. When they run, their order is decided strictly and only by their input data and its
availability: an instruction runs as soon as its inputs are computed, independent
instructions run at the same time, dependent ones keep only the order their data needs.
Local availability (is this input ready, or still being computed?) replaces any global
program or dependency analysis.

- **Granularity: the array operation.** One instruction is one NumPy operation (`a + b`,
  `np.sin(x)`, `x.sum(axis=0)`, `x[idx]`, `a @ b`, ...). viproc does not split single
  operations across threads; it overlaps whole array instructions, the way a superscalar
  CPU overlaps machine instructions.
- **Scope: a dynamic window of active instructions that runs ahead.** The Python thread does
  not compute. It only issues instructions and returns at once, racing far ahead of the
  computation. Every instruction issued but not yet completed is part of the window, so
  the parallel potential is found across loop iterations, function calls and module
  boundaries, wherever the program happens to go. There is no dependency graph and no
  ahead-of-time analysis. Data-dependent control flow is an obstacle, though: a branch on
  array values (`if (x > 0).any():`) needs those values, so the look-ahead stops there.
- **Synchronization points: element access and memory.** Where the program needs actual
  values (`print(x)`, `float(s)`, `if (x > 0).any():`, scalar indexing, handing data to
  code outside viproc), only the Python thread waits, at the position it has run ahead
  to, and only for the instruction it needs. The array computations of all earlier
  instructions keep progressing in parallel. The window is also bounded, which bounds the
  memory held by results in flight: when it is full (today 1000 active instructions), the
  Python thread waits until part of it has completed.

The execution model follows the
[ILNumerics Accelerator](https://ilnumerics.net/auto-parallelizing-numerical-code.html), described in the paper
[*A Virtual Processor brings back the Free Lunch*](https://arxiv.org/abs/2605.30507)
(2026; benchmark artifacts:
[decentralized-array-execution-artifacts2026](https://github.com/hokb/decentralized-array-execution-artifacts2026)).

## What makes it different

- **Array ILP from the sequential instruction stream.** All the parallel potential of wide
  program sections is found automatically from the stream of array operations alone, at
  runtime, without annotations or a rewrite of the program, and put to work on the parallel
  resources available.
- **Identical results.** Every instruction is computed by NumPy's own inner loops (including
  their SIMD paths), called exactly as NumPy calls them. Results are bit-identical to plain
  NumPy; whatever viproc cannot run asynchronously with NumPy's semantics falls back to
  NumPy itself.
- **Never slower than NumPy.** viproc must not make any expression slower than NumPy
  (hard limit 0.95x; where there is nothing to overlap, at most 3% slower by design).
- **Designed for heterogeneous hardware and further JIT optimizations.** viproc runs on
  multi-core CPUs today, but the approach is designed to include heterogeneous compute
  resources into execution fully automatically, by adding cost-based device scheduling to
  the autonomous decisions of the array operations. Further JIT optimizations apply, too:
  pipelining (partial execution starts as soon as dependencies are partially completed),
  operator fusion and multi-stage low-level kernel specialization. Some of them are already
  deployed successfully in the ILNumerics Accelerator.

The goal is strong scaling: a fixed program with a fixed problem size gets faster with more
cores. Goals, non-goals and rules are in [docs/design/goals.md](docs/design/goals.md),
measurements in [docs/benchmarks.md](docs/benchmarks.md), architecture and build details in
[CLAUDE.md](CLAUDE.md).

## Where it helps, and where it does not (yet)

viproc gains where a program issues many array instructions that do not all depend on each
other: several independent expressions, chains over different arrays, reductions next to
element-wise work, iterative algorithms over arrays (k-means, simulations, image and signal
pipelines), on medium and large data.

Programs that spend their time in long algorithms or loops over **scalar** data, one element
at a time, offer little array-level parallelism and gain little from viproc today (they must
not get slower either). For them the manual options remain: vectorizing the loop into array
operations, or parallelizing it explicitly (`multiprocessing`, `concurrent.futures`, Numba's
`prange`, ...). Overlapping consecutive loop iterations automatically (instruction
pipelining) is on the roadmap.

## Usage

Run an unchanged NumPy program with viproc:

```bash
python -m viproc prog.py [args]       # also: -m module, -c "code"; no arguments: console
VIPROC=1 python prog.py               # the same, for a pip-installed viproc (viproc.pth)
```

`import numpy` in the program's own code then imports viproc: creation functions
(`np.zeros`, `np.arange`, `np.loadtxt`, `np.random.*`, `default_rng()`, ...) return viproc
arrays, and NumPy functions on them run asynchronously. The standard library and installed
packages (SciPy, pandas, ...) keep the real NumPy; viproc arrays they receive are synced and
read through NumPy's array protocols. Results are bit-identical to plain NumPy.

Or opt in explicitly:

```python
import numpy as np
import viproc

a = viproc.asarray(np.random.rand(1_000_000))   # or: import viproc as np
b = viproc.asarray(np.random.rand(1_000_000))
c = np.sin(a) * b + 1.0      # returns immediately, computed in the background
d = np.cos(b) - a            # independent of c: runs in parallel
print((c + d)[:5])           # sync point
```

### Benchmarking: the same code with and without viproc

```python
import time
import numpy as np            # viproc under python -m viproc
import viproc

def program():
    x = np.random.default_rng(0).standard_normal((2000, 2000))
    return float((np.sin(x) * np.cos(x)).sum())

t0 = time.perf_counter(); program(); t_viproc = time.perf_counter() - t0
with viproc.disabled():       # creation functions return NumPy arrays: plain NumPy
    t0 = time.perf_counter(); program(); t_numpy = time.perf_counter() - t0
```

`viproc.enable(False)` / `viproc.enable(True)` switch it for longer stretches. Arrays created
while viproc was on stay viproc arrays; convert them with `np.asarray(x)` (copy-free,
read-only) if the NumPy part should not touch them. Whole programs: `python prog.py` vs.
`python -m viproc prog.py`.

Differences from NumPy under `python -m viproc`: `isinstance(x, np.ndarray)` in user code
tests for viproc arrays; `np.array(nd)`/`np.asarray(nd)` of a NumPy array always copy it into
viproc (also with `copy=False`); dtypes viproc does not run (object, str, float16, ...) give
NumPy arrays, as NumPy does.

## Try it

viproc is a regular Python package: one extension module (`_viproc.*.pyd` / `.so`, runtime
linked in statically) plus Python files. It needs Python >= 3.12 (64 bit), NumPy >= 2.0 and,
until there are prebuilt wheels, a C++20 compiler (Visual Studio with the C++ workload on
Windows, GCC or Clang on Linux). NumPy itself is not modified: viproc arrays take part through
NumPy's array protocols.

One click, after every commit (pull, build into `.venv`, run the tests):

```bat
dev.cmd                 :: Windows: double-click, or dev.cmd -NoPull -NoTest -Python <python.exe>
```
```bash
./dev.sh                # Linux: ./dev.sh --no-pull --no-test, PYTHON=python3.13 ./dev.sh
```

Then run programs with `.venv\Scripts\python prog.py` (Linux: `.venv/bin/python prog.py`). The first
run takes a minute; later ones rebuild incrementally (`build/pip-<tag>`). No developer prompt is
needed on Windows: the build picks Visual Studio for the interpreter's architecture.

Into any environment: `pip install .` (or `pip install git+https://github.com/hokb/viproc`).

## Build (development, CMake)

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure     # needs: pip install pytest
PYTHONPATH=build/python python3 python/benchmarks/strong_scaling.py
PYTHONPATH=build/python python3 python/benchmarks/suite.py      # benchmark suite: score + HTML report
```

## Docker

Images for Linux, Windows (native, MSVC) and Windows-under-Wine build this checkout, run all
test suites by default and serve as a command line ([docker/README.md](docker/README.md)):

```bash
docker build -f docker/linux/Dockerfile -t viproc:linux .
docker run --rm viproc:linux                                # all test suites
docker run --rm -v "$PWD:/work" viproc:linux python prog.py # run a program with viproc
```
