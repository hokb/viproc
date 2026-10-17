# viproc

**viproc strongly scales regular NumPy programs for any data on common (parallel)
hardware without code changes.**

viproc, a **vi**rtual **proc**essor for numerical programs, turns array instructions into
autonomously scheduling workloads. At runtime, thousands of such tasks cooperatively make
local decisions based on the actual data and hardware state, involving parallel computing
resources efficiently.

```python
c = np.sin(a) * b     # issued, returns at once
d = np.cos(b) - a     # does not need c: runs at the same time
s = (c + d).sum()     # starts as soon as c and d are done
print(s)              # only here does the program wait
```

Every array instruction becomes a task. A task starts as soon as its inputs exist, on
whichever core is free. The Python thread does not compute: it only issues instructions and
runs ahead. Unlike existing approaches, viproc does not analyse program dependencies at all,
and there is no central scheduler.
Parallelism thus appears wherever the data allows it, across loops, functions and modules,
in unchanged programs (`python -m viproc prog.py`). Today the tasks run on the cores of a
multi-core CPU; the model is designed for heterogeneous hardware.

## How it works

- **Granularity: the array operation.** One instruction is one NumPy operation (`a + b`,
  `np.sin(x)`, `x.sum(axis=0)`, `x[idx]`, `a @ b`, ...). viproc does not split single
  operations across threads; it overlaps whole array instructions, the way a superscalar
  CPU overlaps machine instructions.
- **Scope: a dynamic window of active instructions that runs ahead.** Every instruction
  issued but not yet completed is part of the window. Because the Python thread runs far
  ahead of the computation, the window spans loop iterations, function calls and module
  boundaries, wherever the program happens to go. There is no dependency graph and no
  ahead-of-time analysis. Data-dependent control flow is an obstacle, though: a branch on
  array values (`if (x > 0).any():`) needs those values, so the look-ahead stops there.
- **Synchronization points: element access and memory.** Only reading values
  (`print(x)`, `float(s)`, `if (x > 0).any():`) makes the Python thread wait, for that one
  result; all earlier instructions keep computing. The window is bounded too (today 1000
  active instructions), which bounds the memory held by results in flight.

The execution model follows the
[ILNumerics Accelerator](https://ilnumerics.net/auto-parallelizing-numerical-code.html), described in the paper
[*A Virtual Processor brings back the Free Lunch*](https://arxiv.org/abs/2605.30507)
(2026; benchmark artifacts:
[decentralized-array-execution-artifacts2026](https://github.com/hokb/decentralized-array-execution-artifacts2026)).

## What makes it different

- **Array ILP for NumPy.** The parallelism comes from the sequential stream of array
  instructions alone: no annotations, no rewrite, no ahead-of-time analysis.
- **Fine-grained scheduling parallelizes where manual parallelization cannot.** Scheduling
  decisions are made per array operation, so parallel execution is automated even where
  manual parallelization finds no parallelism because its scope is too coarse: a loop whose
  iterations depend on their order cannot be split with `Parallel.For` or `prange`, but the
  independent array operations inside and across its iterations still overlap. Example from
  the [decentralized array execution artifacts](https://github.com/hokb/decentralized-array-execution-artifacts2026/tree/main/Appendix/Part%202b%20Loop%20Parallelization)
  (in NumPy terms; later iterations overwrite what earlier ones wrote):
  ```python
  for i in range(A.shape[1]):
      ES[:i + 1] = np.abs(np.sin(A[:, i])).sum()
  ```
  `Parallel.For` gives wrong results here; the ILNumerics Accelerator runs the loop about 3x
  faster than sequentially (1500 x 1500), with correct results.
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
