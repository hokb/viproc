# viproc

**Array instruction-level parallelism for NumPy programs.** viproc, a virtual processor for
serial array instruction streams, finds the parallel
potential between the array instructions of a regular, sequentially written NumPy program,
automatically and without annotations, and runs it on parallel hardware.

## What viproc does

- **Finds parallel potential automatically, across large parts of a program.** viproc looks
  at the stream of array instructions a NumPy program issues (`c = a + b`, `m = c.max()`,
  `s = c.sum(axis=0)`, ...) and runs every instruction as soon as its inputs are ready.
  Independent instructions run at the same time, dependent ones keep only the order their
  data needs. The program is not rewritten and carries no annotations, decorators or
  parallel loops: viproc is switched on from outside (`python -m viproc prog.py`).
- **Puts that potential to work on the available parallel resources, dynamically.** Work is
  handed to compute resources at runtime, as instructions become ready, and small
  instructions whose result is needed right away stay on the calling thread. Today the
  resources are the cores of a multi-core CPU; the execution model is meant for
  heterogeneous hardware.

## What makes it different

- **Array ILP instead of data parallelism.** viproc does not split single operations across
  threads. It overlaps whole array instructions, the way a superscalar CPU overlaps machine
  instructions. The parallelism is found while the program runs, from the sequential stream
  of array operations alone: the Python thread only issues instructions and races far
  ahead of the computation, and every decision is local (is this input ready or still being
  computed?). There is no dependency graph and no ahead-of-time analysis, so loops, branches
  and data-dependent control flow are no obstacle. The execution model follows the
  ILNumerics Accelerator.
- **Identical results.** Every instruction is computed by NumPy's own inner loops (including
  their SIMD paths), called exactly as NumPy calls them. Results are bit-identical to plain
  NumPy; whatever viproc cannot run asynchronously with NumPy's semantics falls back to
  NumPy itself.
- **Never slower than NumPy.** viproc must not make any expression slower than NumPy
  (hard limit 0.95x; where there is nothing to overlap, at most 3% slower by design).

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
