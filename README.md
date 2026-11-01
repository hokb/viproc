# numpy2x

**numpy2x runs NumPy programs faster. No annotations. Bit-equal results. Single node strong scaling.**

numpy2x turns array instructions into
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
runs ahead. Unlike existing approaches, numpy2x does not analyse program dependencies at all,
and there is no central scheduler.
Parallelism thus appears wherever the data allows it, across loops, functions and modules,
in unchanged programs (`python -m numpy2x prog.py`). Today the tasks run on the cores of a
multi-core CPU; the model is designed for heterogeneous hardware.

## Complexity becomes performance

The question numpy2x answers is not whether a single array operation runs as fast as, or
faster than, NumPy. It is: **can whole programs scale strongly**, a fixed program with a
fixed problem size getting faster with more cores?

A benchmark that consists of a single operation does not get faster with numpy2x: there is
nothing to overlap it with, and numpy2x does not split single operations. Real programs are
not single operations. They compute several quantities, branch into independent
sub-expressions, mix reductions with element-wise work, loop over arrays and call functions
in modules. **In practice, complexity always comes with parallel potential**: the more a
program does, the more of its array instructions do not depend on each other. numpy2x finds
that potential at runtime, without analysis or annotations, and turns it into parallel
execution. **numpy2x turns complexity into performance.**

The goals follow from this, and they are stated for whole programs, not for single
operations:

- **Every program runs at 0.8x of NumPy's speed or faster** (current goal), also where it
  offers little parallel potential.
- **Complex programs run clearly faster than NumPy**, more so with more cores.
- **Parallel potential becomes parallelism**: wherever independent array instructions
  exist, they run concurrently.

Benchmarks are judged accordingly: by whole programs of growing complexity (the
[benchmark suite](docs/benchmarks.md), the shape maps over program groups, ILNumerics'
k-means program at 3.9x of NumPy on 4 cores), not by the speed of one operation in
isolation.

## How it works

- **Granularity: the array operation.** One instruction is one NumPy operation (`a + b`,
  `np.sin(x)`, `x.sum(axis=0)`, `x[idx]`, `a @ b`, ...). numpy2x does not split single
  operations across threads; it overlaps whole array instructions, the way a superscalar
  CPU overlaps machine instructions.
- **Scope: a dynamic window of active instructions that runs ahead.** Every instruction
  issued but not yet completed is part of the window. Because the Python thread runs far
  ahead of the computation, the window spans loop iterations, function calls and module
  boundaries, wherever the program happens to go. 
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
- **Identical results, guaranteed (today).** numpy2x currently guarantees results that are
  bit-identical to plain NumPy: values, dtypes, shapes, strides, warnings and errors.
  Every instruction is computed by the installed NumPy version's own inner loops and
  operators (including their SIMD paths), called exactly as NumPy calls them; whatever
  numpy2x cannot run asynchronously with NumPy's semantics falls back to NumPy itself.
  The price is a small overhead on small computations: the numpy2x layer (task, arrays,
  dispatch) comes on top of NumPy's own work, while the kernels themselves cannot get
  faster than NumPy's.
- **Faster where there is parallel potential; at least 0.8x of NumPy elsewhere.** Programs
  with independent array operations on medium-sized and large data numpy2x commonly runs
  efficiently in parallel. Every program is to run at 0.8x of NumPy's speed or faster (the
  current goal, see [Complexity becomes performance](#complexity-becomes-performance)); for
  single expressions on small data without parallel potential, the long-term target is
  NumPy's speed (at most 3% slower, never below 0.95x). Bit-identical results come first:
  the small per-op overhead described above cannot always be hidden behind parallel work,
  so these are targets, not guarantees (small data is not quite there yet).

The goal is strong scaling: a fixed program with a fixed problem size gets faster with more
cores. Goals, non-goals and rules are in [docs/design/goals.md](docs/design/goals.md),
measurements in [docs/benchmarks.md](docs/benchmarks.md), architecture and build details in
[CLAUDE.md](CLAUDE.md).

## Where it helps, and where it does not (yet)

numpy2x gains where a program issues many array instructions that do not all depend on each
other: several independent expressions, chains over different arrays, reductions next to
element-wise work, iterative algorithms over arrays (k-means, simulations, image and signal
pipelines), on medium and large data. The more complex a program gets, the more parallel
potential it brings: numpy2x shines on real-world algorithms, not on toy examples of one or
two operations.

numpy2x turns complexity into performance. A benchmark of one operation cannot get faster:
there is nothing to run next to it. Real programs consist of many operations, and complexity
always brings parallel potential with it; numpy2x turns that potential into parallelism. The
goal: every program runs at 0.8x NumPy's speed or faster, complex programs run clearly faster.

Example: ILNumerics' array-pipelining demo, a plain Python loop over the columns of a matrix
(`python/benchmarks/array_pipelining.py`), 1,500 x 1,500 doubles, 4 cores, bit-identical:

```python
for i in range(A.shape[1]):
    C[i] = np.sum(np.abs(np.sin(A[:, i])), axis=0)        # loop
    # C[0:i+1] = ...   every iteration overwrites the earlier ones (loop_dep)
```

| | NumPy | NumPy + thread pool (manual Parallel.For) | numpy2x, 4 workers |
|---|---:|---:|---:|
| `loop` | 1.0x | 0.25x | 1.7-2.0x |
| `loop_dep` | 1.0x | wrong results | 1.7x |
| `np.sum(np.abs(np.sin(A)), axis=0)` | 1.0x | – | 3.4x |

Nothing in the loop is annotated: numpy2x overlaps the small operations of different
iterations and keeps the order where it matters (`loop_dep`). The gain is bounded by how fast
the Python thread issues the operations (~7 µs per iteration here); details in
[docs/benchmarks.md](docs/benchmarks.md).

Programs that spend their time in algorithms or loops over **scalar** data, one element
at a time, offer little array-level parallelism and gain little from numpy2x today (the
target is NumPy's speed for them; the per-op overhead can make them slightly slower). For
them the manual options remain: vectorizing the loop into array operations, or parallelizing it explicitly (`multiprocessing`, `concurrent.futures`, Numba's
`prange`, ...).

## Install

```bash
git clone https://github.com/hokb/numpy2x
cd numpy2x
pip install .           # into the active Python environment (a venv, conda, ...)
```

numpy2x is a regular Python package: one extension module (`_numpy2x.*.pyd` / `.so`, runtime
linked in statically) plus Python files. It needs Python >= 3.12 (64 bit), NumPy >= 2.5 and,
until there are prebuilt wheels, a C++20 compiler (Visual Studio with the C++ workload on
Windows, GCC or Clang on Linux). NumPy itself is not modified: numpy2x arrays take part through
NumPy's array protocols.

Without a checkout, `pip install git+https://github.com/hokb/numpy2x` does the same.

There is no prebuilt wheel yet, so pip builds one from the sources on your machine and installs
it: its build backend (scikit-build-core, see `pyproject.toml`) runs CMake, compiles the
extension module and packs it into a wheel. pip fetches CMake and Ninja itself if they are
missing; the C++20 compiler named above must be installed. The build takes about a minute.
`pip wheel .` instead keeps the wheel (`numpy2x-*.whl`), to install on other machines with the same
platform, Python version and architecture.

Testing numpy2x on your own code, in your existing Python and IDE (PyCharm, VS Code, Spyder,
Jupyter): see [docs/testing.md](docs/testing.md).

## Usage

Installing makes numpy2x available to your Python. To bring it into an existing NumPy program,
use one of three ways:

**1. Run the program with `python -m numpy2x`** (no code change):

```bash
python -m numpy2x prog.py [args]       # also: -m module, -c "code"; no arguments: console
NUMPY2X=1 python prog.py               # the same, through numpy2x.pth
python prog.py                        # plain NumPy, for comparison
```

`import numpy` in the program's own code then actually imports numpy2x: all NumPy functions
(`np.zeros`, `np.arange`, `np.loadtxt`, `np.random.*`, `np.sin`, `np.sum`, ...) are there. The
ones that create arrays return numpy2x arrays, and the ones on numpy2x arrays run asynchronously
where numpy2x supports them, otherwise in NumPy itself. The standard library and installed
packages (SciPy, pandas, ...) keep the real NumPy; numpy2x arrays they receive are synced and
read through NumPy's array protocols. Results are bit-identical to plain NumPy.

**2. Import numpy2x in place of NumPy** (one line changed):

```python
import numpy2x as np          # was: import numpy as np

x = np.random.default_rng(0).standard_normal((2000, 2000))   # a numpy2x array
y = np.sin(x) * np.cos(x)    # returns immediately, computed in the background
print(float(y.sum()))        # sync point
```

**3. Keep NumPy and convert selected arrays** (explicit, in the code):

```python
import numpy as np
import numpy2x

a = numpy2x.asarray(np.random.rand(1_000_000))   # NumPy array -> numpy2x array
b = numpy2x.asarray(np.random.rand(1_000_000))
c = np.sin(a) * b + 1.0      # NumPy's functions on numpy2x arrays run in numpy2x
d = np.cos(b) - a            # independent of c: runs in parallel
print((c + d)[:5])           # sync point
```

### Benchmarking: the same code with and without numpy2x

```python
import time
import numpy as np            # numpy2x under python -m numpy2x
import numpy2x

def program():
    x = np.random.default_rng(0).standard_normal((2000, 2000))
    return float((np.sin(x) * np.cos(x)).sum())

t0 = time.perf_counter(); program(); t_numpy2x = time.perf_counter() - t0
with numpy2x.disabled():       # creation functions return NumPy arrays: plain NumPy
    t0 = time.perf_counter(); program(); t_numpy = time.perf_counter() - t0
```

`numpy2x.enable(False)` / `numpy2x.enable(True)` switch it for longer stretches. Arrays created
while numpy2x was on stay numpy2x arrays; convert them with `np.asarray(x)` (copy-free,
read-only) if the NumPy part should not touch them. Whole programs: `python prog.py` vs.
`python -m numpy2x prog.py`.

Differences from NumPy under `python -m numpy2x`: `isinstance(x, np.ndarray)` in user code
tests for numpy2x arrays; `np.array(nd)`/`np.asarray(nd)` of a NumPy array always copy it into
numpy2x (also with `copy=False`); dtypes numpy2x does not run (object, str, float16, ...) give
NumPy arrays, as NumPy does.

## Build (development, CMake)

`dev.cmd` (Windows, also by double-click) and `./dev.sh` (Linux) pull, build into `.venv` and
run the tests; run programs with `.venv\Scripts\python -m numpy2x prog.py` (Linux:
`.venv/bin/python -m numpy2x prog.py`). Later builds are incremental (`build/pip-<tag>`), and no
developer prompt is needed on Windows: the build picks Visual Studio for the interpreter's
architecture. Options: `dev.cmd -NoPull -NoTest -Python <python.exe>`, `./dev.sh --no-pull
--no-test`, `PYTHON=python3.13 ./dev.sh`.

With CMake directly:

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
docker build -f docker/linux/Dockerfile -t numpy2x:linux .
docker run --rm numpy2x:linux                                # all test suites
docker run --rm -v "$PWD:/work" numpy2x:linux python prog.py # run a program with numpy2x
```
