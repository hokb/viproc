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
- **Identical results, guaranteed (today).** viproc currently guarantees results that are
  bit-identical to plain NumPy: values, dtypes, shapes, strides, warnings and errors.
  Every instruction is computed by the installed NumPy version's own inner loops and
  operators (including their SIMD paths), called exactly as NumPy calls them; whatever
  viproc cannot run asynchronously with NumPy's semantics falls back to NumPy itself.
  The price is a small overhead on small computations: the viproc layer (task, arrays,
  dispatch) comes on top of NumPy's own work, while the kernels themselves cannot get
  faster than NumPy's.
- **Planned: optimization levels.** Bit-identity is today's level, O1. Higher levels are
  to compute correct but not bit-identical results faster (see
  [Optimization levels](#optimization-levels-planned)).
- **Faster where there is parallel potential; NumPy's speed as the target elsewhere.**
  Array operations on medium-sized and large data viproc commonly runs efficiently in
  parallel. For small data and algorithms without much parallel potential, the target is
  NumPy's speed (at most 3% slower, never below 0.95x), but bit-identical results come
  first: the small per-op overhead described above cannot always be hidden, so "never
  slower than NumPy" is a target, not a guarantee, in the bit-identical mode (not quite
  there yet on small data). The planned optimization levels above O1 are where speed may
  win over bit-identity.
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

## Optimization levels (planned)

Bit-identical results are a consequence of today's implementation, not a fixed limit of
viproc. A future, multi-stage optimization setting decides how far viproc may go beyond
NumPy's own execution; each level adds to the one below:

| Level | What viproc does | Results |
|-------|------------------|---------|
| **O1** (current) | array ILP only: independent array instructions run in parallel; NumPy itself executes every function | bit-identical to NumPy |
| **O2** | + stage pipelining (an instruction starts on partial results of its inputs), BLAS threading removed (viproc's workers parallelize instead of BLAS) | correct, may deviate (BLAS results depend on its thread count) |
| **O3** | + micro kernels: viproc's own kernels, fused and split operations, partly re-implemented NumPy internals | correct, may deviate in rounding |
| **O4** | + automatic device offloading (GPUs and other accelerators, chosen per instruction) | correct, may deviate |

"Correct" means what NumPy's semantics allow: the same dtypes, shapes, errors and
warnings, values within the floating-point error of a valid evaluation order. O1 stays
available as the reference and for programs that need NumPy's exact bits. The higher
levels are also where the per-op overhead on small data can disappear, because viproc no
longer has to call NumPy's code exactly as NumPy does. Details:
[docs/design/goals.md](docs/design/goals.md), "Optimization levels".

## Where it helps, and where it does not (yet)

viproc gains where a program issues many array instructions that do not all depend on each
other: several independent expressions, chains over different arrays, reductions next to
element-wise work, iterative algorithms over arrays (k-means, simulations, image and signal
pipelines), on medium and large data. The more complex a program gets the more parallel potential it brings. viproc shines on real-world algorithms not on too simple toy examples. 

viproc turns complexity into performance. A benchmark of one operation cannot get faster:
there is nothing to run next to it. Real programs consist of many operations, and complexity
always brings parallel potential with it; viproc turns that potential into parallelism. The
goal: every program runs at 0.8x NumPy's speed or faster, complex programs run clearly faster.

Example: ILNumerics' array-pipelining demo, a plain Python loop over the columns of a matrix
(`python/benchmarks/array_pipelining.py`), 1,500 x 1,500 doubles, 4 cores, bit-identical:

```python
for i in range(A.shape[1]):
    C[i] = np.sum(np.abs(np.sin(A[:, i])), axis=0)        # loop
    # C[0:i+1] = ...   every iteration overwrites the earlier ones (loop_dep)
```

| | NumPy | NumPy + thread pool (manual Parallel.For) | viproc, 4 workers |
|---|---:|---:|---:|
| `loop` | 1.0x | 0.25x | 1.7-2.0x |
| `loop_dep` | 1.0x | wrong results | 1.7x |
| `np.sum(np.abs(np.sin(A)), axis=0)` | 1.0x | – | 3.4x |

Nothing in the loop is annotated: viproc overlaps the small operations of different
iterations and keeps the order where it matters (`loop_dep`). The gain is bounded by how fast
the Python thread issues the operations (~7 µs per iteration here); details in
[docs/benchmarks.md](docs/benchmarks.md).

Programs that spend their time in algorithms or loops over **scalar** data, one element
at a time, offer little array-level parallelism and gain little from viproc today (the
target is NumPy's speed for them; the per-op overhead can make them slightly slower). For
them the manual options remain: vectorizing the loop into array operations, or parallelizing it explicitly (`multiprocessing`, `concurrent.futures`, Numba's
`prange`, ...). Overlapping consecutive loop iterations automatically (instruction
pipelining) is on the roadmap.

## Install

```bash
git clone https://github.com/hokb/viproc
cd viproc
pip install .           # into the active Python environment (a venv, conda, ...)
```

viproc is a regular Python package: one extension module (`_viproc.*.pyd` / `.so`, runtime
linked in statically) plus Python files. It needs Python >= 3.12 (64 bit), NumPy >= 2.5 and,
until there are prebuilt wheels, a C++20 compiler (Visual Studio with the C++ workload on
Windows, GCC or Clang on Linux). NumPy itself is not modified: viproc arrays take part through
NumPy's array protocols.

Without a checkout, `pip install git+https://github.com/hokb/viproc` does the same.

There is no prebuilt wheel yet, so pip builds one from the sources on your machine and installs
it: its build backend (scikit-build-core, see `pyproject.toml`) runs CMake, compiles the
extension module and packs it into a wheel. pip fetches CMake and Ninja itself if they are
missing; the C++20 compiler named above must be installed. The build takes about a minute.
`pip wheel .` instead keeps the wheel (`viproc-*.whl`), to install on other machines with the same
platform, Python version and architecture.

Testing viproc on your own code, in your existing Python and IDE (PyCharm, VS Code, Spyder,
Jupyter): see [docs/testing.md](docs/testing.md).

## Usage

Installing makes viproc available to your Python. To bring it into an existing NumPy program,
use one of three ways:

**1. Run the program with `python -m viproc`** (no code change):

```bash
python -m viproc prog.py [args]       # also: -m module, -c "code"; no arguments: console
VIPROC=1 python prog.py               # the same, through viproc.pth
python prog.py                        # plain NumPy, for comparison
```

`import numpy` in the program's own code then actually imports viproc: all NumPy functions
(`np.zeros`, `np.arange`, `np.loadtxt`, `np.random.*`, `np.sin`, `np.sum`, ...) are there. The
ones that create arrays return viproc arrays, and the ones on viproc arrays run asynchronously
where viproc supports them, otherwise in NumPy itself. The standard library and installed
packages (SciPy, pandas, ...) keep the real NumPy; viproc arrays they receive are synced and
read through NumPy's array protocols. Results are bit-identical to plain NumPy.

**2. Import viproc in place of NumPy** (one line changed):

```python
import viproc as np          # was: import numpy as np

x = np.random.default_rng(0).standard_normal((2000, 2000))   # a viproc array
y = np.sin(x) * np.cos(x)    # returns immediately, computed in the background
print(float(y.sum()))        # sync point
```

**3. Keep NumPy and convert selected arrays** (explicit, in the code):

```python
import numpy as np
import viproc

a = viproc.asarray(np.random.rand(1_000_000))   # NumPy array -> viproc array
b = viproc.asarray(np.random.rand(1_000_000))
c = np.sin(a) * b + 1.0      # NumPy's functions on viproc arrays run in viproc
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

## Build (development, CMake)

`dev.cmd` (Windows, also by double-click) and `./dev.sh` (Linux) pull, build into `.venv` and
run the tests; run programs with `.venv\Scripts\python -m viproc prog.py` (Linux:
`.venv/bin/python -m viproc prog.py`). Later builds are incremental (`build/pip-<tag>`), and no
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
docker build -f docker/linux/Dockerfile -t viproc:linux .
docker run --rm viproc:linux                                # all test suites
docker run --rm -v "$PWD:/work" viproc:linux python prog.py # run a program with viproc
```
