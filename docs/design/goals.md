# viproc: goals and non-goals

What viproc is for, what it promises, and what it deliberately does not do.
This is the reference for design decisions and for judging changes; other
documents (`CLAUDE.md`, `README.md`, `docs/benchmarks.md`, the other design
documents) point here instead of repeating it. Change it only by decision
of the maintainer.

## G1 — Main goal (central claim)

**viproc automatically identifies the parallel potential within and
throughout large sections of regular, un-annotated NumPy programs and uses it
to feed parallel hardware efficiently.**

- The program stays a regular, sequentially written NumPy program
  (`c = a + b`, `m = max(c)`, `s = sum(c, axis=0)`, …). It is not rewritten
  for parallelism and carries no annotations; viproc is switched on from
  outside (`python -m viproc prog.py`, `VIPROC=1`) or by the import
  (`import viproc as np`).
- The parallelism is **instruction-level parallelism for array
  instructions (array ILP)**: independent array instructions run
  concurrently, dependent ones keep only the local order needed for correct
  results. It is found at runtime by local decisions only (is this argument
  `ready` or `pending`?), without a dependency graph or ahead-of-time
  analysis, by letting the main thread run far ahead of the computation
  (execution model after the ILNumerics Accelerator; see `CLAUDE.md`,
  "How parallelism is found").
- Success is measured as **strong scaling of whole array programs**: a
  fixed program with a fixed problem size gets faster with more cores,
  because independent array instructions overlap. Benchmarks compare whole
  programs against plain NumPy at 1, 2, 4, … worker threads
  (`python/benchmarks/suite.py`, `shape_map.py`; `docs/benchmarks.md`).

### Complexity becomes performance

The question is not whether single operations run as fast as, or faster
than, NumPy; it is whether **whole programs scale strongly**. A program of
one operation does not get faster with viproc: there is nothing to overlap
it with, and viproc does not split single operations (see "Non-goals").
Real programs compute several quantities, branch into independent
sub-expressions, mix reductions with element-wise work and loop over
arrays across functions and modules. In practice, **complexity always
comes with parallel potential**, and viproc turns that potential into
parallelism: **viproc turns complexity into performance.** This is the
story `README.md` and the other documents tell; benchmarks follow it by
measuring whole programs of growing complexity rather than single
operations in isolation.

Goals for programs (decided 2026-10-28, current milestone):

- **Every program runs at 0.8x of NumPy's speed or faster**, also where it
  offers little parallel potential (the per-op layer of G3 must stay
  small enough for that).
- **Complex programs run clearly faster than NumPy**, more so with more
  cores.
- **Parallel potential becomes parallelism:** wherever independent array
  instructions exist, they run concurrently.

The 0.95x / 3% target of G3 for single expressions stays the long-term
target; the 0.8x floor for whole programs is the goal that changes are
judged by now.

## R1–R3 — Design rules for scheduling (maintainer, 2026-11-12)

These rules hold for viproc and the ILNumerics Accelerator alike. Every
design decision is checked against them.

- **R1: The main thread runs ahead as far and as fast as possible.**
  Removing every barrier to that comes first. Every op must be able to run
  asynchronously, and issuing must be as cheap as possible. A sync point that
  the program does not ask for, such as a fallback that waits, a data-dependent
  shape resolved on the main thread or an issue-time check that reads data, is
  a defect to remove.
- **R2: Every op can run on every device, with costs that can be
  determined.** An op that only one device can run, such as a host-only
  BLAS/LAPACK call next to device-capable element-wise ops, breaks the
  scheduling: chains either stay on that device or pay a transfer at every
  such op. Each op therefore needs an implementation per device and a cost
  estimate per device.
- **R3: Costs are reliably predictable, at least after an adaptation
  phase.** A device decision is only as good as its cost estimates. Estimates
  may start from a model (FLOP, bytes, calibration) and are corrected by
  measurement until predictions and measured times agree.

**Where the device is decided.** The device of an op is chosen as late as
possible: when all its inputs have selected their devices (or are ready),
and by the task itself, on the thread that delivers that event, not by the
main thread at issue. A chain may switch devices at any op, also in the
middle, whenever the costs make that the better choice. Issue only creates
the task (R1): the main thread records the op and its dependencies; it
selects no device, computes nothing and waits for nothing. The criterion per
device d is ETA_d + kernel time + transfers; affinity to the device of the
inputs is a cost term (transfers), not a rule. Measurements correct the model
slowly; they do not replace it (R3).

Known violations (2026-11-12, to be removed or justified):
- The direct path and the scalar shortcut run small ready ops on the main
  thread (`sync_below`, `DirectCost`). They are deliberate (an op of a few
  elements costs less than its task), but they are a main-thread cost that R1
  weighs against.
- BLAS threads are set process-wide (OpenBLAS has no per-thread count in
  NumPy's wheels): the task decides per call (dynamic, "BLAS threads"
  below), but a call already running keeps its split. Without OpenBLAS,
  LAPACK calls from 64 x 64 run one at a time (`VP_GUFUNC_SERIAL`).
- O4 device kernels decide CPU or device in the task by size, not yet from
  per-device costs and an ETA per device (R2, R3).

In viproc today, O1 runs everything on the CPU workers. R2 and R3 apply to
the device kernels of O4 (`docs/design/opencl.md`, where the device kernel
already decides by size in the task) and to the per-op costs that the dispatch
policy uses (`DirectCost`, calibration).

## G2 — Results identical to NumPy

Results are always identical to plain sequential NumPy execution:
bit-identical values, NumPy's dtypes, shapes, layouts where they decide
results (e.g. the rounding of pairwise sums), errors and warnings. The
final computation of every instruction is delegated to NumPy's own inner
loops, including their SIMD paths, called exactly as NumPy calls them.
Anything viproc cannot run with NumPy's semantics falls back to eager NumPy.
Errors keep NumPy's type (as a subclass) and message; run-time errors of
asynchronous ops are raised in the main thread as soon as the op failed,
a little after its call returned, by design
(`error-handling.md`, E7).
Every benchmark case and every test must stay bit-identical.

G2 covers the whole interface, not only computed values: viproc is a
semantic drop-in for NumPy, without surprises. Every function, method and
attribute behaves as NumPy's (flags, `base`, identity, in-place effects,
warnings, errors). Where NumPy's own code cannot run on viproc memory
(`ndarray.resize`: NumPy reallocates memory it owns), viproc ports it;
a viproc-specific error or a different result is a defect.

The price of G2 is a small overhead on small computations: NumPy's own
loops and operators do the work, and viproc's layer (tasks, arrays,
dispatch) comes on top. **Planned:** bit-identity becomes a property of
the optimization level ("Optimization levels" below): it holds at O1,
today's only level; higher levels compute correct results that may
deviate from NumPy's, faster. Until they exist, G2 holds without
exception, and O1 stays available afterwards. G2 has priority over G3:
"never slower than NumPy" is the target of the bit-identical mode, not
its guarantee (G3).

## G3 — As fast as possible within G2

**G2 comes first.** Bit-identical results and "never slower than NumPy"
cannot always hold together: when NumPy's own loops do every computation,
viproc's layer (issue, task, arrays, dispatch) comes on top of NumPy's own
work, and an operation on small data without parallel potential has
nothing to hide that cost behind. viproc therefore does not guarantee
"never slower" in the bit-identical mode (today's only mode). Where the
two conflict, G2 wins; trading exactness for speed is left to the planned
optimization levels above O1 ("Optimization levels"), which may deviate
from NumPy's results.

Within G2, viproc must be as fast as it can be:

- **Rule, no avoidable cost:** viproc adds nothing that G2 does not need.
  No pass over the data (scan, check, copy) to catch a rare condition, no
  eager fallback for a supported op, the fewest main-thread instructions
  per op. Data-sized costs are a defect: on data large enough that the
  per-op layer does not count, a measurement below 0.95x of NumPy's speed
  is a defect, not an accepted cost of the asynchronous model.
- **Target, not a guarantee: never below 0.95x of NumPy for any
  expression**, every program, data size, shape and dtype; where a
  measurement has no sufficient parallel potential (for example a simple
  expression on small data), at most 3% slower than NumPy. Designs are
  judged by this target (within measurement noise), and the per-op
  overhead is worked down towards it, but never at the expense of G2.
  Small data misses it today (`docs/benchmarks.md`, shape map).
- **Speed-up where there is parallel potential** (G1): medium and large
  data with independent array instructions run faster than NumPy.
- **Programs: at least 0.8x of NumPy** (current milestone, G1, "Complexity
  becomes performance"): a whole program below 0.8x of NumPy is a defect
  to work on, whatever its data size.

Example: `x = a[x + 1]` on a scalar `x` has nothing to overlap and is not
expected to be faster than NumPy; the target is NumPy's speed. The same
expression in a loop may or may not run faster than NumPy, depending on
whether consecutive iterations can overlap (instruction pipelining, see
"Roadmap").

Judging a change against G3: compare viproc with NumPy on the same machine,
same build and options (`docs/benchmarks.md`, "How to measure"); single
configurations vary by ±25–30% between runs on noisy machines, so repeat
before calling a result a violation or a fix. A slow-down that comes from
the per-op layer on small data misses the target; one that a change could
avoid without giving up G2 is a defect.

## Optimization levels (planned)

viproc is to become configurable through one multi-stage optimization
setting. Each level adds optimizations to the one below; bit-identity is
a property of a level, not a fixed promise of viproc.

| Level | Adds | Results |
|-------|------|---------|
| **O1** (current implementation) | array ILP only: independent array instructions overlap; NumPy itself executes every function (its loops, operators and BLAS, called exactly as NumPy calls them); BLAS threads dynamic: OpenBLAS's own count for a call alone, one thread while other asynchronous work runs | bit-identical to NumPy (G2), for BLAS calls large enough for OpenBLAS to thread with the thread count the call ran with (as NumPy's own results depend on the machine's core count); `VIPROC_BLAS_THREADS=1` or the deterministic mode: NumPy's with `OPENBLAS_NUM_THREADS=1`, reproducible |
| **O2** | stage pipelining (`pipelining.md`: tasks advance through size, device, allocation and completion on their inputs' stage events; partial results start their consumers) | correct; may deviate from NumPy's |
| **O3** | micro kernels: viproc's own kernels where NumPy's calling pattern costs (fusion of element-wise chains, splitting single ops, other summation orders, re-implemented NumPy internals) | correct; may deviate in rounding |
| **O4** | automatic device offloading: cost-based device selection per array instruction (GPUs and other accelerators); started: linalg factorizations, fft and matmul on an OpenCL device (`opencl.md`) | correct; may deviate (device kernels and their rounding) |
| **O5** | reserved | |

**Configuration (implemented):** `viproc.init(level=n)`, env `VIPROC_OPT=n`
(or `On`), `python -m viproc -On prog.py` (also `-O n`, `--opt=n`); levels
1-5 are accepted, O1 is the default, 3 and 5 run as O2 until they are
built; 4 runs as O2 plus the OpenCL device kernels (`opencl.md`).
**BLAS threads at every level (maintainer's decision, 2026-11-12,
hokb/viproc#67):** dynamic, the same in viproc and the ILNumerics
Accelerator. Bit-exactness of BLAS results holds within one setup only:
NumPy's own results differ between machines with different core counts,
because OpenBLAS splits large products by its thread count. So the task
making a BLAS call decides its thread count (`src/kernels/blas_threads.*`,
`_blas.py`): a call alone (no other BLAS call in flight, no other active
task, none found for 5 ms) runs with the count found at init, as plain
NumPy; while other asynchronous work runs, every call runs on one thread
and viproc's workers parallelize across ops (concurrent threaded OpenBLAS
calls oversubscribe the cores and have a heavy tail). OpenBLAS's count is
process-wide: lowered at any time (a running call keeps its split), raised
only by the only call in flight; restored at shutdown. LAPACK calls are
not serialized. `VIPROC_BLAS_THREADS=1` and the deterministic mode
(`adaptive=False`) fix one thread for every call: results are NumPy's with
`OPENBLAS_NUM_THREADS=1`, whatever runs concurrently. (Before, one thread
for every call at every level: a single large product ran at 0.24-0.41x of
NumPy.) **O2 so far:** element-wise ufuncs (no casts, one output) on pending
or large inputs are issued without their size: the task computes it on
its inputs' size signals and its strides on their layouts
(`pipelining.md`, S2), so the main thread does not wait for pending
sizes, a broadcasting error is raised asynchronously, and strided
operands are computed per row (NumPy's iterator needs the GIL to plan;
values may round differently from NumPy's SIMD/scalar choice);
`zeros`/`ones`/`empty`/`full`/`arange` with a pending size issue a task
(S4) instead of a sync, and so do `x[mask]`, `flatnonzero` and 1-d
`nonzero` of pending data (their size is the data). The stage
signals exist at every level (they change no result). O2 is not strict: NumPy functions taken as a
whole may still run where nothing better exists; such ops give their three
stage signals in quick succession. Measured (4 vCPUs,
`python/benchmarks/opt_levels.py`, O1): 8 independent `inv`/`solve` of
64-1000 1.0-2.1x vs. NumPy with dynamic BLAS threads, as with one thread
throughout (before, NumPy's BLAS threads and LAPACK serialized from 64 x
64: 0.6-0.9x); a single product with nothing to overlap 0.84-0.95x from
300 x 300 on (one thread throughout: 0.28-0.33x).

"Correct" means a result NumPy's semantics allow: the same dtypes,
shapes, errors and warnings, values within the floating-point error of a
valid evaluation order. O1 stays available at every stage, as the
reference and for programs that need NumPy's exact bits.
G3 changes with the level: at O1 "never slower than NumPy" is a target
(G3); at higher levels viproc may compute differently, so the per-op
overhead on small data can be removed (own micro kernels, fused ops) and
the target can become a firm rule.

## Non-goals

- **Weak scaling.** Bigger problems on more cores are not the yardstick;
  a fixed program getting faster is (G1).
- **Data parallelism inside a single array op** (splitting one op across
  threads). One instruction runs on one thread, as one NumPy loop call (or
  the same sequence of loop calls NumPy itself would make).
- **Global program analysis:** no dependency graph (DAG) of the program, no
  ahead-of-time (AOT) analysis. Every decision is local.
- **Changing what NumPy computes** to gain speed: no other summation order,
  no approximations (G2; only the planned optimization levels above O1
  may deviate, see "Optimization levels"). Exception, decided: BLAS
  thread counts are chosen per call (see "BLAS threads" above).
- **Dedicated array types or scopes in user code** (temporary/return types,
  scope APIs as on .NET): they would rule out running unchanged NumPy
  programs (`docs/design/memory-management.md`, D1).
- **Python implementations without reference counting** (PyPy).

## Not goals yet

- **Any optimization besides array ILP (levels O2-O4):** stage pipelining,
  own SIMD code, kernel fusion, JIT code generation, GPU offloading. SIMD
  comes from NumPy's loops. Do not add these yet, but do not make design
  decisions that rule them out (keep device and memory handling behind an
  interface).
- **Phase 1 is CPUs only:** autonomous, asynchronous execution of array
  instructions on multicore CPUs, with NumPy's kernels taken from the
  installed NumPy at runtime. Out of scope for phase 1: .NET/ILNumerics
  bindings and distributed execution.

## Roadmap

- **Instruction pipelining** (known from the ILNumerics Accelerator):
  overlapping consecutive iterations of a loop, so that even a loop body
  without sufficient parallel potential of its own (`x = a[x + 1]` repeated) can run
  faster than NumPy.
- **Stage pipelining** (the full scheme of the ILNumerics Accelerator,
  `docs/design/pipelining.md`): the main thread only creates tasks and
  arrays; tasks advance their outputs through the stages (size completed,
  device selected, allocated, completed) on the stage events of their
  inputs. Sizes propagate before data, allocation and device decisions
  happen in tasks, and an unknown size forces a sync only where a branch
  depends on it. First step: creators with pending size arguments issue a
  task instead of syncing.
- Scaling beyond 4 cores, the remaining gufuncs (linalg, fft), per-op
  overhead on small data (G3): see `CLAUDE.md`, "Status and next steps"
  and "Open questions".
