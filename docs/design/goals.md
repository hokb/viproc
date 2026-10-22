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

## G3 — Never slower than NumPy

**viproc may never be slower than NumPy (below 0.95x of NumPy's speed) for
any expression.** This is a hard rule: it holds for every expression,
program, data size, shape and dtype, not only on average. A measurement
below 0.95x is a defect, not an accepted cost of the asynchronous model.

**Soft rule, no sufficient parallel potential: no slow-down beyond 3%.**
Where a measurement has no sufficient parallel potential (for example a simple expression on
small data), no speed-up is expected, but viproc must not slow it down
significantly either: the design target is at most 3% slower than NumPy.
This is a soft design rule (a target that designs are judged by, within
measurement noise); the 0.95x limit above is the hard bound.

Example: `x = a[x + 1]` on a scalar `x` has nothing to overlap and is not
expected to be faster than NumPy; it must run at NumPy's speed. The same
expression in a loop may or may not run faster than NumPy, depending on
whether consecutive iterations can overlap (instruction pipelining, see
"Roadmap").

Judging a change against G3: compare viproc with NumPy on the same machine,
same build and options (`docs/benchmarks.md`, "How to measure"); single
configurations vary by ±25–30% between runs on noisy machines, so repeat
before calling a result a violation or a fix.

## Non-goals

- **Weak scaling.** Bigger problems on more cores are not the yardstick;
  a fixed program getting faster is (G1).
- **Data parallelism inside a single array op** (splitting one op across
  threads). One instruction runs on one thread, as one NumPy loop call (or
  the same sequence of loop calls NumPy itself would make).
- **Global program analysis:** no dependency graph (DAG) of the program, no
  ahead-of-time (AOT) analysis. Every decision is local.
- **Changing what NumPy computes** to gain speed: no other summation order,
  no other BLAS threading, no approximations (G2).
- **Dedicated array types or scopes in user code** (temporary/return types,
  scope APIs as on .NET): they would rule out running unchanged NumPy
  programs (`docs/design/memory-management.md`, D1).
- **Python implementations without reference counting** (PyPy).

## Not goals yet

- **Any optimization besides array ILP:** own SIMD code, kernel fusion,
  JIT code generation, GPU offloading. SIMD comes from NumPy's loops. Do not
  add these, but do not make design decisions that rule them out (keep
  device and memory handling behind an interface).
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
