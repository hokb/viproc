# viproc optimizations

Every optimization of viproc with the case it covers, how it is
implemented, its state and its measured effect. All implemented
optimizations keep results bit-identical to NumPy (optimization level O1,
`docs/design/goals.md`). Keep this list up to date with every change that
affects performance; the measurements behind it live in
`docs/benchmarks.md`.

**Target mode:** Python starts with `python -m viproc` or `VIPROC=1`. In
user code `np` is then the viproc module; every function creates and
returns viproc arrays. There is no switch to ndarrays and no ndarray hook.
The hook stays shelved until it is investigated for Python paths
specifically. `shape_map.py --module` measures this mode.

**Measurements:** Linux VM, 4 cores, 4 workers, Release build, NumPy
2.5.3. Numbers from `docs/benchmarks.md`, `shape_map.py`, `break_even.py`
and callgrind. Speedup = NumPy time / viproc time. Runs on different VMs
(2.1 and 2.8 GHz) do not compare directly.

**State:** Done = in `main`. Prototype = built, not the
default. Open = analyzed, not built. Rejected = measured and decided
against.

## Current state (2026-10-28)

In the target mode viproc reaches parity at ~1,000 elements and 1.6x at
1 million elements; below that the median is 0.80-0.90x. Quick shape map
`--module`, 34 cases, 566 measurements, all bit-identical, build
`#aec431ca` (code of `d4dc513`).

| elements per array | median | interquartile range |
| --- | --- | --- |
| 0-d | 0.91x | 0.75-1.04x |
| 1 | 0.88x | 0.77-1.11x |
| 10 | 0.80-0.84x | 0.68-1.05x |
| 100 | 0.85x | 0.62-1.00x |
| 1,000 | 0.95-1.01x | 0.75-1.20x |
| 10,000 | 1.11-1.12x | 0.78-1.80x |
| 100,000 | 1.47-1.55x | 1.07-2.42x |
| 1,000,000 | 1.61x | 1.12-2.30x |

Best single case 9.4x (`ind8_cheap`, full map), worst 0.37x
(`cond_data`, 1000 x 10). Three classes of outliers stay below 0.95x:
a sync every step (`cond_data`, `converge`, `ind3_sync`: 0.37-0.45x, also
on large data), Python loops over rows and columns (`loop_*`: 0.42-0.9x at
every size) and the per-op overhead on small data (0.8-0.9x).

## Runtime and scheduling

The runtime finds the parallelism and keeps the cost per task small; the
largest lever was the continuation, the latest the spinner hand-off.

| optimization | case | implementation | state | effect |
| --- | --- | --- | --- | --- |
| Continuation (trampoline) | Dependent chains | The thread that completes the last input runs the consumer next, not nested | Done (phase 1) | One hand-off per chain instead of per op; `strong_scaling` 8 chains: 1e7 7.5x, 1e6 4.5x, 1e4 ~3x |
| Lock-free consumer list | Every task with a pending input | Stack in the consumer tasks, push by CAS, closed by one exchange (`kClosed`) | Done (#49) | Main thread ~50, worker ~180 instructions less per op |
| Back-pressure with hysteresis | Main thread far ahead | At most 1,000 active tasks; the main thread resumes after a quarter of them completed | Done (`2f7cac1`) | No wake-up and context switch per op at the limit |
| Adaptive dispatch policy | Result read right away | Per (issue site, log2 size): inline from 3/4 read right away, explorations stay, reversible | Done (`b938697`) | `sync_latency` 8 elements: 8.1 → 1.4-2.4 µs; 1e4 ≈ NumPy |
| One spinning worker | Stream of small tasks | One idle worker spins 50 µs, `enqueue` wakes a sleeper only without a spinner | Done | No wake-up syscall per small task |
| Spin at sync points | Result is read | The main thread spins up to 50 µs before it sleeps | Done (#45) | Single synced op of 1e4: 18.8 → 6.8 µs (NumPy 3.0) |
| Local queue for small tasks | One completion unblocks several tasks | Tasks whose largest operand has < 65,536 elements stay with the worker; a spinner may steal them | Done (`47670fe`) | `expr_where` 1.3-1.8x → 2.1-4.1x; trade-off `sum_cores` 2.6-2.9x → 1.6-1.7x; geomean 1.02 |
| Wake-ups bounded by the queue | More workers than the issue rate feeds (`array_pipelining.py` loop, 11 workers on a 12700K slower than 4) | `enqueue` notifies only while fewer wake-ups are pending than tasks queued; a woken worker that leaves tasks queued wakes the next (as .NET's ThreadPool re-requests a thread) | Done | 1,500² loop, 8 workers on 4 vCPUs: notifies by `enqueue` from ~2 of 3 queued tasks (104,630 of 153,057, whole benchmark) to 2-114 per run of 1,500 iterations; times on the VM within noise |
| Spinner hand-off while tasks stream in | Workers faster than the main thread | A spinner that finds work hands the spinning to a sleeper while `enqueue` had to wake one | Done (`d4dc513`) | 4 chains of cheap ops: 3.6 → 1.2 µs per op; break-even of cheap ops ~7,000 → ~3,000 elements |
| Workers = physical cores | SMT machines | `physical_core_count()`, SMT siblings once | Done | i7-12700K: 12 instead of 20 workers |
| Workers = physical cores - 1 | Fine-grained loops, the main thread issuing | `default_worker_count()`: one core stays with the main thread | Done | `array_pipelining.py` 1,500 x 1,500 loop on 4 cores: 2.74x with 3 workers, 2.30x with 4 (the main thread's issue rate fell with every worker added: 4.2 µs per iteration with 1 worker, 5.3 with 4). Trade-off: data-parallel programs lose a core (`vectorized` 4.29x → 3.14x) |
| GIL released while blocked | An issue waits on back-pressure | The GIL is released while an issue blocks | Done (`1ecc18f`) | Correctness, no measurement of its own |

## Memory

The memory optimizations make large data faster than NumPy even without
parallelism; D11, element memory taken by the task, gained most.
Decisions in detail: `docs/design/memory-management.md`.

| optimization | case | implementation | state | effect |
| --- | --- | --- | --- | --- |
| Element pool (D3) | Every op that needs memory | Process-wide pool, quantized size bins, huge pages where allowed | Done (`2c67ed2`) | `((x + 1) * 2) + 3`, 1e7, 1 worker: 117 → 40 ms |
| Temporary elision (D5) | Operator on a temporary | The result writes into the dying operand's buffer | Done (`36e555b`) | Same chain 40 → 17 ms (NumPy 60 ms); `strong_scaling` 1e7 7.5x |
| Memory taken by the task (D11) | Dependent chains, many ops in flight | The producing task takes its block right before the kernel, from a dying input or the pool | Done | Shape map median 0.85 → 1.06x; parity from 10,000-30,000 instead of 100,000 elements; dependent chains 0.70 → 1.05x |
| Shared LIFO per size bin | Blocks moving between threads | One LIFO per bin under a small lock, no per-thread caches | Done | Equal within noise; pool misses halved (138 → 66) |
| Pools for runtime objects (D4) | Every op | Task, CountableArray, Metadata, ArrayBlock from per-thread pools | Done (`bca06df`) | `x * y` (8 elements): ~6,600 → ~5,400 instructions, ~0.65 µs |
| Array and AliasGroup in one block | Every op | One pool block, intrusive non-atomic reference counts | Done | ~7,100 → ~6,600 instructions per op |
| Copy-free export (D7) | `np.asarray(x)`, handing data to NumPy | The export is a writable alias; while it lives, ops on that memory run synchronously | Done (`e0732f4`, `e29895c`) | `np.asarray(x)` of 1e7 float64: 18.8 → 0.001 ms |
| Copy-on-write in the writing task (D8) | A write while readers are pending | The task copies right before its kernel, or takes the block over when nobody reads it any more | Done | `loop_scan` from 100,000 elements: 0.02-0.22x → 0.65-1.06x |

## Per-op overhead and bridge

The main thread's cost per op fell from ~35 µs to ~0.55 µs; in the target
mode a small ufunc call costs ~10% more instructions than NumPy's
(`np.abs(x)`, 100 elements: 5,468 vs. 4,990).

| optimization | case | implementation | state | effect |
| --- | --- | --- | --- | --- |
| Hot path in C | Every op | `_viproc._Array`: operators, `__array_ufunc__`, dtype resolution, broadcasting in C instead of Python | Done (`f443b29`) | ~35 µs → ~3-7 µs per op |
| Issue site keyed by code address | Every op | Cache per (code address, bytecode offset), never a dict keyed by code objects | Done (`ad55be7`) | Hashing was ~20% of the per-op cost |
| C lookup tables | Every op | Open-addressing tables keyed by (pointer, int) instead of dicts with tuple keys | Done (`e2f5eb2`) | ~10,500 → ~7,100 instructions; ~1.1 → ~0.8 µs |
| Direct path | Small ready ops (< `sync_below` = 2048) | The kernel runs in the issue, the result is ready, no task | Done (`bc07d90`) | `x * y` (8 elements) 0.63 µs (NumPy 0.35); before ~1 µs of task protocol per op |
| Scalar shortcut | Ops on scalars only | NumPy's scalar math directly, NumPy's scalar as the result | Done | 0-d at parity (0.98x) |
| Cached scalar operands, inline task lists | `x * 0.5` in loops | Table of 64 ready 0-d arrays; producers/consumers inline | Done (#47) | Main thread 7,750 → 5,100 instructions; issue 1.21 → 0.74 µs |
| Reductions and methods in C, NumPy delegate | `x.sum()`, `x.copy()`, `x.astype()` on small data | C methods; NumPy computes small ready data on a borrowed view | Done | `x.sum()` 2.57 → 0.69-0.76 µs (NumPy 0.54); suite quick 69 → 73 |
| Views without ndarrays, `a[key]` in the C slot | Slicing, `.T`, `reshape` | Layout computed after NumPy's rules, basic keys in C | Done (`5db8972`) | `a[1:3]` 1.45 → 0.12 µs (NumPy 0.10) |
| Index assignment in C | `out[i] = s` in loops | `mp_ass_subscript`: small and ready in NumPy's slot, else an assign task | Done | `out[3] = s` 3.1 → 0.22 µs (NumPy 0.16) |
| `np.where` in C | `where` on small data | Select task, or NumPy on borrowed views | Done | 30,300 → 12,000 instructions (NumPy 10,100) |
| Ufunc proxies in the viproc module | `np.abs(x)` with `np` = viproc | The call goes straight to `fast_ufunc`, without NumPy's override dispatch | Done | `abs` 7,900 → 5,400 instructions (NumPy 4,500) |
| Reduction proxies in the viproc module | `np.sum(x, axis=0)` with `np` = viproc, Python loops over rows/columns | `viproc.sum`, `prod`, `max`, `min`, `amax`, `amin`, `any`, `all`, `mean`, `var`, `std`, `argmax`, `argmin` take `Array_array_function`'s steps without NumPy's `__array_function__` dispatch | Done | `np.sum(x, axis=0)` 15,100 → 8,800 instructions, `np.sum(x)` 22,800 → 16,600 |
| Expensive loops below `sync_below` as tasks | Ready ops of a few hundred to 2,048 elements with costly loops (`sin`, `power`, ...), loops over columns | The direct path times the shared kernel of each loop (`DirectCost`, ns per element); an op estimated above 10 µs of NumPy time takes the task path and the adaptive policy (adaptive mode only; every 32nd re-timed by its task on the worker, not on the main thread). Lower budgets lost where cheap consumers follow: 2 µs `ind_cores_f32` at 1,000 elements 0.6x → 0.15x, 5 µs `expr8` at 1,000 elements 1.0x → 0.6x | Done | `array_pipelining.py` loop over 1,500 columns 0.88x → 1.7-2.0x on 4 cores; `break_even.py` `chains_sin` at 2,000 elements 1.48 → 1.79; quick shape map `--module` (with the reduction proxies) 0.943 → 0.984, loops 0.76 → 0.85 |
| Direct-path budget per machine | The same on faster cores: a 12700K runs a column's `sin` (1,500 doubles) in 6.1 µs, below the fixed 10 µs | The bridge sets `direct_budget_ns` to 9 times a direct `a * b` of 8 doubles, timed once per process (`_scale_budget`, ~0.5 ms; VM 4.7 µs, 12700K 3.0 µs; `VIPROC_DIRECT_BUDGET_NS` overrides). Rejected: a C probe (pointer walks) rated the 12700K as fast as the VM; the VM's 10 µs scaled by the `a * b` ratio (6.2 µs there) left `sin` at the edge | Done | 12700K, 4 workers: `array_pipelining.py` `loop` 1,000² 1.13x → 2.9x, 1,500² 1.10x → 4.2x; quick module shape map 1.08 → 1.10, `expr8` at 1,000 elements 1.0x → 1.6x; VM unchanged within noise |
| FFT plans | Repeated `fft` of one shape | Plan once per (dtype, shape, strides, n, axis, norm) | Done (`75715d2`) | Issue ~70-100 → ~30 µs; ILNumerics FFFT n = 100: 0.65x → 2.7x |
| Reduction and gufunc plans | Repeated calls of one signature | `_reduce_plans`, `_gufunc_plans`: one dict lookup per call | Done | No measurement of its own |

## Kernels and op coverage

Every op that runs asynchronously instead of in the fallback is an
optimization: a fallback syncs and computes serially. All kernels call
NumPy's own loops the way NumPy calls them.

| optimization | case | implementation | state | effect |
| --- | --- | --- | --- | --- |
| Asynchronous reductions | `sum`, `max`, `any`, ... | NumPy's reduction loop in NumPy's blocks (`ReducePlan`), else NumPy's iterator | Done (`e1e3857`) | Suite 45 → 52, reductions category 31 → 75 |
| Exact NumPy iteration, casts, views, `mean`/`var`/`std`, `argmin`/`argmax`, advanced indexing | Broad op coverage | Iterator templates as `execute_ufunc_loop` builds them, gather/scatter tasks | Done (`1d42f0f`..`bc07d90`) | Suite 52 → 69 |
| Iterator copy taken by the task | Strided, cast, broadcast operands | Template per signature; the task takes its copy when it runs and gives it back | Done (`2d67864`) | `bool * float64`, 1e4, chained: 30-40 → 5 µs per op (NumPy 6-10) |
| Reduction templates once per signature | Reductions with casts or strides | Template over a dummy address, `NpyIter_ResetBasePointers` on the worker | Done | Side 100: 39-59 → 20-39 µs per step; 8 x 8: 7.7 → 4.3 µs per op |
| Branch-free `np.where` | Random condition | Masked blend per word instead of a branch per element | Done (`0c99e19`) | 1e4 elements: 5 µs (NumPy 40 µs) |
| `matmul`, `@`, `np.dot` | Matrix products | NumPy's matmul gufunc loop per batch row | Done (`233fe84`) | `matmul_batch`: 11,540 fallbacks → 0; small 0.48 → 1.04x |
| Linalg serialized from 64 x 64 | Many concurrent LAPACK calls | Process-wide mutex (`VP_GUFUNC_SERIAL`) | Done | 8 concurrent `inv` of 300 x 300: 20 ms-1.1 s → 17-47 ms |
| Linalg, FFT, sort asynchronously | `numpy.linalg`, `numpy.fft`, `np.sort` | NumPy's wrappers ported onto `vp_gufunc` and `vp_sort` | Done (#34-#37) | `np.sort` of 10 elements 0.76 µs (NumPy 0.79); FFT see plans |
| Joining, compositions, indexing functions, accumulation, ufunc methods | `concatenate`, `clip`, `take`, `cumsum`, `where=`, ... | One placement task per join; NumPy's code ported without the syncing `asanyarray` | Done (#27-#33) | Fewer fallbacks; no overall measurement |

## Shelved or rejected

The ndarray hook and the ndarray/viproc switch are shelved; the other
ideas were measured and rejected.

| optimization | case | implementation | state | effect / reason |
| --- | --- | --- | --- | --- |
| ndarray operator hook | ndarray inputs, programs that do not create viproc arrays | `numpy.ndarray` operators routed to viproc (`enable_hook()`) | Prototype, shelved (branch `claude/project-thread-vf3lz1`) | Suite: arrays 85, module + hook 62, hook 57; small data better, medium and large worse (`stencil_1d` medium 6.62 → 0.56x, `kmeans_points` large 6.56 → 0.60x) |
| Small results as ndarrays (switch) | Small data | `VIPROC_HOST_SMALL=1`: small results stay NumPy's | Prototype, shelved (branch `claude/project-thread-vf3lz1`) | A single op does not know how its result is used later (k-means trace); copies and syncs at the boundary |
| `sync_below` at 200 or 500 | Chains on small data | Lower the threshold in elements | Measured, not adopted (`942b5a1`) | Expensive ops up to 2x faster, cheap ones 3x and results read right away up to 6x slower |
| Spinner hand-off on every pickup | Task stream | Hand off unconditionally | Rejected | Sync-heavy loops -12% (`converge`, `cond_newton`) |
| Wake sleepers for local tasks | Local queue | Wake-up on every local push | Rejected | Undid the `expr_where` gain, `sum_cores` stayed |
| Main thread runs a queued task itself | Sync on a task not started yet | Take it from the queue at the sync point | Rejected | No sync found its task queued (1,278 continuations vs. 336 pops) |
| Memory at issue on the main thread (D10) | Every op | Allocation at issue | Replaced by D11 | Dependent chains 2-4x slower than NumPy |
| One BLAS thread per worker | Concurrent matrix products | `OPENBLAS_NUM_THREADS=1` | Rejected | Changes the results (O1) |
| Pre-scan of integer exponents | `power` with a negative exponent | Pass over the data before the kernel | Rejected | 3-10% slower; a watched call instead (E6) |
| Dense copy of strided ndarrays | Strided NumPy views as inputs | Import by copy | Rejected | Other strides may round differently (O1) and cost a pass on the main thread |
| Caching `ndarray` objects | Every op | Reuse of the Python objects | Rejected | Only ~1% of the instructions per op |

## Open and next

The per-op overhead on small data gains most next: it affects all small
cases and the Python loops at once.

| optimization | case | implementation (idea) | state | expected effect |
| --- | --- | --- | --- | --- |
| Remaining overhead of the direct path | Small data, loops over rows | ~500-800 instructions over NumPy: operand vector, buffer refcounts, release of the result; `x.sum()` +31%, `np.sum(x)` +17% | Open | Small-data median 0.8-0.9x → parity |
| Measured, cost-based switch above `sync_below` | Ops just above `sync_below`, sync loops | The policy learns per issue site which path was faster (below `sync_below` the loop cost decides already: expensive loops as tasks) | Open | `ind_cores_f32`, `cond_data`, `ind3_sync` at 3,000-10,000 elements: 0.23-0.45x → ≥ 1x |
| Main-thread issue cost of fine-grained loops | `array_pipelining.py`: view, `sin`, `abs`, `sum`, assignment per column | Done: re-tests of expensive loops timed by their task on the worker (−5,300 instructions per iteration), one shared reduce kernel per signature instead of plan, identity conversion (an ndarray) and kernel per op (−3,000 to −3,800), initial-exec TLS for the pools (~−100 per op), `C[i] = v` without a Python view object (−700 to −1,100): 26,200 → 16,700 instructions, issue 4.7-5.1 → 3.2-3.6 µs per iteration (4 vCPUs). Open, in order: a plan cache per issue site (dtype resolution, kernel, layouts, decision; guarded by dtype, shape and strides; ~−25 to −30%), locality of the issue path (the main thread is ~1.6x slower per instruction than NumPy's: 117 vs. 26 L1 misses per iteration, lines shared with the workers; measure with `perf c2c`), then several ops per issue (chains as one task: against "one task per op"; a loop-replaying interpreter hook: research) | Partly done | Issue 1.45x faster; the loop at 1,500 x 1,500 is now bound by the workers on 4 cores |
| A sync every step | `cond_data`, `converge`, `ind3_sync` | Break down where the time between issue and sync goes | Open | Large data too at 0.37-0.45x today |
| Python loops over rows | `loop_accum`, `loop_cols*`, `loop_rows`, `loop_scan` | Follows from the direct-path overhead; `loop_scan` also copy-on-write per row (D8) | Open | Below 0.95x at every size, also 1000 x 1000 |
| Hook for Python paths | Python parts, bypassing NumPy parts | The hook used selectively instead of as an input switch | Later | Not investigated yet |
| Scaling beyond 4 cores | i7-12700K, 12 cores | Find whether memory bandwidth, the issue rate or wake-ups limit it | Open | Suite 22 there (against ideal 20 cores, not comparable) |
| Stage pipelining (O2) | Pending sizes, early allocation | Six stages, tasks advance on stage events | Planned (`docs/design/pipelining.md`) | Fewer syncs for data-dependent sizes |
