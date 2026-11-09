# OpenCL device kernels (optimization level O4)

Level O4 adds automatic device offloading (goals.md, "Optimization
levels"). This document covers its first part: `numpy.linalg`'s
factorizations, `numpy.fft` and matrix products on an OpenCL device. Results
are correct, not bit-identical: values within the rounding of the device's
computation (in NumPy's precision), and NumPy's dtypes, shapes, layouts,
errors and warnings.

## Library choice

| Need | Chosen | Alternatives and why not |
|------|--------|--------------------------|
| BLAS (matmul, and trsm/gemm/syrk/herk inside the factorizations) | **CLBlast** (Apache-2.0) | clBLAS: archived by AMD, last release 2017. ViennaCL: unmaintained since 2019, C++ templates around its own kernels. ArrayFire's OpenCL backend: a whole array library that itself builds on CLBlast and clFFT. |
| FFT | **VkFFT** (MIT, header-only) | clFFT: unmaintained since 2017 and limited to sizes of the radices 2, 3, 5, 7, 11, 13 (NumPy takes any n). VkFFT does any n (Bluestein for large primes), R2C/C2R, batches, double precision, and compiles into viproc. |
| LAPACK (LU, Cholesky) | **viproc's own OpenCL kernels**, CLBlast for the blocked updates | No maintained OpenCL LAPACK exists (clMAGMA's last release is from 2014). The LU family and Cholesky are small kernels (`cl_linalg_source.cpp`); large matrices use MAGMA's blocked scheme with CLBlast for the O(n^3) part, all on the device. |

CLBlast is actively maintained, tuned for many GPUs, has batched `gemm` in
all four precisions with row-major layouts and a C API, and is packaged
(Debian/Ubuntu `libclblast1`, conda-forge, vcpkg). VkFFT is maintained,
benchmarks at cuFFT's speed and pocketfft's accuracy (with its lookup
tables, `useLUT`), and its OpenCL backend needs nothing but the ICD loader.

**Not on the device:** `eig`, `eigvals`, `eigh`, `eigvalsh`, `svd`,
`svdvals`, `qr`, `lstsq` and what NumPy builds on them (`pinv`,
`matrix_rank`, `cond`, `norm(ord=2)`). They keep NumPy's LAPACK loops (as
tasks where viproc has them, see CLAUDE.md). Iterative eigen and SVD solvers
on OpenCL are a project of their own, and their results depend on the
algorithm beyond rounding (eigenvector signs and bases of degenerate
eigenvalues): a follow-up, not a drop-in.

## Loading

viproc links neither the ICD loader nor CLBlast (`cl_loader.cpp`):
`import viproc` must work on machines without OpenCL. On first use it opens
`libOpenCL.so.1` / `OpenCL.dll` / the OpenCL framework (env
`VIPROC_OPENCL_LIBRARY`) and `libclblast.so.1` / `clblast.dll` (env
`VIPROC_CLBLAST`). The `cl*` functions viproc's code and VkFFT call are
forwarders to the loaded library (hidden symbols); CLBlast's C API types
are mirrored, so building needs only the OpenCL headers and VkFFT's header
(taken from the system, else fetched by CMake: `VIPROC_OPENCL=OFF` builds
without). Without the loader no device is selected; without CLBlast,
matmul stays on the CPU and factorizations are limited to matrices the
unblocked kernels handle (order <= 256).

Objects that live for the process (the selected context, the FFT plans) are
never destroyed: releasing OpenCL objects from static destructors at exit
races with the implementation's own teardown (pocl crashed).

## Device selection

`viproc.init(level=4, device=spec)`, env `VIPROC_OPENCL_DEVICE`; spec
`"gpu"` (default: the first GPU or accelerator), `"cpu"`, `"any"`,
`"none"`, `"P:D"` (platform and device index), `"N"` (index over all
devices) or part of the device's name. Selecting a device configures the
policy (below) and measures the device's latency and bandwidths. **CPU devices only on request:**
an OpenCL CPU device (pocl, Intel's CPU runtime) computes on the cores
viproc's workers use; it is for tests and machines whose CPU runtime is
faster than NumPy's loops. `viproc.device_info()` names the device, its
fp64 support and whether CLBlast was found. Levels 1-3 never use a device.

## Execution

- **Where the decision is made.** `vp_gufunc` and `vp_matmul` make the CPU
  kernel as before (NumPy's loop) and, at O4 with a device, wrap it in a
  device kernel (`cl_kernels.cpp`) when the op has a device implementation
  for its dtypes (`supports`). The device kernel asks the device policy
  (`cl_policy.cpp`, below) in the task (the stage *device selected*), with
  the op's actual sizes and the state of its operands. A local decision per
  instruction, no global analysis; deferred (unsized, O2) matmuls are
  covered too, since sizes are known when the task runs.
- **Data stays where it was computed** (below, "Device-resident arrays"):
  a device op's result keeps its values on the device; the next device op
  reads them there, the host downloads them when it reads them.
- **Threads.** Each thread running device work has its own in-order queue;
  calls block until their results are read back, so a device op occupies
  its worker like a CPU op. Kernel objects are made per call; an FFT plan
  runs one transform at a time (VkFFT sets its kernels' arguments).
- **Failures of the device** (out of memory, larger than its largest
  allocation, a build or launch error) run the CPU kernel instead:
  `offload_stats()["device_fallbacks"]` (and are remembered, above).
  `offload_stats()["device"]` counts the ops computed on the device.

## Device or CPU

`DevicePolicy` (`cl_policy.hpp`), one per process, decides per op. Two
modes:

- **Threshold**: the device for work >= `min_work` (the op's work estimate:
  multiply-adds of a product, n^3 of a factorization, 4 n log2 n of a
  transform). Taken when the user sets one (`viproc.init(device_min_work=
  ...)`, env `VIPROC_OPENCL_MIN_WORK`) and in deterministic mode
  (`adaptive=False`: where an op runs must not depend on timings), there
  with 2^24 unless set.
- **Cost model** (default): the op runs where its estimated time is
  smaller.
  - Device: a fixed cost per op (the device's round trip), the upload of
    the operands not on the device yet, the download of the results unless
    they stay on the device, the compute time, and the wait for the device
    ops in flight. The device shares its time between them, so each delays
    the op by at most its own remaining compute time and by at most the
    op's own (a fair share): wait = min(backlog, ops in flight x the op's
    compute time), the backlog being the summed estimated compute times of
    the ops in flight (`DevicePolicy::begin`/`end`, atomics). A large
    matmul in flight delays a small op by about the small op's own time; a
    small op in flight delays a large one by its own time only. (First
    version: compute time x (1 + ops in flight), whatever their sizes; a
    tiny `det` in flight doubled the estimate of a large matmul.)
  - CPU: the download of operands whose values are only on the device, and
    the compute time of NumPy's loop on one core.
  - Latency and bandwidths are measured when the device is selected (a
    64-byte round trip, a 16 MB transfer each way, best of a few; a few
    ms). Compute rates start from a prior (device: compute units x clock x
    lanes x a share per op class; CPU: one core) and then follow the run
    times measured per op, dtype and size class (log2 of the work), as a
    decaying average (the device's normalized to an idle device: the
    measured time less the wait estimated at the start, at least the fair
    share of it). `tests/test_cl_policy.cpp` checks the model without a
    device.
  - Every 8th decision of a class runs the other side if its estimate is
    within 4x (an exploration, so that both sides stay measured); the
    period doubles up to 1024 while the choice holds and restarts when it
    flips. Ops whose work at the CPU's fastest rate costs less than the
    device's round trip go to the CPU without a lock (most small ops).
  - Not in the model: the load of viproc's workers. A device op blocks its
    worker until its results are ready (in-order queues, blocking reads),
    so it occupies a core's thread like a CPU op; idle cores elsewhere do
    not make either side cheaper. That changes with asynchronous queues
    (open).
- **Failure memory**, in both modes: when the device fails on an op (out
  of memory, larger than its largest allocation, a build or launch error),
  the smallest failing size (bytes moved) is kept per op and dtype; ops of
  that size or larger go to the CPU at once (`policy["skipped"]`), except
  every 16th, doubling up to every 1024th while it fails again (the failure
  may have been transient: memory held by other ops). A success at that
  size clears it.

`viproc.device_info()["policy"]` shows the mode, the measured profile and
the decision counts.

## Device-resident arrays

The data model has foreseen it since phase 1 (array-data-model.md, A1: one
handle per device in a `CountableArray`); O4 fills it in:

- **The copy.** A `CountableArray` may hold a `DeviceCopy` beside its host
  memory: the values on the device, a byte-for-byte mirror of the host
  block (same size and layout), so every view of the data version
  addresses the same bytes on both. The runtime knows only the interface
  (`download`); the OpenCL kernels make `ClCopy`s (a device buffer and its
  context).
- **Who makes it.** A device op whose output covers its whole buffer (a new
  result, or an `out=` array written whole) and has no live NumPy view
  keeps the result on the device: the buffer it computed into becomes the
  mirror if it is laid out as the output's block, else a device-side copy
  (`copy_nd`, viproc's kernel) puts it into one. The host block is
  allocated as before but stale (`host_stale()`). Partial outputs (`out=`
  a view) and det/slogdet (computed on the host) are downloaded as before.
  The op finishes its queue (`clFinish`) before the task completes, so
  readers on other threads' queues see the values.
- **Device readers.** A device op whose operand has a copy on its device
  packs the operand there (the same strided copy kernel: any view,
  transposed, strided, reversed, broadcast; at most 8 axes after dropping
  extent-1 ones, offsets and strides in whole words; else the host path)
  instead of gathering and uploading it.
- **Host readers.** `CountableArray::data()` downloads the values on first
  use when the host's are stale (any thread, once, under a lock striped by
  address) and keeps the device copy: CPU kernels, exports, scalars and
  copies read through it, so no reader can see stale memory. One atomic
  load per call when nothing is on the device. A download that fails (the
  device was lost) throws; inside a task the op fails with the message.
  `raw_data()` is the address without a download (allocation checks, the
  device kernels' own offsets).
- **Writers.** Before a task or the direct path writes an output in place,
  the output's host values are made valid and its device copy is dropped
  (`drop_device_copy`); so are writable exports (`np.asarray`, borrowed
  views) and the eager fallback's `prepare_write`. Copy-on-write and
  temporary elision read through `data()` / download before moving a
  block.
- **Memory.** A device copy lives as long as its data version (released
  with the `CountableArray`'s last reference, on any thread). There is no
  eviction yet: when the device runs out of memory, ops fail over to the
  CPU (failure memory above).
- `VIPROC_OPENCL_RESIDENT=0` (at device selection) or
  `_viproc.opencl_resident(False)` downloads every result at once (#64's
  behaviour).

## Semantics

- **Precision is NumPy's:** linalg computes in double or complex double
  (`_commonType`, the input cast first, the result cast back, as NumPy);
  fft in its loop's dtype (double for `norm="backward"` also for single
  precision data, as NumPy picks that loop); matmul in the operands' dtype.
  Double precision needs `cl_khr_fp64`; without it those ops stay on the
  CPU.
- **LU** (`getf2`, `laswp`, `getrs`; det, slogdet, inv, solve): partial
  pivoting with LAPACK's rule (the largest |re| + |im|, the first of equal
  ones), so the pivots are LAPACK's for all but ties broken by rounding.
  Matrices of order <= 64: one work-group per matrix, the whole batch in one
  launch. Larger: panels of 64 columns, `laswp`, CLBlast `trsm` and `gemm`.
  `det`/`slogdet` take NumPy's formula on the factored diagonal (sign from
  the pivots, the sum of the logs, `sign * exp(logdet)`), computed on the
  host from the downloaded factors.
- **Cholesky** (`potf2`, CLBlast `syrk`/`herk` for large matrices) reads
  the lower triangle (the diagonal's real part), as LAPACK's `potrf('L')`;
  `upper=True` factors `A^H` and returns `L^H`, NumPy's `potrf('U')`. The
  strict other triangle is zeroed, as NumPy does.
- **FFT**: pocketfft's conventions: the input truncated or zero-padded to
  n per row, irfft ignoring the imaginary parts of the zero and the
  Nyquist frequency, transforms unnormalized and multiplied by NumPy's `fct`
  afterwards (on the device; `fct` <= 1, nothing to overflow). R2C/C2R use VkFFT's in-place layout (rows of n/2 + 1 complex
  values). n-d transforms are NumPy's sequence of 1-d ones (`_fft.py`).
- **Errors** are NumPy's: a zero pivot fails `inv`/`solve` with
  "LinAlgError: Singular matrix" (an exact zero, as LAPACK's `info`), a
  non-positive pivot fails `cholesky` with "Matrix is not positive
  definite"; `det`/`slogdet` of a singular matrix give `0` and
  `(0, -inf)`. Raised asynchronously (`AsyncLinAlgError`), as on the CPU.
- **FP warnings**: the FP flags OpenCL, CLBlast and VkFFT's host code raise
  (plans, downloads inside CPU kernels) are cleared or restored; only the
  host math NumPy's loops do too (det's `exp`/`log`) can report one.

## Functions

On the device at O4 (where the policy sends them): `det`, `slogdet`, `inv`,
`solve`, `cholesky` (both triangles), `matmul`/`@`/`np.dot` of float and
complex operands (also with `out=` and the batch broadcasting of the gufunc
path), `fft`, `ifft`, `rfft`, `irfft`, `hfft`, `ihfft` and the n-d
transforms NumPy builds of them. Through them, the compositions this change
ports from NumPy's code onto viproc's ops (at every level, bit-identical at
O1): `matrix_power`, `multi_dot`, `tensordot` (`np.` and `np.linalg.`),
`tensorsolve`, `tensorinv`, `linalg.matmul`, `linalg.outer`; also
`linalg.vecdot`, `linalg.diagonal`, `linalg.trace`,
`linalg.matrix_transpose`, `matrix_norm`, `fft.fftshift`, `fft.ifftshift`
no longer take the eager fallback.

## Testing

`python/tests/test_opencl.py` runs every device op against NumPy (values
within the precision's rounding; dtypes, shapes, layouts, warnings and
errors exactly) on the device `VIPROC_TEST_OPENCL_DEVICE` names (default
`"any"`), with `device_min_work=0`, and checks the device counter. Without
a device the module is skipped. Device-resident results: chains of device
ops, CPU ops, views (transposed, strided, reversed, broadcast) and
transforms reading them, writes, `out=`, exports (`device_state`: where an
array's values are). The cost model with fixed profiles
(`opencl_set_profile`: a fast and a slow device; the same op sent to the
CPU with host operands and to the device with operands there) and the
failure memory (`opencl_limit_allocation`). CI installs pocl's CPU device and CLBlast
on Linux (`pocl-opencl-icd`, `libclblast1`). pocl compiles kernels on first
use (CLBlast's take seconds each) and caches them (`~/.cache/pocl`).

## Assumptions (taken without review)

- O4 is opt-in and process-wide, like the other levels; levels 1-3 are
  unchanged apart from the new compositions, which are bit-identical.
- Default device: GPUs and accelerators only; CPU OpenCL devices only by
  name or `"cpu"`/`"any"`.
- The cost model's priors (device: 64 lanes per GPU compute unit, fp64 at
  1/16, 40% of the peak for gemm, 5% for factorizations and FFTs; CPU: one
  core, 8-16 multiply-adds per ns for gemm) are rough; they decide only
  until a size class has measured runs. Not tuned on a GPU (none here).
- Deterministic mode keeps a threshold (2^24, a first guess for a discrete
  GPU with a PCIe transfer per op).
- Linalg on the device needs fp64 because NumPy computes it in double;
  computing single precision inputs in single precision would deviate by
  more than rounding.
- Verified on Linux with pocl 5.0 (CPU device), CLBlast 1.6.2, VkFFT 1.3.4;
  Windows and macOS builds compile the same code, unverified on a device.

## Open

- Asynchronous queues (events instead of blocking reads): a device op
  would free its worker while the device computes, and the workers' load
  would enter the cost model.
- Element-wise ufuncs and reductions on the device, so chains between the
  device ops stay there too (today an element-wise op on a device result
  downloads it).
- Eviction of device copies under memory pressure (today the device fails
  and the ops fall back to the CPU).
- eig/eigh/svd/qr/lstsq on the device.
