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
devices) or part of the device's name. **CPU devices only on request:**
an OpenCL CPU device (pocl, Intel's CPU runtime) computes on the cores
viproc's workers use; it is for tests and machines whose CPU runtime is
faster than NumPy's loops. `viproc.device_info()` names the device, its
fp64 support and whether CLBlast was found. Levels 1-3 never use a device.

## Execution

- **Where the decision is made.** `vp_gufunc` and `vp_matmul` make the CPU
  kernel as before (NumPy's loop) and, at O4 with a device, wrap it in a
  device kernel (`cl_kernels.cpp`) when the op has a device implementation
  for its dtypes (`supports`). The device kernel decides when it runs, in
  the task (the stage *device selected*), from the op's actual sizes: a flop
  estimate below `device_min_work` (env `VIPROC_OPENCL_MIN_WORK`, default
  2^24, about a 256^3 product) runs the CPU kernel. A local decision per
  instruction, no global analysis; deferred (unsized, O2) matmuls are
  covered too, since sizes are known when the task runs.
- **Data stays host-resident (first step).** The kernel packs each operand
  into a dense buffer (any strides, broadcast batch axes), uploads it,
  computes, downloads and unpacks into the outputs, whose layouts the CPU
  path chose at issue. Element memory on the device (`CountableArray`'s
  per-device handles) and chains of device ops without transfers are the
  next step.
- **Threads.** Each thread running device work has its own in-order queue;
  calls block until their results are read back, so a device op occupies
  its worker like a CPU op. Kernel objects are made per call; an FFT plan
  runs one transform at a time (VkFFT sets its kernels' arguments).
- **Failures of the device** (out of memory, larger than its largest
  allocation, a build or launch error) run the CPU kernel instead:
  `offload_stats()["device_fallbacks"]`. `offload_stats()["device"]`
  counts the ops computed on the device.

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
  afterwards. R2C/C2R use VkFFT's in-place layout (rows of n/2 + 1 complex
  values). n-d transforms are NumPy's sequence of 1-d ones (`_fft.py`).
- **Errors** are NumPy's: a zero pivot fails `inv`/`solve` with
  "LinAlgError: Singular matrix" (an exact zero, as LAPACK's `info`), a
  non-positive pivot fails `cholesky` with "Matrix is not positive
  definite"; `det`/`slogdet` of a singular matrix give `0` and
  `(0, -inf)`. Raised asynchronously (`AsyncLinAlgError`), as on the CPU.
- **FP warnings**: the FP flags OpenCL, CLBlast and VkFFT's host code raise
  (plans) are cleared; only the host math NumPy's loops do too (det's
  `exp`/`log`, fft's scaling) can report one.

## Functions

On the device at O4 (above `device_min_work`): `det`, `slogdet`, `inv`,
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
a device the module is skipped. CI installs pocl's CPU device and CLBlast
on Linux (`pocl-opencl-icd`, `libclblast1`). pocl compiles kernels on first
use (CLBlast's take seconds each) and caches them (`~/.cache/pocl`).

## Assumptions (taken without review)

- O4 is opt-in and process-wide, like the other levels; levels 1-3 are
  unchanged apart from the new compositions, which are bit-identical.
- Default device: GPUs and accelerators only; CPU OpenCL devices only by
  name or `"cpu"`/`"any"`.
- `device_min_work` 2^24 is a first guess for a discrete GPU with a PCIe
  transfer per op; it is not calibrated per device yet (open).
- Linalg on the device needs fp64 because NumPy computes it in double;
  computing single precision inputs in single precision would deviate by
  more than rounding.
- Verified on Linux with pocl 5.0 (CPU device), CLBlast 1.6.2, VkFFT 1.3.4;
  Windows and macOS builds compile the same code, unverified on a device.

## Open

- Device-resident arrays (no transfer between consecutive device ops) and
  asynchronous queues (events instead of blocking reads).
- eig/eigh/svd/qr/lstsq on the device.
- A calibrated cost model per device (transfer bandwidth, launch latency,
  flop rate) instead of one threshold.
