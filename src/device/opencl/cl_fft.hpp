// 1-d FFTs on the OpenCL device with VkFFT (MIT, header-only, compiled into
// viproc; docs/design/opencl.md, "FFT").
#pragma once

#include "device/opencl/cl_device.hpp"

#include <cstdint>

namespace viproc::opencl {

enum class FftKind { Forward, Backward, RealForward, RealBackward };

// Runs `batch` transforms of `n` points in place on `buf` (`dbl`: double
// precision), unnormalized (pocketfft's convention; the caller applies
// NumPy's factor). Complex kinds: batch rows of n complex values. Real
// kinds (R2C, C2R): batch rows of 2 * (n / 2 + 1) reals, the first n of
// them the real signal and the whole row the n / 2 + 1 complex values of the
// half spectrum (VkFFT's in-place layout). Plans are cached per (context,
// kind, precision, n, batch); a plan runs one transform at a time. Throws
// Error. Any thread.
void fft(Context& ctx, cl_command_queue q, FftKind kind, bool dbl, std::uint64_t n,
         std::uint64_t batch, Buffer& buf);

// Drops the cached plans of `ctx` (before it is destroyed).
void drop_fft_plans(const Context* ctx);

} // namespace viproc::opencl
