// Device kernels: NumPy's linalg and fft gufuncs and matmul on the OpenCL
// device (optimization level O4; docs/design/opencl.md).
//
// A device kernel wraps the CPU kernel of the same op (NumPy's loop) and
// keeps its contract: the same operands, outputs, sizes and layouts. When it
// runs, it decides by the op's actual sizes: work below min_work() runs the
// CPU kernel (the device's transfers and launches cost more than they save);
// larger work is packed into dense device buffers, computed there (CLBlast,
// VkFFT, viproc's LU/Cholesky kernels) and unpacked into the outputs. If the
// device fails (Error: out of memory, a build or launch error), the CPU
// kernel runs instead (counted in device_fallbacks). Errors NumPy raises
// (singular matrix, not positive definite) fail the op with NumPy's message,
// as the CPU kernel would.
#pragma once

#include "kernels/numpy_gufunc.hpp" // first: Python.h

#include "device/opencl/cl_device.hpp"
#include "runtime/task.hpp"

#include <memory>
#include <string>

namespace viproc::opencl {

enum class Op {
    None,
    Matmul,
    Det,
    Slogdet,
    Inv,
    Solve,
    Solve1,
    CholeskyLo,
    CholeskyUp,
    Fft,
    Ifft,
    RfftEven,
    RfftOdd,
    Irfft,
};

// The device op of a gufunc, by its name (numpy.linalg._umath_linalg,
// numpy.fft._pocketfft_umath, numpy.matmul) and operand counts; None if the
// device has none.
Op gufunc_op(const char* name, int nin, int nout);

// True if `ctx` can run `op` on operands of the loop dtype `dtype` (the
// first input's): float types only, double precision only with fp64,
// matmul only with CLBlast.
bool supports(const Context& ctx, Op op, DType dtype);

// A device kernel for a gufunc op with core axes `core` (vp_gufunc's) and
// the loop dtypes `dtypes` (inputs, then outputs). `invalid_error` as the CPU
// kernel's (numpy_gufunc.hpp).
std::shared_ptr<Kernel> make_gufunc_kernel(std::shared_ptr<Context> ctx, Op op,
                                           std::shared_ptr<Kernel> cpu,
                                           const numpy::GufuncCore& core, std::vector<DType> dtypes,
                                           std::string invalid_error);

// A device kernel for vp_matmul's op (operands a, b; result c; 1-d operands
// as NumPy's matmul: missing m or p).
std::shared_ptr<Kernel> make_matmul_kernel(std::shared_ptr<Context> ctx,
                                           std::shared_ptr<Kernel> cpu, DType dtype);

} // namespace viproc::opencl
