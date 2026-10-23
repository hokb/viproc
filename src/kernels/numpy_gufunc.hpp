// Kernel adapter: generalized ufuncs (gufuncs with core dimensions: vecdot,
// matvec, vecmat, numpy.linalg's _umath_linalg, numpy.fft's
// _pocketfft_umath) with NumPy's own gufunc loops.
//
// NumPy runs a gufunc through PyUFunc_GeneralizedFunctionInternal: an
// iterator over the broadcast batch (loop) dimensions without buffering
// (operands are copied only for a cast, a misalignment or an overlap with
// an output), and one loop call per inner batch row with
//   dims    = {count, size of each core dimension label...}
//   strides = {outer stride per operand..., core strides per operand axis...}
// where a core stride is 0 for an axis of extent 1 or a missing flexible
// dimension. The loops viproc runs this way compute every outer index on its
// own (linalg: copy the matrix into a LAPACK buffer, call LAPACK; fft: copy,
// pocketfft's plan per call; vecdot & co.: a dot per index), so how the batch
// is split into calls does not change any value. viproc passes NumPy's core
// strides, so each index gets the calls NumPy makes for it.
#pragma once

#include "kernels/numpy_ufunc.hpp"

#include <memory>
#include <string>
#include <vector>

namespace viproc::numpy {

// The core axes of a gufunc's operands: for each operand (inputs, then
// outputs) the label index of each core axis of the signature, in order, or
// missing(label) where the operand lacks that flexible dimension (matmul's
// `m?` of a 1-d operand: size 1, stride 0, no axis).
struct GufuncCore {
    static constexpr int missing(int label) { return -1 - label; }
    int nlabels = 0;
    std::vector<std::vector<int>> axes; // per operand

    bool operator==(const GufuncCore&) const = default;
    // Number of axes operand `i` has for its core dimensions.
    std::size_t present(std::size_t i) const;
};

// Kernel calling the gufunc loop `loop` (resolved for the operands' dtypes)
// as NumPy calls it. Operands: inputs (nin, core axes last, batch axes
// broadcasting to the outputs'), outputs (nout, batch axes followed by
// their core axes). `invalid_error` non-empty: the loop reports failure
// through the floating-point invalid flag (numpy.linalg's loops, which NumPy's
// wrappers turn into LinAlgError with errstate(invalid='call')); the op
// then fails with that message ("LinAlgError: Singular matrix").
// `options` (bits, as vp_gufunc's flags):
//  - kGufuncCheckFinite: the op fails with "LinAlgError: Array must not
//    contain infs or NaNs" if input 0 has a non-finite element
//    (numpy.linalg's _assert_finite, which NumPy runs before eig and
//    eigvals; one pass over the input, as NumPy's isfinite(a).all()).
//  - kGufuncSerial: the loop calls of all such ops run one at a time
//    (a process-wide mutex): LAPACK on matrices OpenBLAS threads itself.
//    Several threaded OpenBLAS calls at once oversubscribe its thread pool
//    (measured: 8 concurrent 200x200 inv took 1x-4.5x NumPy's time,
//    bimodal); one at a time they take NumPy's.
// Any thread, without the GIL.
enum : int { kGufuncCheckFinite = 2, kGufuncSerial = 4 };
std::shared_ptr<Kernel> make_gufunc_kernel(const UfuncLoop* loop, GufuncCore core,
                                           std::string invalid_error, int options = 0);

} // namespace viproc::numpy
