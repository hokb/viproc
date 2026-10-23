// Kernel adapter: reductions (sum, prod, min, max, any, all, ...) with NumPy's
// own reduction loops.
//
// viproc does not compute anything itself: it calls the binary loop NumPy
// resolves for the reduction (ufunc._resolve_dtypes_and_context(...,
// reduction=True)) with args {acc, in, acc}, in exactly the blocks NumPy's
// reduction passes to it. That keeps results bit-identical, also for
// floating-point `add`, whose loop sums each block pairwise.
//
// The blocks follow from NumPy's iterator (nditer with EXTERNAL_LOOP |
// GROWINNER | REDUCE_OK | DONT_NEGATE_STRIDES, result allocated in KEEPORDER):
//  - axes are ordered by the input's |stride|, smallest innermost (insertion
//    sort over C order; zero strides and extent-1 axes never decide);
//  - adjacent reduced axes are coalesced when the input is contiguous across
//    them (the result's stride is 0 along both);
//  - the loop is called once per position of the outer axes, over the whole
//    innermost axis (NumPy >= 2.3; earlier versions split it at the buffer
//    size, the bridge does not use this kernel there);
//  - the accumulator starts at the ufunc's identity (add, multiply, logical
//    and/or, ...); without one (minimum, maximum) it starts at the first
//    element along the reduced axes, and those elements are skipped.
// Not covered: reduced axes that do not coalesce into one (e.g. a full
// reduction of a non-contiguous array). There NumPy (>= 2.3) buffers the
// input and calls the loop on buffer-sized blocks; such reductions are left
// to NumPy (plan.supported() is false).
// Checked against NumPy for many shapes, axes and layouts
// (python/tests/test_reductions.py).
#pragma once

#include "kernels/numpy_ufunc.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace viproc::numpy {

struct ReduceAxis {
    std::int64_t shape;
    std::int64_t in_stride;  // bytes
    std::int64_t out_stride; // bytes; 0 along reduced axes
    bool reduced;
};

// Iteration of one reduction, outermost axis first; the last axis is the one
// each loop call covers. Never empty.
struct ReducePlan {
    SmallVector<ReduceAxis, kInlineRank> axes;

    // Strides of the loop calls: {acc, in, acc}.
    std::array<std::intptr_t, 3> inner_strides() const;
    // At most one reduced axis after coalescing: NumPy calls its loop as
    // planned (see above).
    bool supported() const;
};

// Plans reducing `in` over the axes flagged in `reduced` (one flag per input
// dimension) into the C-contiguous result `out` (with or without the reduced
// dimensions). Main thread, at issue time.
ReducePlan plan_reduction(const Layout& in, std::span<const bool> reduced, const Layout& out);

// The reduction: initializes the result (with `identity`, the identity's
// bytes in the result dtype, or with the first elements if `identity` is
// empty), then calls `loop` as planned. Operands: inputs = {in}, outputs =
// {result}. `loop` must stay alive while tasks use the kernel (LoopCache).
std::shared_ptr<Kernel> make_reduce_kernel(const UfuncLoop* loop, ReducePlan plan,
                                           std::vector<std::byte> identity);

// ufunc.accumulate along `axis` (cumsum, cumprod, ...), as
// PyUFunc_Accumulate runs it without casts: per position of the other axes,
// the first element is copied, then `loop` (the reduction loop of the
// accumulator dtype, fixed strides {out, in, out} along the axis) is called
// once with args {out[0..], in[1..], out[1..]} over the rest of the row, so
// each element is computed from the one before it. Input and result have the
// loop's dtype (the bridge casts first, as NumPy's iterator copies). Operands:
// inputs = {in}, outputs = {result, in's shape}.
std::shared_ptr<Kernel> make_accumulate_kernel(const UfuncLoop* loop, int axis);

// Reductions ReducePlan does not cover (casts, initial=, reduced axes that do
// not coalesce, strided inputs) run on NumPy's own iterator:
// LoopCache::iter_reduction() (numpy_ufunc.hpp), implemented in
// numpy_reduce.cpp.

// argmin / argmax with NumPy's per-dtype functions
// (PyDataType_GetArrFuncs(descr)->argmin/argmax), called as PyArray_ArgMax
// calls them: once per row of a C-contiguous input whose last axis is the
// reduced one (the bridge moves the axis last and copies, as NumPy does).
// Operands: inputs = {in}, outputs = {intp result, one per row}. Main thread
// with the GIL; nullptr if NumPy has no such function for `dtype`.
std::shared_ptr<Kernel> make_arg_reduce_kernel(DType dtype, bool argmax, std::string& error);

} // namespace viproc::numpy
