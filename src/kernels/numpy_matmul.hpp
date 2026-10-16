// Kernel adapter: matrix products (np.matmul, the `@` operator, np.dot) with
// NumPy's own matmul gufunc loop.
//
// matmul is the gufunc (m?,n),(n,p?)->(m?,p?). NumPy runs it through
// PyUFunc_GeneralizedFunctionInternal: an iterator over the broadcast batch
// dimensions without buffering (operands are only copied for a cast, a
// misalignment or an overlap with the output; viproc issues none of these),
// and one loop call per inner batch row with
//   dims    = {count, m, n, p}                (missing m or p: 1)
//   strides = {a, b, out outer strides, a_m, a_n, b_n, b_p, out_m, out_p}
// where a core stride is 0 for a missing dimension or one of extent 1.
// The loop (matmul.c.src) handles every outer index on its own: BLAS gemm,
// syrk (a @ a.T), gemv or dot where the strides allow it, copies to a
// BLAS-able buffer otherwise, a plain loop for integers and bool. Only the
// core strides and pointers decide which; how the batch is split into calls
// does not change any value. viproc passes the same core strides, so it
// makes the same BLAS calls.
//
// BLAS results depend on the BLAS library's own thread count (OpenBLAS
// splits large products differently with 1, 2 or 4 threads), so viproc
// leaves BLAS threading as NumPy has it: concurrent calls from several
// workers give the values of the same calls made one after the other.
#pragma once

#include "kernels/numpy_ufunc.hpp"

#include <memory>

namespace viproc::numpy {

// Shape and layout of np.matmul(a, b) (NumPy's gufunc rules): false if NumPy
// raises (0-d operand, core dimension mismatch, batch shapes that do not
// broadcast). `out` gets NumPy's result layout: the batch axes in the order
// NumPy's iterator allocates them (order K over the inputs' batch strides),
// the core axes C-contiguous innermost. `work` sizes the op for the dispatch
// policy (multiply-adds, scaled to element-wise ops).
bool matmul_layout(const Layout& a, const Layout& b, Layout& out, std::int64_t& work);

// Kernel calling NumPy's matmul loop `loop` (resolved for (t, t) -> t) with
// NumPy's strides. Operands: inputs {a, b}, output {out} as from
// matmul_layout(). Any thread, without the GIL.
std::shared_ptr<Kernel> make_matmul_kernel(const UfuncLoop* loop);

} // namespace viproc::numpy
