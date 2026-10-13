// Advanced indexing kernels: gather (r = v[i0, i1, ...]), scatter
// (v[i0, i1, ...] = values) and masked assignment (v[mask] = value). Pure
// data movement (element copies), so any element order gives NumPy's
// result; the bridge supplies NumPy's result shape and layout.
#pragma once

#include "task.hpp"

#include <memory>
#include <vector>

namespace viproc {

// Integer array indices into some axes of a source (or destination) view.
// The index arrays (int32 or int64) broadcast to `bshape`. Each axis of the
// indexing result comes from either a non-indexed axis of the view
// (result_axes[k] >= 0, the view's axis) or a dimension of `bshape`
// (result_axes[k] = -1 - dimension).
struct IndexSpec {
    std::vector<int> index_axes; // view axis of each index array
    Shape bshape;
    std::vector<int> result_axes;
};

// Gather. Operands: inputs = {src view, index arrays...}, outputs = {result}.
// An index out of bounds fails the task with an IndexError message.
std::shared_ptr<Kernel> make_gather_kernel(IndexSpec spec);

// Scatter (NumPy's order: the last of repeated indices wins). Operands:
// inputs = {values (broadcastable to the indexing result's shape, the view's
// dtype), index arrays..., dst view (in/out)}, outputs = {dst view}.
std::shared_ptr<Kernel> make_scatter_kernel(IndexSpec spec);

// dst[mask] = value for a 0-d `value`; `mask` (bool) broadcasts to dst.
// Operands: inputs = {value, mask, dst (in/out)}, outputs = {dst}.
std::shared_ptr<Kernel> make_mask_assign_kernel();

// np.where(cond, x, y): out = cond ? x : y element by element, all three
// (bool cond; x, y of out's dtype) broadcast to out. Operands: inputs =
// {cond, x, y}, outputs = {out}.
std::shared_ptr<Kernel> make_select_kernel();

} // namespace viproc
