// Element copy kernels: dst[...] = src (with broadcasting, same dtype), and a
// new array assembled from parts (concatenate, stack, tile, pad, ...).
#pragma once

#include "task.hpp"

#include <memory>
#include <vector>

namespace viproc {

// Kernel operands: inputs = [src, dst (as in/out)], outputs = [dst].
std::shared_ptr<Kernel> make_assign_kernel();

// One part of a placed array: the elements at `layout` (shape, byte strides
// and byte offset from the output's first element) get input i.
struct Part {
    Shape shape;
    Strides strides;
    std::int64_t offset = 0;
};

// out = a new array whose part i is a copy of inputs[i] (out's dtype,
// broadcastable to the part's shape), in the order of the parts (a later part
// overwrites an earlier one where they overlap). Operands: inputs = one per
// part, outputs = [out]. The parts must lie within out's elements.
std::shared_ptr<Kernel> make_place_kernel(std::vector<Part> parts);

} // namespace viproc
