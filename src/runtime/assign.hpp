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

namespace viproc {

// A creator with sizes from pending arrays (pipelining.md, S4; O2):
// np.zeros/ones/empty/full(shape) where `dims` has -1 for each extent taken
// from an input (a 0-d int32/int64 array, in order), or np.arange(stop)
// (`iota`: one input, values 0..stop-1 of the output dtype, int64). The task
// sizes its output when the inputs' data is there (Kernel::sizes_from_data;
// NumPy's "negative dimensions are not allowed"), lays it out in `order`
// ('C' or 'F') and fills it with `fill` (itemsize bytes; empty: any). `name`:
// a string with static storage ("numpy.zeros").
std::shared_ptr<Kernel> make_create_kernel(std::vector<std::int64_t> dims, char order,
                                           std::vector<std::byte> fill, bool iota,
                                           const char* name);

} // namespace viproc
