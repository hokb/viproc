// Element types and array layouts (shape, byte strides, offset).
#pragma once

#include "small_vector.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace viproc {

enum class DType : std::uint8_t { Bool, Int32, Int64, Float32, Float64, Complex64, Complex128 };

constexpr std::size_t itemsize(DType dtype) {
    switch (dtype) {
    case DType::Bool:
        return 1;
    case DType::Int32:
    case DType::Float32:
        return 4;
    case DType::Int64:
    case DType::Float64:
    case DType::Complex64:
        return 8;
    case DType::Complex128:
        return 16;
    }
    return 0;
}

// Shapes and strides live inline up to this rank (no heap allocation).
inline constexpr std::size_t kInlineRank = 4;
using Shape = SmallVector<std::int64_t, kInlineRank>;
using Strides = SmallVector<std::int64_t, kInlineRank>;

// Logical view onto a buffer. Strides are in bytes, like NumPy's.
struct Layout {
    DType dtype = DType::Float64;
    Shape shape;
    Strides strides;
    std::int64_t offset = 0; // byte offset of the first element in the buffer

    // C-contiguous layout of `shape`; a 0-d shape is a scalar with one element.
    static Layout contiguous(DType dtype, Shape shape);

    std::int64_t size() const;
    // True if the elements lie C-contiguously (dimensions of extent 1 may
    // have any stride).
    bool c_contiguous() const;
    // Bytes a buffer needs to hold this layout when it is contiguous.
    std::size_t nbytes() const;
};

// NumPy broadcasting of two shapes. Returns false if they are incompatible.
bool broadcast_shapes(const Shape& a, const Shape& b, Shape& out);

// View of `in` with shape `target` (`in.shape` must broadcast to it).
// Broadcast dimensions get stride 0.
Layout broadcast_to(const Layout& in, const Shape& target);

// Calls `row(ptrs, count, strides)` once per innermost row of `shape` for a
// set of operands, all already broadcast to `shape` (`layouts[k]`, first
// element at `base[k]`). A 0-d shape is one row of one element. Returns
// false as soon as `row` returns false.
using OperandPointers = SmallVector<char*, 8>;

template <class RowFn>
bool for_each_row(const Shape& shape, const std::vector<Layout>& layouts, OperandPointers ptrs,
                  RowFn&& row) {
    for (std::int64_t d : shape) {
        if (d == 0) {
            return true;
        }
    }
    const std::size_t nops = layouts.size();
    const std::size_t nd = shape.size();
    const std::intptr_t count = nd == 0 ? 1 : static_cast<std::intptr_t>(shape[nd - 1]);
    SmallVector<std::intptr_t, 8> strides(nops);
    for (std::size_t k = 0; k < nops; ++k) {
        strides[k] = nd == 0 ? 0 : static_cast<std::intptr_t>(layouts[k].strides[nd - 1]);
    }
    SmallVector<std::int64_t, kInlineRank> index(nd > 0 ? nd - 1 : 0, 0);
    for (;;) {
        if (!row(ptrs.data(), count, strides.data())) {
            return false;
        }
        // Odometer over all but the innermost dimension.
        std::size_t dim = index.size();
        while (dim-- > 0) {
            if (++index[dim] < shape[dim]) {
                for (std::size_t k = 0; k < nops; ++k) {
                    ptrs[k] += layouts[k].strides[dim];
                }
                break;
            }
            for (std::size_t k = 0; k < nops; ++k) {
                ptrs[k] -= layouts[k].strides[dim] * (shape[dim] - 1);
            }
            index[dim] = 0;
        }
        if (dim == static_cast<std::size_t>(-1)) {
            return true;
        }
    }
}

} // namespace viproc
