// Element types and array layouts (shape, byte strides, offset).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace viproc {

enum class DType : std::uint8_t { Bool, Int32, Int64, Float32, Float64, Complex64, Complex128 };

std::size_t itemsize(DType dtype);

using Shape = std::vector<std::int64_t>;

// Logical view onto a buffer. Strides are in bytes, like NumPy's.
struct Layout {
    DType dtype = DType::Float64;
    Shape shape;
    std::vector<std::int64_t> strides;
    std::int64_t offset = 0; // byte offset of the first element in the buffer

    // C-contiguous layout of `shape`; a 0-d shape is a scalar with one element.
    static Layout contiguous(DType dtype, Shape shape);

    std::int64_t size() const;
    // Bytes a buffer needs to hold this layout when it is contiguous.
    std::size_t nbytes() const;
};

// NumPy broadcasting of two shapes. Returns false if they are incompatible.
bool broadcast_shapes(const Shape& a, const Shape& b, Shape& out);

// View of `in` with shape `target` (`in.shape` must broadcast to it).
// Broadcast dimensions get stride 0.
Layout broadcast_to(const Layout& in, const Shape& target);

} // namespace viproc
