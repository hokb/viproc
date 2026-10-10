#include "layout.hpp"

#include <algorithm>
#include <cassert>

namespace viproc {

std::size_t itemsize(DType dtype) {
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

Layout Layout::contiguous(DType dtype, Shape shape) {
    Layout l;
    l.dtype = dtype;
    l.strides.resize(shape.size());
    std::int64_t stride = static_cast<std::int64_t>(itemsize(dtype));
    for (std::size_t i = shape.size(); i-- > 0;) {
        l.strides[i] = stride;
        stride *= shape[i];
    }
    l.shape = std::move(shape);
    return l;
}

bool Layout::c_contiguous() const {
    std::int64_t stride = static_cast<std::int64_t>(itemsize(dtype));
    for (std::size_t i = shape.size(); i-- > 0;) {
        if (shape[i] != 1 && strides[i] != stride) {
            return false;
        }
        stride *= shape[i];
    }
    return true;
}

std::int64_t Layout::size() const {
    std::int64_t n = 1;
    for (std::int64_t d : shape) {
        n *= d;
    }
    return n;
}

std::size_t Layout::nbytes() const { return static_cast<std::size_t>(size()) * itemsize(dtype); }

bool broadcast_shapes(const Shape& a, const Shape& b, Shape& out) {
    const std::size_t nd = std::max(a.size(), b.size());
    Shape r(nd);
    for (std::size_t i = 0; i < nd; ++i) {
        const std::int64_t da = i < nd - a.size() ? 1 : a[i - (nd - a.size())];
        const std::int64_t db = i < nd - b.size() ? 1 : b[i - (nd - b.size())];
        if (da != db && da != 1 && db != 1) {
            return false;
        }
        r[i] = da == 1 ? db : da;
    }
    out = std::move(r);
    return true;
}

Layout broadcast_to(const Layout& in, const Shape& target) {
    assert(target.size() >= in.shape.size());
    Layout l;
    l.dtype = in.dtype;
    l.offset = in.offset;
    l.shape = target;
    l.strides.assign(target.size(), 0);
    const std::size_t lead = target.size() - in.shape.size();
    for (std::size_t i = 0; i < in.shape.size(); ++i) {
        assert(in.shape[i] == target[lead + i] || in.shape[i] == 1);
        l.strides[lead + i] = in.shape[i] == 1 ? 0 : in.strides[i];
    }
    return l;
}

} // namespace viproc
