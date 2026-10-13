#include "layout.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <utility>
#include <vector>

namespace viproc {

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

Layout Layout::keep_order(DType dtype, Shape shape, std::span<const Layout> inputs) {
    const std::size_t nd = shape.size();
    // Strides of each input along the result's axes; 0 where it broadcasts
    // and along extent-1 axes (as NumPy's iterator sets them).
    std::vector<Strides> st;
    for (const Layout& in : inputs) {
        Strides s(nd, 0);
        const std::size_t lead = nd - std::min(nd, in.shape.size());
        for (std::size_t i = 0; i < in.shape.size() && lead + i < nd; ++i) {
            if (in.shape[i] != 1 && shape[lead + i] != 1) {
                s[lead + i] = in.strides[i];
            }
        }
        st.push_back(std::move(s));
    }
    // npyiter_find_best_axis_ordering: insertion sort, innermost first,
    // starting from C order. An input votes for moving an axis inward if its
    // |stride| is smaller; zero strides abstain; any "no" vote keeps C order.
    std::vector<std::size_t> perm(nd);
    for (std::size_t i = 0; i < nd; ++i) {
        perm[i] = nd - 1 - i;
    }
    for (std::size_t i0 = 1; i0 < nd; ++i0) {
        const std::size_t j0 = perm[i0];
        std::size_t ipos = i0;
        for (std::size_t i1 = i0; i1-- > 0;) {
            const std::size_t j1 = perm[i1];
            bool ambiguous = true;
            bool swap = false;
            for (const Strides& s : st) {
                if (s[j0] != 0 && s[j1] != 0) {
                    if (std::llabs(s[j1]) <= std::llabs(s[j0])) {
                        swap = false;
                    } else if (ambiguous) {
                        swap = true;
                    }
                    ambiguous = false;
                }
            }
            if (!ambiguous) {
                if (!swap) {
                    break;
                }
                ipos = i1;
            }
        }
        if (ipos != i0) {
            const std::size_t moved = perm[i0];
            for (std::size_t k = i0; k > ipos; --k) {
                perm[k] = perm[k - 1];
            }
            perm[ipos] = moved;
        }
    }
    Layout l;
    l.dtype = dtype;
    l.strides.resize(nd);
    std::int64_t step = static_cast<std::int64_t>(itemsize(dtype));
    for (std::size_t ax : perm) {
        l.strides[ax] = step;
        step *= shape[ax];
    }
    l.shape = std::move(shape);
    return l;
}

bool Layout::dense() const {
    if (offset != 0) {
        return false;
    }
    std::vector<std::pair<std::int64_t, std::int64_t>> axes; // (stride, extent)
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0) {
            return true;
        }
        if (shape[i] != 1) {
            axes.emplace_back(strides[i], shape[i]);
        }
    }
    std::sort(axes.begin(), axes.end());
    std::int64_t step = static_cast<std::int64_t>(itemsize(dtype));
    for (const auto& [stride, extent] : axes) {
        if (stride != step) {
            return false;
        }
        step *= extent;
    }
    return true;
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
