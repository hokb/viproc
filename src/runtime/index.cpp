#include "index.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

namespace viproc {

namespace {

std::int64_t read_index(const char* p, DType dtype) {
    if (dtype == DType::Int32) {
        std::int32_t v;
        std::memcpy(&v, p, sizeof v);
        return v;
    }
    std::int64_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

// Visits every element of the indexing result in C order. For each one,
// calls fn(view_element, other) where `other` advances along the result
// with the strides in `other_strides` (the result for gather, the values for
// scatter). The innermost result axis runs as a tight loop. Returns an error
// message ("" on success).
template <class Fn>
std::string for_each_indexed(const IndexSpec& spec, const Operand& view,
                             std::span<const Operand> index, char* other,
                             const Strides& other_strides, Fn&& fn) {
    const Layout& vl = view.layout();
    const std::size_t nr = spec.result_axes.size();
    Shape shape(nr, 0);
    // Per result axis: stride in the view (non-indexed axes) and in each
    // index array (bshape dimensions, broadcast: 0 where extent 1).
    Strides vstride(nr, 0);
    std::vector<Strides> istride(index.size(), Strides(nr, 0));
    const std::size_t nb = spec.bshape.size();
    for (std::size_t k = 0; k < nr; ++k) {
        const int a = spec.result_axes[k];
        if (a >= 0) {
            shape[k] = vl.shape[static_cast<std::size_t>(a)];
            vstride[k] = vl.strides[static_cast<std::size_t>(a)];
        } else {
            const std::size_t d = static_cast<std::size_t>(-1 - a);
            shape[k] = spec.bshape[d];
            for (std::size_t j = 0; j < index.size(); ++j) {
                const Layout& il = index[j].layout();
                const std::size_t lead = nb - il.shape.size();
                if (d >= lead && il.shape[d - lead] != 1) {
                    istride[j][k] = il.strides[d - lead];
                }
            }
        }
        if (shape[k] == 0) {
            return "";
        }
    }
    const std::size_t nidx = index.size();
    SmallVector<char*, 8> ip;
    for (const Operand& op : index) {
        ip.push_back(reinterpret_cast<char*>(op.first()));
    }
    SmallVector<std::int64_t, 8> extent;
    SmallVector<std::int64_t, 8> vstep;
    SmallVector<std::int64_t, 8> inner_is;
    for (std::size_t j = 0; j < nidx; ++j) {
        const int a = spec.index_axes[j];
        extent.push_back(vl.shape[static_cast<std::size_t>(a)]);
        vstep.push_back(vl.strides[static_cast<std::size_t>(a)]);
        inner_is.push_back(nr != 0 ? istride[j][nr - 1] : 0);
    }
    const std::int64_t n_inner = nr != 0 ? shape[nr - 1] : 1;
    const std::int64_t vs_inner = nr != 0 ? vstride[nr - 1] : 0;
    const std::int64_t os_inner = nr != 0 ? other_strides[nr - 1] : 0;
    const bool one_int64 = nidx == 1 && index[0].layout().dtype == DType::Int64;
    // An index outside [0, n) as NumPy's mode resolves it; -1 if it is out of
    // bounds (also for an empty axis, where no index is valid).
    const IndexMode mode = spec.mode;
    auto resolve = [mode](std::int64_t i, std::int64_t n) -> std::int64_t {
        if (n == 0) {
            return -1;
        }
        switch (mode) {
        case IndexMode::Raise:
            return i >= -n && i < 0 ? i + n : -1;
        case IndexMode::Wrap:
            i %= n;
            return i < 0 ? i + n : i;
        default:
            return i < 0 ? 0 : n - 1;
        }
    };
    auto bad = [&](std::size_t j, std::int64_t i) {
        return "IndexError: index " + std::to_string(i) + " is out of bounds for axis " +
               std::to_string(spec.index_axes[j]) + " with size " + std::to_string(extent[j]);
    };
    Shape idx(nr, 0);
    char* vp = reinterpret_cast<char*>(view.first());
    for (;;) {
        // One row along the innermost result axis.
        if (one_int64) {
            const char* p = ip[0];
            const std::int64_t n = extent[0];
            const std::int64_t step = vstep[0];
            for (std::int64_t t = 0; t < n_inner; ++t) {
                std::int64_t i;
                std::memcpy(&i, p + t * inner_is[0], sizeof i);
                if (i < 0 || i >= n) {
                    const std::int64_t r = resolve(i, n);
                    if (r < 0) {
                        return bad(0, i);
                    }
                    i = r;
                }
                fn(vp + t * vs_inner + i * step, other + t * os_inner);
            }
        } else {
            for (std::int64_t t = 0; t < n_inner; ++t) {
                char* e = vp + t * vs_inner;
                for (std::size_t j = 0; j < nidx; ++j) {
                    std::int64_t i = read_index(ip[j] + t * inner_is[j], index[j].layout().dtype);
                    const std::int64_t n = extent[j];
                    if (i < 0 || i >= n) {
                        const std::int64_t r = resolve(i, n);
                        if (r < 0) {
                            return bad(j, i);
                        }
                        i = r;
                    }
                    e += i * vstep[j];
                }
                fn(e, other + t * os_inner);
            }
        }
        // The outer axes (odometer).
        std::size_t k = nr != 0 ? nr - 1 : 0;
        for (;;) {
            if (k == 0) {
                return "";
            }
            --k;
            if (++idx[k] < shape[k]) {
                vp += vstride[k];
                other += other_strides[k];
                for (std::size_t j = 0; j < nidx; ++j) {
                    ip[j] += istride[j][k];
                }
                break;
            }
            const std::int64_t back = shape[k] - 1;
            vp -= back * vstride[k];
            other -= back * other_strides[k];
            for (std::size_t j = 0; j < nidx; ++j) {
                ip[j] -= back * istride[j][k];
            }
            idx[k] = 0;
        }
    }
}

// Element copy of a fixed size (the compiler turns it into one move). A type
// of its own per size, not a function pointer: the copy is inlined into the
// element loop instead of being an indirect call per element.
template <std::size_t N> struct CopyElement {
    void operator()(char* dst, const char* src) const { std::memcpy(dst, src, N); }
};

// Runs `body(copy)` with the element copy for `item` bytes.
template <class Body> std::string with_copy(std::size_t item, Body&& body) {
    switch (item) {
    case 1:
        return body(CopyElement<1>{});
    case 4:
        return body(CopyElement<4>{});
    case 8:
        return body(CopyElement<8>{});
    default:
        return body(CopyElement<16>{});
    }
}

class GatherKernel : public Kernel {
  public:
    explicit GatherKernel(IndexSpec spec) : spec_(std::move(spec)) {}
    const char* name() const override { return "indexing (a[index])"; }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Operand& out = outputs[0];
        // Any element order gives the same result (copies only): visit the
        // result in its memory order, so the innermost loop writes
        // contiguously (NumPy's index-outer layouts, a[:, idx]).
        IndexSpec spec = spec_;
        Strides strides = out.layout().strides;
        const std::size_t nr = spec.result_axes.size();
        std::vector<std::size_t> order(nr);
        for (std::size_t k = 0; k < nr; ++k) {
            order[k] = k;
        }
        std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) {
            return std::llabs(strides[x]) > std::llabs(strides[y]);
        });
        for (std::size_t k = 0; k < nr; ++k) {
            spec.result_axes[k] = spec_.result_axes[order[k]];
            strides[k] = out.layout().strides[order[k]];
        }
        const std::string e = with_copy(itemsize(out.layout().dtype), [&](auto copy) {
            return for_each_indexed(spec, inputs[0], inputs.subspan(1),
                                    reinterpret_cast<char*>(out.first()), strides,
                                    [copy](char* src, char* dst) { copy(dst, src); });
        });
        return e.empty() ? KernelResult{} : KernelResult{1, e, 0};
    }

  private:
    IndexSpec spec_;
};

class ScatterKernel : public Kernel {
  public:
    explicit ScatterKernel(IndexSpec spec) : spec_(std::move(spec)) {}
    const char* name() const override { return "index assignment (a[index] = v)"; }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Operand& dst = outputs[0];
        const std::size_t item = itemsize(dst.layout().dtype);
        const std::size_t nidx = spec_.index_axes.size();
        // The values, broadcast to the indexing result's shape.
        const Layout& vl = inputs[0].layout();
        const std::size_t nr = spec_.result_axes.size();
        Strides vs(nr, 0);
        const std::size_t lead = nr - vl.shape.size();
        for (std::size_t k = lead; k < nr; ++k) {
            if (vl.shape[k - lead] != 1) {
                vs[k] = vl.strides[k - lead];
            }
        }
        const std::string e = with_copy(item, [&](auto copy) {
            return for_each_indexed(spec_, dst, inputs.subspan(1, nidx),
                                    reinterpret_cast<char*>(inputs[0].first()), vs,
                                    [copy](char* d, char* src) { copy(d, src); });
        });
        return e.empty() ? KernelResult{} : KernelResult{1, e, 0};
    }

    bool writes_all() const override { return false; }

  private:
    IndexSpec spec_;
};

class MaskAssignKernel : public Kernel {
  public:
    const char* name() const override { return "masked assignment (a[mask] = v)"; }
    bool writes_all() const override { return false; }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Layout& dst = outputs[0].layout();
        const std::size_t item = itemsize(dst.dtype);
        const char* value = reinterpret_cast<const char*>(inputs[0].first());
        std::vector<Layout> layouts{broadcast_to(inputs[1].layout(), dst.shape), dst};
        OperandPointers base{reinterpret_cast<char*>(inputs[1].first()),
                             reinterpret_cast<char*>(outputs[0].first())};
        for_each_row(dst.shape, layouts, std::move(base),
                     [item, value](char** p, std::intptr_t n, const std::intptr_t* st) {
                         for (std::intptr_t i = 0; i < n; ++i) {
                             if (p[0][i * st[0]] != 0) {
                                 std::memcpy(p[1] + i * st[1], value, item);
                             }
                         }
                         return true;
                     });
        return {};
    }
};

// Branch-free select of `n` contiguous elements of N bytes: the condition
// of np.where is random data as often as not, and a branch per element
// mispredicts (NumPy's contiguous loop is branch-free too).
template <std::size_t N>
void select_dense(const char* cond, const char* x, const char* y, char* out, std::intptr_t n) {
    using W = std::conditional_t<N == 1, std::uint8_t,
                                 std::conditional_t<N == 4, std::uint32_t, std::uint64_t>>;
    constexpr std::size_t words = N <= 8 ? 1 : N / 8;
    constexpr std::size_t step = N <= 8 ? N : 8;
    for (std::intptr_t i = 0; i < n; ++i) {
        const W m = static_cast<W>(-static_cast<W>(cond[i] != 0));
        for (std::size_t k = 0; k < words; ++k) {
            const std::size_t at = static_cast<std::size_t>(i) * N + k * step;
            W a;
            W b;
            std::memcpy(&a, x + at, step);
            std::memcpy(&b, y + at, step);
            const W r = static_cast<W>((a & m) | (b & static_cast<W>(~m)));
            std::memcpy(out + at, &r, step);
        }
    }
}

template <std::size_t N> constexpr std::size_t item_size_of(CopyElement<N>) { return N; }

class SelectKernel : public Kernel {
  public:
    const char* name() const override { return "numpy.where"; }
    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Layout& out = outputs[0].layout();
        const std::size_t item = itemsize(out.dtype);
        std::vector<Layout> layouts{broadcast_to(inputs[0].layout(), out.shape),
                                    broadcast_to(inputs[1].layout(), out.shape),
                                    broadcast_to(inputs[2].layout(), out.shape), out};
        OperandPointers base{reinterpret_cast<char*>(inputs[0].first()),
                             reinterpret_cast<char*>(inputs[1].first()),
                             reinterpret_cast<char*>(inputs[2].first()),
                             reinterpret_cast<char*>(outputs[0].first())};
        with_copy(item, [&](auto copy) {
            for_each_row(out.shape, layouts, std::move(base),
                         [copy](char** p, std::intptr_t n, const std::intptr_t* st) {
                             constexpr auto N = item_size_of(decltype(copy){});
                             if (st[0] == 1 && st[1] == N && st[2] == N && st[3] == N) {
                                 select_dense<N>(p[0], p[1], p[2], p[3], n);
                                 return true;
                             }
                             for (std::intptr_t i = 0; i < n; ++i) {
                                 const char* src =
                                     p[0][i * st[0]] != 0 ? p[1] + i * st[1] : p[2] + i * st[2];
                                 copy(p[3] + i * st[3], src);
                             }
                             return true;
                         });
            return std::string();
        });
        return {};
    }
};

} // namespace

std::shared_ptr<Kernel> make_select_kernel() { return std::make_shared<SelectKernel>(); }

std::shared_ptr<Kernel> make_gather_kernel(IndexSpec spec) {
    return std::make_shared<GatherKernel>(std::move(spec));
}

std::shared_ptr<Kernel> make_scatter_kernel(IndexSpec spec) {
    return std::make_shared<ScatterKernel>(std::move(spec));
}

std::shared_ptr<Kernel> make_mask_assign_kernel() { return std::make_shared<MaskAssignKernel>(); }

namespace {

// Calls f(element pointer) for each element of `l` at `base`, in C order.
template <typename F> void for_each_c(const Layout& l, const std::byte* base, F&& f) {
    const std::size_t nd = l.shape.size();
    for (const std::int64_t e : l.shape) {
        if (e == 0) {
            return;
        }
    }
    SmallVector<std::int64_t, 4> idx(nd, 0);
    const std::byte* p = base;
    for (;;) {
        f(p);
        std::size_t d = nd;
        for (;;) {
            if (d == 0) {
                return;
            }
            --d;
            if (++idx[d] < l.shape[d]) {
                p += l.strides[d];
                break;
            }
            p -= l.strides[d] * (l.shape[d] - 1);
            idx[d] = 0;
        }
    }
}

bool nonzero_element(DType dt, const std::byte* p) {
    switch (dt) {
    case DType::Bool:
        return *reinterpret_cast<const std::uint8_t*>(p) != 0;
    case DType::Int32:
        return *reinterpret_cast<const std::int32_t*>(p) != 0;
    case DType::Int64:
        return *reinterpret_cast<const std::int64_t*>(p) != 0;
    case DType::Float32:
        return *reinterpret_cast<const float*>(p) != 0.0f; // NaN != 0
    case DType::Float64:
        return *reinterpret_cast<const double*>(p) != 0.0;
    case DType::Complex64: {
        const auto* c = reinterpret_cast<const float*>(p);
        return c[0] != 0.0f || c[1] != 0.0f;
    }
    case DType::Complex128: {
        const auto* c = reinterpret_cast<const double*>(p);
        return c[0] != 0.0 || c[1] != 0.0;
    }
    }
    return false;
}

class FlatNonzeroKernel : public Kernel {
  public:
    const char* name() const override { return "numpy.flatnonzero"; }
    bool sizes_from_data() const override { return true; }
    KernelResult size_from_data(std::span<const Operand> in, std::span<Layout> out) override {
        std::int64_t n = 0;
        const DType dt = in[0].layout().dtype;
        for_each_c(in[0].layout(), in[0].first(),
                   [&](const std::byte* p) { n += nonzero_element(dt, p) ? 1 : 0; });
        out[0].shape = Shape{n};
        return {};
    }
    KernelResult run(std::span<const Operand> in, std::span<const Operand> out) override {
        auto* r = reinterpret_cast<std::int64_t*>(out[0].first());
        const DType dt = in[0].layout().dtype;
        std::int64_t i = 0;
        for_each_c(in[0].layout(), in[0].first(), [&](const std::byte* p) {
            if (nonzero_element(dt, p)) {
                *r++ = i;
            }
            ++i;
        });
        return {};
    }
};

class MaskSelectKernel : public Kernel {
  public:
    const char* name() const override { return "indexing (a[mask])"; }
    bool sizes_from_data() const override { return true; }
    KernelResult size_from_data(std::span<const Operand> in, std::span<Layout> out) override {
        const Layout& x = in[0].layout();
        const Layout& m = in[1].layout();
        if (m.shape.size() > x.shape.size()) {
            return KernelResult{1,
                                "IndexError: too many indices for array: array is " +
                                    std::to_string(x.shape.size()) + "-dimensional, but " +
                                    std::to_string(m.shape.size()) + " were indexed",
                                0};
        }
        for (std::size_t d = 0; d < m.shape.size(); ++d) {
            if (m.shape[d] != x.shape[d]) {
                return KernelResult{
                    1,
                    "IndexError: boolean index did not match indexed array along axis " +
                        std::to_string(d) + "; size of axis is " + std::to_string(x.shape[d]) +
                        " but size of corresponding boolean axis is " + std::to_string(m.shape[d]),
                    0};
            }
        }
        std::int64_t n = 0;
        for_each_c(m, in[1].first(), [&](const std::byte* p) { n += *p != std::byte{0} ? 1 : 0; });
        Shape shape{n};
        for (std::size_t d = m.shape.size(); d < x.shape.size(); ++d) {
            shape.push_back(x.shape[d]);
        }
        out[0].shape = std::move(shape);
        return {};
    }
    KernelResult run(std::span<const Operand> in, std::span<const Operand> out) override {
        const Layout& x = in[0].layout();
        const Layout& m = in[1].layout();
        const std::size_t k = m.shape.size();
        Layout sub; // x's subarray at one mask position
        sub.dtype = x.dtype;
        for (std::size_t d = k; d < x.shape.size(); ++d) {
            sub.shape.push_back(x.shape[d]);
            sub.strides.push_back(x.strides[d]);
        }
        const std::size_t item = itemsize(x.dtype);
        std::byte* r = out[0].first(); // C order
        // The mask positions in C order, walking the mask and x's leading axes
        // together.
        SmallVector<std::int64_t, 4> idx(k, 0);
        bool done = false;
        for (const std::int64_t e : m.shape) {
            done = done || e == 0;
        }
        const std::byte* pm = in[1].first();
        const std::byte* px = in[0].first();
        while (!done) {
            if (*pm != std::byte{0}) {
                for_each_c(sub, px, [&](const std::byte* p) {
                    std::memcpy(r, p, item);
                    r += item;
                });
            }
            std::size_t d = k;
            for (;;) {
                if (d == 0) {
                    done = true;
                    break;
                }
                --d;
                if (++idx[d] < m.shape[d]) {
                    pm += m.strides[d];
                    px += x.strides[d];
                    break;
                }
                pm -= m.strides[d] * (m.shape[d] - 1);
                px -= x.strides[d] * (m.shape[d] - 1);
                idx[d] = 0;
            }
        }
        return {};
    }
};

} // namespace

std::shared_ptr<Kernel> make_mask_select_kernel() { return std::make_shared<MaskSelectKernel>(); }
std::shared_ptr<Kernel> make_flatnonzero_kernel() { return std::make_shared<FlatNonzeroKernel>(); }

} // namespace viproc
