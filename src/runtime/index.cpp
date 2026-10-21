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
                if (i < -n || i >= n) {
                    return bad(0, i);
                }
                fn(vp + t * vs_inner + (i < 0 ? i + n : i) * step, other + t * os_inner);
            }
        } else {
            for (std::int64_t t = 0; t < n_inner; ++t) {
                char* e = vp + t * vs_inner;
                for (std::size_t j = 0; j < nidx; ++j) {
                    const std::int64_t i =
                        read_index(ip[j] + t * inner_is[j], index[j].layout().dtype);
                    const std::int64_t n = extent[j];
                    if (i < -n || i >= n) {
                        return bad(j, i);
                    }
                    e += (i < 0 ? i + n : i) * vstep[j];
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

// Element copy of a fixed size (the compiler turns it into one move).
template <std::size_t N> void copy_element(char* dst, const char* src) { std::memcpy(dst, src, N); }

// Runs `body(copy)` with the element copy for `item` bytes.
template <class Body> std::string with_copy(std::size_t item, Body&& body) {
    switch (item) {
    case 1:
        return body(copy_element<1>);
    case 4:
        return body(copy_element<4>);
    case 8:
        return body(copy_element<8>);
    default:
        return body(copy_element<16>);
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

} // namespace viproc
