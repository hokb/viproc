#include "index.hpp"

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
// scatter). Returns an error message ("" on success).
template <class Fn>
std::string for_each_indexed(const IndexSpec& spec, const Operand& view,
                             std::span<const Operand> index, char* other,
                             const Strides& other_strides, Fn&& fn) {
    const Layout& vl = view.layout;
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
                const Layout& il = index[j].layout;
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
    char* vbase = reinterpret_cast<char*>(view.first());
    SmallVector<char*, 8> ip;
    for (const Operand& op : index) {
        ip.push_back(reinterpret_cast<char*>(op.first()));
    }
    SmallVector<std::int64_t, 8> extent;
    SmallVector<std::int64_t, 8> vstep;
    for (int a : spec.index_axes) {
        extent.push_back(vl.shape[static_cast<std::size_t>(a)]);
        vstep.push_back(vl.strides[static_cast<std::size_t>(a)]);
    }
    Shape idx(nr, 0);
    char* vp = vbase;
    for (;;) {
        char* e = vp;
        for (std::size_t j = 0; j < index.size(); ++j) {
            std::int64_t i = read_index(ip[j], index[j].layout.dtype);
            const std::int64_t n = extent[j];
            if (i < -n || i >= n) {
                return "IndexError: index " + std::to_string(i) + " is out of bounds for axis " +
                       std::to_string(spec.index_axes[j]) + " with size " + std::to_string(n);
            }
            e += (i < 0 ? i + n : i) * vstep[j];
        }
        fn(e, other);
        std::size_t k = nr;
        for (;;) {
            if (k == 0) {
                return "";
            }
            --k;
            if (++idx[k] < shape[k]) {
                vp += vstride[k];
                other += other_strides[k];
                for (std::size_t j = 0; j < ip.size(); ++j) {
                    ip[j] += istride[j][k];
                }
                break;
            }
            const std::int64_t back = shape[k] - 1;
            vp -= back * vstride[k];
            other -= back * other_strides[k];
            for (std::size_t j = 0; j < ip.size(); ++j) {
                ip[j] -= back * istride[j][k];
            }
            idx[k] = 0;
        }
    }
}

class GatherKernel : public Kernel {
  public:
    explicit GatherKernel(IndexSpec spec) : spec_(std::move(spec)) {}

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Operand& out = outputs[0];
        const std::size_t item = itemsize(out.layout.dtype);
        const std::string e = for_each_indexed(
            spec_, inputs[0], inputs.subspan(1), reinterpret_cast<char*>(out.first()),
            out.layout.strides, [item](char* src, char* dst) { std::memcpy(dst, src, item); });
        return e.empty() ? KernelResult{} : KernelResult{1, e, 0};
    }

  private:
    IndexSpec spec_;
};

class ScatterKernel : public Kernel {
  public:
    explicit ScatterKernel(IndexSpec spec) : spec_(std::move(spec)) {}

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Operand& dst = outputs[0];
        const std::size_t item = itemsize(dst.layout.dtype);
        const std::size_t nidx = spec_.index_axes.size();
        // The values, broadcast to the indexing result's shape.
        const Layout& vl = inputs[0].layout;
        const std::size_t nr = spec_.result_axes.size();
        Strides vs(nr, 0);
        const std::size_t lead = nr - vl.shape.size();
        for (std::size_t k = lead; k < nr; ++k) {
            if (vl.shape[k - lead] != 1) {
                vs[k] = vl.strides[k - lead];
            }
        }
        const std::string e = for_each_indexed(
            spec_, dst, inputs.subspan(1, nidx), reinterpret_cast<char*>(inputs[0].first()), vs,
            [item](char* d, char* src) { std::memcpy(d, src, item); });
        return e.empty() ? KernelResult{} : KernelResult{1, e, 0};
    }

  private:
    IndexSpec spec_;
};

class MaskAssignKernel : public Kernel {
  public:
    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Layout& dst = outputs[0].layout;
        const std::size_t item = itemsize(dst.dtype);
        const char* value = reinterpret_cast<const char*>(inputs[0].first());
        std::vector<Layout> layouts{broadcast_to(inputs[1].layout, dst.shape), dst};
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

} // namespace

std::shared_ptr<Kernel> make_gather_kernel(IndexSpec spec) {
    return std::make_shared<GatherKernel>(std::move(spec));
}

std::shared_ptr<Kernel> make_scatter_kernel(IndexSpec spec) {
    return std::make_shared<ScatterKernel>(std::move(spec));
}

std::shared_ptr<Kernel> make_mask_assign_kernel() { return std::make_shared<MaskAssignKernel>(); }

} // namespace viproc
