// Gather at O2 (pipelining.md, "Gathers at O2"): sized and laid out in the
// task. A kernel of its own, in a file of its own, wrapping the gather
// kernel of index.cpp: that file's inlining (and so the O1 gather's code)
// stays as it is.
#include "index.hpp"

#include <string>
#include <vector>

namespace viproc {

namespace {

// "(3,)" / "(3,2)": NumPy's shape format in index error messages.
std::string shape_text(const Shape& s) {
    std::string t = "(";
    for (std::size_t i = 0; i < s.size(); ++i) {
        t += (i ? "," : "") + std::to_string(s[i]);
    }
    return t + (s.size() == 1 ? ",)" : ")");
}

class DeferredGatherKernel : public Kernel {
  public:
    explicit DeferredGatherKernel(IndexSpec spec) : spec_(std::move(spec)) {}
    const char* name() const override { return "indexing (a[index])"; }

    // The index arrays' broadcast shape (NumPy's IndexError if they do not
    // broadcast) and the result shape. Runs once, before layout() and run()
    // (the task's stages order them): it completes the spec and makes the
    // gather kernel that run() calls.
    KernelResult size(std::span<const Layout* const> inputs, std::span<Layout> outputs) override {
        for (const Layout* l : inputs.subspan(1)) {
            Shape b;
            if (!broadcast_shapes(spec_.bshape, l->shape, b)) {
                std::string e = "IndexError: shape mismatch: indexing arrays could not be "
                                "broadcast together with shapes ";
                for (const Layout* m : inputs.subspan(1)) {
                    e += shape_text(m->shape) + " ";
                }
                return KernelResult{1, e, 0};
            }
            spec_.bshape = std::move(b);
        }
        Shape shape;
        for (const int a : spec_.result_axes) {
            shape.push_back(a >= 0 ? inputs[0]->shape[static_cast<std::size_t>(a)]
                                   : spec_.bshape[static_cast<std::size_t>(-1 - a)]);
        }
        outputs[0].shape = std::move(shape);
        gather_ = make_gather_kernel(spec_);
        return {};
    }

    // NumPy's layout for the result (mapiter): the broadcast index
    // dimensions outermost, in C order; inside them the other axes in the
    // order NumPy's iterator keeps the source's (Layout::iteration_order);
    // an empty result has all strides 0.
    void layout(std::span<const Layout* const> inputs, std::span<Layout> outputs) override {
        Layout& o = outputs[0];
        const Layout& v = *inputs[0];
        const std::size_t nr = spec_.result_axes.size();
        std::vector<std::size_t> sub; // result axes from the source, in result order
        Layout sv;
        sv.dtype = v.dtype;
        for (std::size_t k = 0; k < nr; ++k) {
            const int a = spec_.result_axes[k];
            if (a >= 0) {
                sub.push_back(k);
                sv.shape.push_back(v.shape[static_cast<std::size_t>(a)]);
                sv.strides.push_back(v.strides[static_cast<std::size_t>(a)]);
            }
        }
        o.strides.assign(nr, 0);
        std::int64_t step = static_cast<std::int64_t>(itemsize(o.dtype));
        const Layout one[] = {sv};
        for (std::size_t i : Layout::iteration_order(sv.shape, one)) { // innermost first
            o.strides[sub[i]] = step;
            step *= o.shape[sub[i]];
        }
        for (std::size_t k = nr; k-- > 0;) {
            if (spec_.result_axes[k] < 0) {
                o.strides[k] = step;
                step *= o.shape[k];
            }
        }
        if (step == 0) {
            o.strides.assign(nr, 0);
        }
    }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        return gather_->run(inputs, outputs);
    }

  private:
    IndexSpec spec_;
    std::shared_ptr<Kernel> gather_;
};

} // namespace

std::shared_ptr<Kernel> make_deferred_gather_kernel(IndexSpec spec) {
    return std::make_shared<DeferredGatherKernel>(std::move(spec));
}

} // namespace viproc
