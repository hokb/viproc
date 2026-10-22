#include "kernels/numpy_matmul.hpp"

#include <algorithm>
#include <vector>

namespace viproc::numpy {

namespace {

// Number of core dimensions of a matmul operand (1-d: the missing m or p).
std::size_t ncore(const Layout& l) { return l.shape.size() >= 2 ? 2 : 1; }

Shape batch_shape(const Layout& l) {
    return Shape(l.shape.begin(), l.shape.end() - static_cast<std::ptrdiff_t>(ncore(l)));
}

// NumPy's inner stride for a core axis: 0 where the extent is 1.
std::intptr_t core_stride(const Layout& l, std::size_t axis) {
    return l.shape[axis] != 1 ? static_cast<std::intptr_t>(l.strides[axis]) : 0;
}

// Stride of `l`'s batch axes along the result's `nb` batch axes (0 where `l`
// broadcasts or lacks the axis).
std::vector<std::intptr_t> batch_strides(const Layout& l, const Shape& out_batch) {
    const std::size_t nb = out_batch.size();
    const std::size_t lb = l.shape.size() - ncore(l);
    std::vector<std::intptr_t> s(nb, 0);
    for (std::size_t i = 0; i < lb; ++i) {
        const std::size_t k = nb - lb + i;
        if (l.shape[i] != 1) {
            s[k] = static_cast<std::intptr_t>(l.strides[i]);
        }
    }
    return s;
}

class MatmulKernel : public Kernel {
  public:
    explicit MatmulKernel(const UfuncLoop* loop) : loop_(loop) {}
    const char* name() const override { return loop_->name(); }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Layout& a = inputs[0].layout();
        const Layout& b = inputs[1].layout();
        const Layout& o = outputs[0].layout();
        const std::size_t na = a.shape.size();
        const std::size_t nbd = b.shape.size();
        const bool has_m = na >= 2;
        const bool has_p = nbd >= 2;
        const std::size_t ob = o.shape.size() - (has_m ? 1 : 0) - (has_p ? 1 : 0);
        const Shape batch(o.shape.begin(), o.shape.begin() + static_cast<std::ptrdiff_t>(ob));

        // dims = {count, m, n, p}; strides = {outer a, b, out, then core}.
        std::intptr_t dims[4] = {1, has_m ? static_cast<std::intptr_t>(a.shape[na - 2]) : 1,
                                 static_cast<std::intptr_t>(a.shape[na - 1]),
                                 has_p ? static_cast<std::intptr_t>(b.shape[nbd - 1]) : 1};
        std::intptr_t strides[9] = {0,
                                    0,
                                    0,
                                    has_m ? core_stride(a, na - 2) : 0,
                                    core_stride(a, na - 1),
                                    core_stride(b, has_p ? nbd - 2 : nbd - 1),
                                    has_p ? core_stride(b, nbd - 1) : 0,
                                    has_m ? core_stride(o, ob) : 0,
                                    has_p ? core_stride(o, o.shape.size() - 1) : 0};
        char* base[3] = {reinterpret_cast<char*>(inputs[0].first()),
                         reinterpret_cast<char*>(inputs[1].first()),
                         reinterpret_cast<char*>(outputs[0].first())};
        if (ob == 0) {
            return call(base, dims, strides);
        }
        std::int64_t total = 1;
        for (std::int64_t e : batch) {
            total *= e;
        }
        if (total == 0) {
            return {}; // NumPy's iterator is empty: the loop is not called
        }
        // Batch strides per operand; the innermost batch axis is the call's
        // outer loop, the others are walked here (any order: each outer index
        // is computed on its own).
        std::vector<std::intptr_t> st[3] = {batch_strides(a, batch), batch_strides(b, batch),
                                            std::vector<std::intptr_t>(ob, 0)};
        for (std::size_t i = 0; i < ob; ++i) {
            st[2][i] = batch[i] != 1 ? static_cast<std::intptr_t>(o.strides[i]) : 0;
        }
        dims[0] = static_cast<std::intptr_t>(batch[ob - 1]);
        for (int k = 0; k < 3; ++k) {
            strides[k] = st[k][ob - 1];
        }
        std::vector<std::int64_t> idx(ob, 0);
        for (;;) {
            char* ptrs[3];
            for (int k = 0; k < 3; ++k) {
                std::intptr_t off = 0;
                for (std::size_t i = 0; i + 1 < ob; ++i) {
                    off += static_cast<std::intptr_t>(idx[i]) * st[k][i];
                }
                ptrs[k] = base[k] + off;
            }
            KernelResult r = call(ptrs, dims, strides);
            if (r.status != 0) {
                return r;
            }
            std::size_t i = ob - 1;
            while (i > 0) {
                --i;
                if (++idx[i] < batch[i]) {
                    break;
                }
                idx[i] = 0;
                if (i == 0) {
                    return {};
                }
            }
            if (ob == 1) {
                return {};
            }
        }
    }

  private:
    KernelResult call(char** ptrs, const std::intptr_t* dims, const std::intptr_t* strides) const {
        return loop_->call(ptrs, dims, strides) == 0 ? KernelResult{}
                                                     : KernelResult{1, "NumPy matmul failed", 0};
    }

    const UfuncLoop* loop_;
};

} // namespace

bool matmul_layout(const Layout& a, const Layout& b, Layout& out, std::int64_t& work) {
    if (a.shape.empty() || b.shape.empty() || a.dtype != b.dtype) {
        return false;
    }
    const std::size_t na = a.shape.size();
    const std::size_t nbd = b.shape.size();
    const std::int64_t n = a.shape[na - 1];
    if (n != b.shape[nbd >= 2 ? nbd - 2 : nbd - 1]) {
        return false;
    }
    Shape batch;
    if (!broadcast_shapes(batch_shape(a), batch_shape(b), batch)) {
        return false;
    }
    const std::size_t nb = batch.size();
    Shape shape = batch;
    if (na >= 2) {
        shape.push_back(a.shape[na - 2]);
    }
    if (nbd >= 2) {
        shape.push_back(b.shape[nbd - 1]);
    }
    // NumPy's iterator allocates the result over the batch axes followed by
    // the output's core axes; the inputs have zero strides along the core
    // axes, so those stay innermost in C order and only the batch axes are
    // ordered by the inputs' strides (Layout::keep_order with the inputs
    // padded by extent-1 core axes).
    const std::size_t ncore_out = shape.size() - nb;
    Layout pa = a, pb = b;
    for (Layout* p : {&pa, &pb}) {
        const std::size_t keep = p->shape.size() - ncore(*p);
        p->shape.resize(keep);
        p->strides.resize(keep);
        for (std::size_t i = 0; i < ncore_out; ++i) {
            p->shape.push_back(1);
            p->strides.push_back(0);
        }
    }
    const Layout ins[] = {pa, pb};
    out = Layout::keep_order(a.dtype, std::move(shape), ins);
    const std::int64_t size = out.size();
    work = std::max<std::int64_t>(size, size / 16 * std::max<std::int64_t>(n, 1));
    return true;
}

std::shared_ptr<Kernel> make_matmul_kernel(const UfuncLoop* loop) {
    return std::make_shared<MatmulKernel>(loop);
}

} // namespace viproc::numpy
