#include "kernels/numpy_reduce.hpp"

#include <cstdlib>
#include <cstring>
#include <utility>

namespace viproc::numpy {

std::array<std::intptr_t, 3> ReducePlan::inner_strides() const {
    const ReduceAxis& in = axes.back();
    return {static_cast<std::intptr_t>(in.out_stride), static_cast<std::intptr_t>(in.in_stride),
            static_cast<std::intptr_t>(in.out_stride)};
}

bool ReducePlan::supported() const {
    int reduced = 0;
    for (const ReduceAxis& a : axes) {
        reduced += a.reduced ? 1 : 0;
    }
    return reduced <= 1;
}

ReducePlan plan_reduction(const Layout& in, std::span<const bool> reduced, const Layout& out) {
    const std::size_t nd = in.shape.size();
    const auto isz = static_cast<std::int64_t>(itemsize(out.dtype));

    // Result stride of each kept input axis: the result is C-contiguous over
    // the kept axes (extent-1 dimensions for reduced axes, if kept, add
    // nothing).
    std::vector<std::int64_t> out_stride(nd, 0);
    std::int64_t step = isz;
    for (std::size_t i = nd; i-- > 0;) {
        if (!reduced[i]) {
            out_stride[i] = step;
            step *= in.shape[i];
        }
    }

    // NumPy's axis data, innermost first (C order reversed). Extent-1 axes
    // are dropped: their stride is 0 for every operand, which makes them
    // transparent to the ordering and coalescing below.
    std::vector<ReduceAxis> ax;
    for (std::size_t i = nd; i-- > 0;) {
        if (in.shape[i] != 1) {
            ax.push_back({in.shape[i], in.strides[i], reduced[i] ? 0 : out_stride[i], reduced[i]});
        }
    }

    // npyiter_find_best_axis_ordering with one operand that has strides (the
    // result is allocated after the ordering): insertion sort, smaller
    // |stride| inner; a comparison involving a zero stride is ambiguous and
    // does not stop the search.
    for (std::size_t i0 = 1; i0 < ax.size(); ++i0) {
        const std::int64_t s0 = std::llabs(ax[i0].in_stride);
        std::size_t ipos = i0;
        for (std::size_t i1 = i0; i1-- > 0;) {
            const std::int64_t s1 = std::llabs(ax[i1].in_stride);
            if (s0 == 0 || s1 == 0) {
                continue; // ambiguous
            }
            if (s1 <= s0) {
                break;
            }
            ipos = i1;
        }
        if (ipos != i0) {
            ReduceAxis moved = ax[i0];
            for (std::size_t k = i0; k > ipos; --k) {
                ax[k] = ax[k - 1];
            }
            ax[ipos] = moved;
        }
    }

    // npyiter_coalesce_axes, for adjacent reduced axes (the result's stride is
    // 0 along both). Kept axes are left alone: NumPy may merge them too, but
    // across kept axes the loop accumulates element-wise, so where the calls
    // are split does not change any result.
    std::vector<ReduceAxis> merged;
    for (const ReduceAxis& a : ax) {
        if (!merged.empty()) {
            ReduceAxis& inner = merged.back();
            if (inner.reduced && a.reduced && a.in_stride == inner.shape * inner.in_stride) {
                inner.shape *= a.shape;
                if (inner.in_stride == 0) {
                    inner.in_stride = a.in_stride;
                }
                continue;
            }
        }
        merged.push_back(a);
    }
    if (merged.empty()) {
        merged.push_back({1, 0, 0, true}); // a single element
    }

    ReducePlan plan;
    for (std::size_t k = merged.size(); k-- > 0;) {
        plan.axes.push_back(merged[k]); // outermost first
    }
    return plan;
}

namespace {

class ReduceKernel : public Kernel {
  public:
    ReduceKernel(const UfuncLoop* loop, ReducePlan plan, std::vector<std::byte> identity)
        : loop_(loop), plan_(std::move(plan)), identity_(std::move(identity)) {}

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Layout& out_layout = outputs[0].layout;
        const std::size_t isz = itemsize(out_layout.dtype);
        char* const in = reinterpret_cast<char*>(inputs[0].first());
        char* const out = reinterpret_cast<char*>(outputs[0].first());
        const auto& axes = plan_.axes;
        const std::size_t nouter = axes.size() - 1;

        // Initial values: the identity, or the first elements along the
        // reduced axes (as PyArray_CopyInitialReduceValues).
        if (!identity_.empty()) {
            const std::int64_t n = out_layout.size();
            for (std::int64_t i = 0; i < n; ++i) {
                std::memcpy(out + i * static_cast<std::int64_t>(isz), identity_.data(), isz);
            }
        } else {
            for_each(
                axes.size(),
                [&](char* src, char* dst, bool first) {
                    if (first) {
                        std::memcpy(dst, src, isz);
                    }
                    return true;
                },
                in, out, /*kept_only=*/true);
        }

        // The loop calls, once per position of the outer axes. Without an
        // identity, elements already copied are skipped, as NumPy's
        // reduce_loop does on the first visit of a result element.
        const ReduceAxis& inner = axes.back();
        const std::intptr_t strides[3] = {static_cast<std::intptr_t>(inner.out_stride),
                                          static_cast<std::intptr_t>(inner.in_stride),
                                          static_cast<std::intptr_t>(inner.out_stride)};
        const bool skip_first = identity_.empty();
        const bool ok = for_each(
            nouter,
            [&](char* src, char* dst, bool first) {
                std::intptr_t count = static_cast<std::intptr_t>(inner.shape);
                if (skip_first && first) {
                    if (inner.reduced) {
                        --count;
                        src += inner.in_stride;
                    } else {
                        count = 0;
                    }
                }
                if (count <= 0) {
                    return true;
                }
                char* data[3] = {dst, src, dst};
                return loop_->call(data, &count, strides) == 0;
            },
            in, out, /*kept_only=*/false);
        return ok ? KernelResult{} : KernelResult{1, "NumPy reduction loop failed", 0};
    }

  private:
    // Visits the positions of the first `n` plan axes (outermost first),
    // calling fn(in_ptr, out_ptr, first) where `first` tells whether
    // all reduced coordinates are 0. With kept_only, reduced axes stay at 0
    // (one visit per result element; n then covers all axes).
    template <class Fn>
    bool for_each(std::size_t n, Fn&& fn, char* in, char* out, bool kept_only) const {
        const auto& axes = plan_.axes;
        SmallVector<std::int64_t, kInlineRank> idx(n, 0);
        for (;;) {
            bool first = true;
            for (std::size_t k = 0; k < n; ++k) {
                first = first && (!axes[k].reduced || idx[k] == 0);
            }
            if (!fn(in, out, first)) {
                return false;
            }
            std::size_t k = n;
            for (;;) {
                if (k == 0) {
                    return true;
                }
                --k;
                if (kept_only && axes[k].reduced) {
                    continue;
                }
                if (++idx[k] < axes[k].shape) {
                    in += axes[k].in_stride;
                    out += axes[k].out_stride;
                    break;
                }
                in -= (axes[k].shape - 1) * axes[k].in_stride;
                out -= (axes[k].shape - 1) * axes[k].out_stride;
                idx[k] = 0;
            }
        }
    }

    const UfuncLoop* loop_;
    ReducePlan plan_;
    std::vector<std::byte> identity_;
};

} // namespace

std::shared_ptr<Kernel> make_reduce_kernel(const UfuncLoop* loop, ReducePlan plan,
                                           std::vector<std::byte> identity) {
    return std::make_shared<ReduceKernel>(loop, std::move(plan), std::move(identity));
}

} // namespace viproc::numpy
