#include "kernels/numpy_gufunc.hpp"

#include <cfenv>
#include <cmath>
#include <cstring>
#include <mutex>

namespace viproc::numpy {

std::size_t GufuncCore::present(std::size_t i) const {
    std::size_t n = 0;
    for (int label : axes[i]) {
        n += label >= 0 ? 1 : 0;
    }
    return n;
}

namespace {

std::mutex& serial_mutex() {
    static std::mutex m;
    return m;
}

// Whether every element of `l` at `base` is finite (floating point and
// complex: every component; other dtypes are finite).
bool all_finite(const Layout& l, const std::byte* base) {
    std::size_t parts = 1;
    bool single = false;
    switch (l.dtype) {
    case DType::Float32:
        single = true;
        break;
    case DType::Float64:
        break;
    case DType::Complex64:
        single = true;
        parts = 2;
        break;
    case DType::Complex128:
        parts = 2;
        break;
    default:
        return true;
    }
    const std::size_t nd = l.shape.size();
    for (std::int64_t n : l.shape) {
        if (n == 0) {
            return true;
        }
    }
    std::vector<std::int64_t> idx(nd, 0);
    for (;;) {
        std::intptr_t off = 0;
        for (std::size_t a = 0; a < nd; ++a) {
            off += static_cast<std::intptr_t>(idx[a] * l.strides[a]);
        }
        const std::byte* p = base + off;
        for (std::size_t c = 0; c < parts; ++c) {
            if (single) {
                float v;
                std::memcpy(&v, p + c * sizeof v, sizeof v);
                if (!std::isfinite(v)) {
                    return false;
                }
            } else {
                double v;
                std::memcpy(&v, p + c * sizeof v, sizeof v);
                if (!std::isfinite(v)) {
                    return false;
                }
            }
        }
        std::size_t a = nd;
        for (;;) {
            if (a == 0) {
                return true;
            }
            --a;
            if (++idx[a] < l.shape[a]) {
                break;
            }
            idx[a] = 0;
        }
    }
}

class GufuncKernel : public Kernel {
  public:
    GufuncKernel(const UfuncLoop* loop, GufuncCore core, std::string invalid_error, int options)
        : loop_(loop), core_(std::move(core)), invalid_error_(std::move(invalid_error)),
          options_(options) {
        for (const std::vector<int>& a : core_.axes) {
            ncore_ += a.size();
        }
    }
    const char* name() const override { return loop_->name(); }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const std::size_t nin = static_cast<std::size_t>(loop_->nin());
        const std::size_t nop = core_.axes.size();
        // inputs: the gufunc's inputs, then in/out outputs again (ignored)
        std::vector<const Operand*> ops(nop);
        for (std::size_t i = 0; i < nop; ++i) {
            ops[i] = i < nin ? &inputs[i] : &outputs[i - nin];
        }
        if ((options_ & kGufuncCheckFinite) != 0 &&
            !all_finite(inputs[0].layout(), inputs[0].first())) {
            return KernelResult{1, "LinAlgError: Array must not contain infs or NaNs", 0};
        }
        // dims = {count, label sizes}; strides = {outer per operand, core}
        std::vector<std::intptr_t> dims(1 + static_cast<std::size_t>(core_.nlabels), 1);
        std::vector<std::intptr_t> strides(nop + ncore_, 0);
        std::size_t k = nop;
        for (std::size_t i = 0; i < nop; ++i) {
            const Layout& l = ops[i]->layout();
            std::size_t ax = l.shape.size() - core_.present(i);
            for (int label : core_.axes[i]) {
                if (label >= 0) {
                    dims[1 + static_cast<std::size_t>(label)] =
                        static_cast<std::intptr_t>(l.shape[ax]);
                    strides[k] = l.shape[ax] != 1 ? static_cast<std::intptr_t>(l.strides[ax]) : 0;
                    ++ax;
                }
                ++k;
            }
        }
        const Layout& o = outputs[0].layout();
        const std::size_t nb = o.shape.size() - core_.present(nin);
        std::int64_t total = 1;
        for (std::size_t i = 0; i < nb; ++i) {
            total *= o.shape[i];
        }
        if (total == 0) {
            return {}; // NumPy's iterator is empty: the loop is not called
        }
        std::unique_lock<std::mutex> serial;
        if ((options_ & kGufuncSerial) != 0) {
            serial = std::unique_lock<std::mutex>(serial_mutex());
        }
        std::vector<char*> base(nop);
        // batch strides per operand along the outputs' nb batch axes
        std::vector<std::intptr_t> st(nop * nb, 0);
        for (std::size_t i = 0; i < nop; ++i) {
            base[i] = reinterpret_cast<char*>(ops[i]->first());
            const Layout& l = ops[i]->layout();
            const std::size_t lb = l.shape.size() - core_.present(i);
            for (std::size_t a = 0; a < lb; ++a) {
                if (l.shape[a] != 1) {
                    st[i * nb + nb - lb + a] = static_cast<std::intptr_t>(l.strides[a]);
                }
            }
        }
        if (nb == 0) {
            return call(base.data(), dims.data(), strides.data());
        }
        // The innermost batch axis is the call's outer loop, the others are
        // walked here (any order: each outer index is computed on its own).
        dims[0] = static_cast<std::intptr_t>(o.shape[nb - 1]);
        for (std::size_t i = 0; i < nop; ++i) {
            strides[i] = st[i * nb + nb - 1];
        }
        std::vector<std::int64_t> idx(nb, 0);
        std::vector<char*> ptrs(nop);
        for (;;) {
            for (std::size_t i = 0; i < nop; ++i) {
                std::intptr_t off = 0;
                for (std::size_t a = 0; a + 1 < nb; ++a) {
                    off += static_cast<std::intptr_t>(idx[a]) * st[i * nb + a];
                }
                ptrs[i] = base[i] + off;
            }
            KernelResult r = call(ptrs.data(), dims.data(), strides.data());
            if (r.status != 0) {
                return r;
            }
            std::size_t a = nb - 1;
            for (;;) {
                if (a == 0) {
                    return {};
                }
                --a;
                if (++idx[a] < o.shape[a]) {
                    break;
                }
                idx[a] = 0;
            }
        }
    }

  private:
    KernelResult call(char** ptrs, const std::intptr_t* dims, const std::intptr_t* strides) const {
        if (!invalid_error_.empty()) {
            std::feclearexcept(FE_INVALID);
        }
        if (loop_->call(ptrs, dims, strides) != 0) {
            return KernelResult{1, std::string("NumPy ") + loop_->name() + " failed", 0};
        }
        if (!invalid_error_.empty() && std::fetestexcept(FE_INVALID) != 0) {
            std::feclearexcept(FE_INVALID); // NumPy's wrapper handles the flag
            return KernelResult{1, invalid_error_, 0};
        }
        return {};
    }

    const UfuncLoop* loop_;
    GufuncCore core_;
    std::string invalid_error_;
    int options_;
    std::size_t ncore_ = 0;
};

} // namespace

std::shared_ptr<Kernel> make_gufunc_kernel(const UfuncLoop* loop, GufuncCore core,
                                           std::string invalid_error, int options) {
    return std::make_shared<GufuncKernel>(loop, std::move(core), std::move(invalid_error), options);
}

} // namespace viproc::numpy
