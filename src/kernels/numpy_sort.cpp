#include "kernels/numpy_sort.hpp"

#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL viproc_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace viproc::numpy {

namespace {

class SortKernel : public Kernel {
  public:
    SortKernel(PyArray_SortFunc* sort, PyArray_ArgSortFunc* argsort, DType dtype, int axis)
        : sort_(sort), argsort_(argsort), dtype_(dtype), axis_(static_cast<std::size_t>(axis)) {}
    const char* name() const override { return argsort_ != nullptr ? "argsort" : "sort"; }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        // inputs: [in] or [in/out]; outputs: [out] or [in/out]
        const Operand& src = inputs[0];
        const Operand& dst = outputs[0];
        const Layout& ls = src.layout();
        const Layout& ld = dst.layout();
        bool in_place = src.first() == dst.first() && ls.strides == ld.strides;
        const auto es = static_cast<std::intptr_t>(itemsize(dtype_));
        if (argsort_ == nullptr && !in_place && ls.strides == ld.strides &&
            ld.strides[axis_] != es && (ld.c_contiguous() || ld.f_contiguous())) {
            // as NumPy: the whole copy first (one pass in memory order), then
            // each row sorted in place through a buffer
            std::memcpy(dst.first(), src.first(), static_cast<std::size_t>(ld.size() * es));
            in_place = true;
        }
        const std::size_t nd = ls.shape.size();
        const std::int64_t n = ls.shape[axis_];
        const auto ss = static_cast<std::intptr_t>(ls.strides[axis_]);
        const auto ds = static_cast<std::intptr_t>(ld.strides[axis_]);
        std::int64_t rows = n == 0 ? 0 : 1;
        for (std::size_t a = 0; a < nd; ++a) {
            if (a != axis_) {
                rows *= ls.shape[a];
            }
        }
        std::vector<std::byte> vbuf;
        std::vector<npy_intp> ibuf;
        std::vector<std::int64_t> idx(nd, 0);
        for (std::int64_t r = 0; r < rows; ++r) {
            std::intptr_t so = 0;
            std::intptr_t dof = 0;
            for (std::size_t a = 0; a < nd; ++a) {
                so += static_cast<std::intptr_t>(idx[a] * ls.strides[a]);
                dof += static_cast<std::intptr_t>(idx[a] * ld.strides[a]);
            }
            const std::byte* s = (in_place ? dst.first() : src.first()) + so;
            std::byte* d = dst.first() + dof;
            if (argsort_ == nullptr) {
                if (!sort_row(s, ss, d, ds, n, es, in_place, vbuf)) {
                    return {1, "MemoryError: NumPy's sort failed", 0};
                }
            } else if (!argsort_row(s, ss, d, ds, n, es, vbuf, ibuf)) {
                return {1, "MemoryError: NumPy's argsort failed", 0};
            }
            for (std::size_t a = nd; a-- > 0;) {
                if (a == axis_) {
                    continue;
                }
                if (++idx[a] < ls.shape[a]) {
                    break;
                }
                idx[a] = 0;
            }
        }
        return {};
    }

  private:
    // n elements of es bytes from s (stride ss) to d (stride ds).
    static void copy(std::byte* d, std::intptr_t ds, const std::byte* s, std::intptr_t ss,
                     std::int64_t n, std::intptr_t es) {
        if (ds == es && ss == es) {
            std::memcpy(d, s, static_cast<std::size_t>(n * es));
            return;
        }
        switch (es) {
        case 1:
            copy_as<std::uint8_t>(d, ds, s, ss, n);
            return;
        case 4:
            copy_as<std::uint32_t>(d, ds, s, ss, n);
            return;
        case 8:
            copy_as<std::uint64_t>(d, ds, s, ss, n);
            return;
        default:
            for (std::int64_t i = 0; i < n; ++i) {
                std::memcpy(d + i * ds, s + i * ss, static_cast<std::size_t>(es));
            }
        }
    }

    template <class T>
    static void copy_as(std::byte* d, std::intptr_t ds, const std::byte* s, std::intptr_t ss,
                        std::int64_t n) {
        for (std::int64_t i = 0; i < n; ++i) {
            T v;
            std::memcpy(&v, s + i * ss, sizeof(T));
            std::memcpy(d + i * ds, &v, sizeof(T));
        }
    }

    static bool aligned(const std::byte* p, std::intptr_t es) {
        const auto align = static_cast<std::uintptr_t>(es >= 16 ? es / 2 : es);
        return reinterpret_cast<std::uintptr_t>(p) % align == 0;
    }

    bool sort_row(const std::byte* s, std::intptr_t ss, std::byte* d, std::intptr_t ds,
                  std::int64_t n, std::intptr_t es, bool in_place,
                  std::vector<std::byte>& buf) const {
        const bool direct = ds == es && aligned(d, es);
        std::byte* row = d;
        if (!direct) {
            buf.resize(static_cast<std::size_t>(n * es));
            row = buf.data();
        }
        if (!(in_place && direct)) {
            copy(row, es, s, ss, n, es);
        }
        if (n > 1 && sort_(row, static_cast<npy_intp>(n), nullptr) < 0) {
            return false;
        }
        if (!direct) {
            copy(d, ds, row, es, n, es);
        }
        return true;
    }

    bool argsort_row(const std::byte* s, std::intptr_t ss, std::byte* d, std::intptr_t ds,
                     std::int64_t n, std::intptr_t es, std::vector<std::byte>& vbuf,
                     std::vector<npy_intp>& ibuf) const {
        const std::byte* vals = s;
        if (ss != es || !aligned(s, es)) {
            vbuf.resize(static_cast<std::size_t>(n * es));
            copy(vbuf.data(), es, s, ss, n, es);
            vals = vbuf.data();
        }
        const bool direct =
            ds == static_cast<std::intptr_t>(sizeof(npy_intp)) && aligned(d, sizeof(npy_intp));
        npy_intp* out = reinterpret_cast<npy_intp*>(d);
        if (!direct) {
            ibuf.resize(static_cast<std::size_t>(n));
            out = ibuf.data();
        }
        for (std::int64_t i = 0; i < n; ++i) {
            out[i] = static_cast<npy_intp>(i);
        }
        // NumPy argsorts rows of one element too (a no-op); rows of none
        // have no output.
        if (n > 1 &&
            argsort_(const_cast<std::byte*>(vals), out, static_cast<npy_intp>(n), nullptr) < 0) {
            return false;
        }
        if (!direct) {
            copy(d, ds, reinterpret_cast<const std::byte*>(out),
                 static_cast<std::intptr_t>(sizeof(npy_intp)), n,
                 static_cast<std::intptr_t>(sizeof(npy_intp)));
        }
        return true;
    }

    PyArray_SortFunc* sort_;
    PyArray_ArgSortFunc* argsort_;
    DType dtype_;
    std::size_t axis_;
};

} // namespace

std::shared_ptr<Kernel> make_sort_kernel(DType dtype, int kind, bool arg, int axis,
                                         std::string& error) {
    if (kind < 0 || kind >= NPY_NSORTS || axis < 0) {
        error = "sort: unsupported kind or axis";
        return nullptr;
    }
    PyArray_Descr* descr = PyArray_DescrFromType(type_num(dtype));
    PyArray_ArrFuncs* f = PyDataType_GetArrFuncs(descr);
    Py_DECREF(descr);
    PyArray_SortFunc* sort = arg ? nullptr : f->sort[kind];
    PyArray_ArgSortFunc* argsort = arg ? f->argsort[kind] : nullptr;
    if (sort == nullptr && argsort == nullptr) {
        error = "sort: no NumPy sort function for this dtype";
        return nullptr;
    }
    return std::make_shared<SortKernel>(sort, argsort, dtype, axis);
}

} // namespace viproc::numpy
