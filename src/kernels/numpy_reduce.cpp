#include "kernels/numpy_reduce.hpp"

#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL viproc_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL viproc_UFUNC_API
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#include <cstdlib>
#include <cstring>
#include <string>
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

namespace {

// op_axes entry of a reduction axis (NumPy's NPY_ITER_REDUCTION_AXIS, which
// its public headers do not export; the encoding is part of the nditer API).
constexpr int reduction_axis(int axis) { return axis + (1 << (NPY_BITSOF_INT - 2)); }

std::string fetch_error() {
    PyObject* exc = PyErr_GetRaisedException();
    if (exc == nullptr) {
        return "unknown error";
    }
    PyObject* str = PyObject_Str(exc);
    std::string msg = str != nullptr ? PyUnicode_AsUTF8(str) : "unknown error";
    Py_XDECREF(str);
    Py_DECREF(exc);
    return msg;
}

constexpr const char* kOperandCapsule = "viproc.operand";

void release_operand(PyObject* capsule) {
    delete static_cast<std::shared_ptr<Buffer>*>(PyCapsule_GetPointer(capsule, kOperandCapsule));
}

// A read-only NumPy view of `layout` in `buffer` (kept alive by the view).
PyObject* operand_view(const Layout& layout, const std::shared_ptr<Buffer>& buffer) {
    std::vector<npy_intp> shape(layout.shape.begin(), layout.shape.end());
    std::vector<npy_intp> strides(layout.strides.begin(), layout.strides.end());
    PyArray_Descr* descr = PyArray_DescrFromType(type_num(layout.dtype));
    PyObject* view = PyArray_NewFromDescr(
        &PyArray_Type, descr, static_cast<int>(shape.size()), shape.data(), strides.data(),
        buffer->data() + layout.offset, NPY_ARRAY_ALIGNED, nullptr); // steals descr
    if (view == nullptr) {
        return nullptr;
    }
    auto* hold = new std::shared_ptr<Buffer>(buffer);
    PyObject* capsule = PyCapsule_New(hold, kOperandCapsule, release_operand);
    if (capsule == nullptr) {
        delete hold;
        Py_DECREF(view);
        return nullptr;
    }
    if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(view), capsule) < 0) {
        Py_DECREF(capsule);
        Py_DECREF(view);
        return nullptr;
    }
    return view;
}

class IterReduceKernel : public Kernel {
  public:
    IterReduceKernel(NpyIter* iter, std::unique_ptr<UfuncLoop> loop, DeferRelease defer)
        : iter_(iter), loop_(std::move(loop)), defer_(std::move(defer)) {}

    ~IterReduceKernel() override {
        NpyIter* iter = iter_;
        UfuncLoop* loop = loop_.release();
        defer_([iter, loop] {
            NpyIter_Deallocate(iter); // releases the operand view and the result
            delete loop;              // releases the call info capsule
        });
    }

    KernelResult run(std::span<const Operand>, std::span<const Operand>) override {
        // Initial values (PyUFunc_ReduceWrapper): the identity everywhere, or
        // the first elements along the reduced axes.
        if (!initial_.empty()) {
            for (npy_intp i = 0; i < result_size_; ++i) {
                std::memcpy(result_data_ + i * itemsize_, initial_.data(), itemsize_);
            }
        } else {
            copy_first();
        }
        char* errmsg = nullptr;
        if (!NpyIter_Reset(iter_, &errmsg)) { // no GIL: errmsg variant
            return {1, errmsg != nullptr ? errmsg : "NpyIter_Reset failed", 0};
        }
        return reduce_loop() ? KernelResult{} : KernelResult{1, "NumPy reduction loop failed", 0};
    }

    // Set up by make_iter_reduction().
    NpyIter_IterNextFunc* iternext_ = nullptr;
    char** dataptr_ = nullptr;
    npy_intp* strideptr_ = nullptr;
    npy_intp* countptr_ = nullptr;
    std::vector<std::byte> initial_; // empty: start at the first elements
    char* result_data_ = nullptr;
    npy_intp result_size_ = 0;
    npy_intp itemsize_ = 0;
    npy_intp skip_first_count_ = 0;
    // First-element copy: kept input axes (shape, input stride, result stride).
    char* operand_data_ = nullptr;
    std::vector<std::array<npy_intp, 3>> kept_;

  private:
    void copy_first() const {
        const std::size_t n = kept_.size();
        std::vector<npy_intp> idx(n, 0);
        char* in = operand_data_;
        char* out = result_data_;
        for (;;) {
            std::memcpy(out, in, static_cast<std::size_t>(itemsize_));
            std::size_t k = n;
            for (;;) {
                if (k == 0) {
                    return;
                }
                --k;
                if (++idx[k] < kept_[k][0]) {
                    in += kept_[k][1];
                    out += kept_[k][2];
                    break;
                }
                in -= (kept_[k][0] - 1) * kept_[k][1];
                out -= (kept_[k][0] - 1) * kept_[k][2];
                idx[k] = 0;
            }
        }
    }

    // NumPy's reduce_loop (ufunc_object.c), unmasked.
    bool reduce_loop() {
        npy_intp skip = skip_first_count_;
        char* data[3];
        npy_intp strides[3];
        if (skip > 0) {
            for (;;) {
                npy_intp count = *countptr_;
                char* in = dataptr_[1];
                if (NpyIter_IsFirstVisit(iter_, 0)) {
                    if (strideptr_[0] == 0) {
                        --count;
                        --skip;
                        in += strideptr_[1];
                    } else {
                        skip -= count;
                        count = 0;
                    }
                }
                if (count > 0) {
                    data[0] = dataptr_[0];
                    data[1] = in;
                    data[2] = dataptr_[0];
                    strides[0] = strideptr_[0];
                    strides[1] = strideptr_[1];
                    strides[2] = strideptr_[0];
                    if (loop_->call(data, &count, strides) < 0) {
                        return false;
                    }
                }
                if (!iternext_(iter_)) {
                    return true;
                }
                if (skip == 0) {
                    break;
                }
            }
        }
        do {
            data[0] = dataptr_[0];
            data[1] = dataptr_[1];
            data[2] = dataptr_[0];
            strides[0] = strideptr_[0];
            strides[1] = strideptr_[1];
            strides[2] = strideptr_[0];
            if (loop_->call(data, countptr_, strides) < 0) {
                return false;
            }
        } while (iternext_(iter_));
        return true;
    }

    NpyIter* iter_;
    std::unique_ptr<UfuncLoop> loop_;
    DeferRelease defer_;
};

} // namespace

bool make_iter_reduction(PyObject* ufunc, const Layout& in, std::shared_ptr<Buffer> in_buffer,
                         std::span<const bool> reduced, bool keepdims, PyObject* dtype,
                         PyObject* initial, long buffersize, DeferRelease defer, IterReduction& out,
                         std::string& error) {
    const int nd = static_cast<int>(in.shape.size());
    if (in.size() == 0) {
        error = "empty reduction";
        return false;
    }
    PyObject* operand = operand_view(in, in_buffer);
    if (operand == nullptr) {
        error = fetch_error();
        return false;
    }
    PyObject* in_descr =
        reinterpret_cast<PyObject*>(PyArray_DESCR(reinterpret_cast<PyArrayObject*>(operand)));
    PyObject* info = resolve_reduction_info(ufunc, in_descr, dtype, error);
    if (info == nullptr) {
        Py_DECREF(operand);
        return false;
    }
    PyObject* descrs = PyTuple_GetItem(info, 0);
    PyObject* capsule = PyTuple_GetItem(info, 1);
    auto* descr0 = reinterpret_cast<PyArray_Descr*>(PyTuple_GetItem(descrs, 0));
    auto* descr1 = reinterpret_cast<PyArray_Descr*>(PyTuple_GetItem(descrs, 1));

    // The iterator, as PyUFunc_ReduceWrapper sets it up.
    int result_axes[NPY_MAXDIMS];
    int curr = 0;
    for (int i = 0; i < nd; ++i) {
        if (reduced[static_cast<std::size_t>(i)]) {
            result_axes[i] = keepdims ? reduction_axis(curr++) : reduction_axis(-1);
        } else {
            result_axes[i] = curr++;
        }
    }
    int* op_axes[2] = {result_axes, nullptr};
    PyArrayObject* op[2] = {nullptr, reinterpret_cast<PyArrayObject*>(operand)};
    PyArray_Descr* op_dtypes[2] = {descr0, descr1};
    npy_uint32 op_flags[2] = {NPY_ITER_READWRITE | NPY_ITER_ALIGNED | NPY_ITER_ALLOCATE |
                                  NPY_ITER_NO_SUBTYPE,
                              NPY_ITER_READONLY | NPY_ITER_ALIGNED | NPY_ITER_NO_BROADCAST};
    const npy_uint32 flags = NPY_ITER_BUFFERED | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_GROWINNER |
                             NPY_ITER_ZEROSIZE_OK | NPY_ITER_REFS_OK | NPY_ITER_DELAY_BUFALLOC |
                             NPY_ITER_DONT_NEGATE_STRIDES | NPY_ITER_COPY_IF_OVERLAP;
    NpyIter* iter =
        NpyIter_AdvancedNew(2, op, flags, NPY_KEEPORDER, NPY_UNSAFE_CASTING, op_flags, op_dtypes,
                            nd, op_axes, nullptr, static_cast<npy_intp>(buffersize));
    Py_DECREF(operand); // the iterator holds it
    if (iter == nullptr) {
        error = fetch_error();
        Py_DECREF(info);
        return false;
    }
    auto fail = [&](const std::string& why) {
        error = why;
        PyErr_Clear();
        NpyIter_Deallocate(iter);
        Py_DECREF(info);
        return false;
    };
    if (NpyIter_GetIterSize(iter) == 0) {
        return fail("empty reduction");
    }
    PyArrayObject* result = NpyIter_GetOperandArray(iter)[0];
    npy_intp* strideptr = NpyIter_GetInnerStrideArray(iter);
    const std::intptr_t fixed[3] = {strideptr[0], strideptr[1], strideptr[0]};
    std::unique_ptr<UfuncLoop> loop = UfuncLoop::from_call_info(ufunc, capsule, fixed, error);
    Py_DECREF(info);
    if (!loop) {
        PyErr_Clear();
        NpyIter_Deallocate(iter);
        return false;
    }
    auto kernel = std::make_shared<IterReduceKernel>(iter, std::move(loop), std::move(defer));
    kernel->iternext_ = NpyIter_GetIterNext(iter, nullptr);
    kernel->dataptr_ = NpyIter_GetDataPtrArray(iter);
    kernel->strideptr_ = strideptr;
    kernel->countptr_ = NpyIter_GetInnerLoopSizePtr(iter);
    kernel->result_data_ = PyArray_BYTES(result);
    kernel->result_size_ = PyArray_SIZE(result);
    kernel->itemsize_ = PyArray_ITEMSIZE(result);
    if (kernel->iternext_ == nullptr) {
        error = fetch_error();
        return false; // the kernel releases the iterator
    }
    if (initial != Py_None) {
        Py_INCREF(descr0);
        PyObject* v = PyArray_FromAny(initial, descr0, 0, 0, 0, nullptr); // steals descr0
        if (v == nullptr) {
            error = fetch_error();
            return false;
        }
        const auto* p =
            static_cast<const std::byte*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(v)));
        kernel->initial_.assign(p, p + kernel->itemsize_);
        Py_DECREF(v);
    } else {
        // First elements copied as they are: only without a cast.
        if (descr0->type_num != type_num(in.dtype) || descr1->type_num != type_num(in.dtype)) {
            error = "a reduction without initial value that casts";
            return false;
        }
        kernel->operand_data_ = reinterpret_cast<char*>(in_buffer->data() + in.offset);
        int j = 0;
        for (int i = 0; i < nd; ++i) {
            if (!reduced[static_cast<std::size_t>(i)]) {
                kernel->kept_.push_back({in.shape[static_cast<std::size_t>(i)],
                                         in.strides[static_cast<std::size_t>(i)],
                                         PyArray_STRIDES(result)[j]});
                ++j;
            } else if (keepdims) {
                ++j;
            }
        }
        kernel->skip_first_count_ = kernel->result_size_;
    }
    Py_INCREF(result);
    out.result = reinterpret_cast<PyObject*>(result);
    out.kernel = std::move(kernel);
    return true;
}

} // namespace viproc::numpy
