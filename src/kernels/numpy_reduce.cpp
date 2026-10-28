#include "kernels/numpy_reduce.hpp"

#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL viproc_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL viproc_UFUNC_API
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#include "kernels/numpy_iter.hpp"

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
    const char* name() const override { return loop_->name(); }
    const char* fp_name() const override { return loop_->reports_fp_errors() ? "reduce" : nullptr; }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Layout& out_layout = outputs[0].layout();
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

// Calls fn(in, out) once per lane along `axis`: per position of the other
// axes of `il` and `ol` (same extents there; any order, lanes are
// independent), with pointers to the lane's first elements. Stops at the
// first fn returning false and returns false then.
template <class F>
bool for_each_lane(char* in, const Layout& il, char* out, const Layout& ol, std::size_t axis,
                   F fn) {
    const std::size_t nd = il.shape.size();
    SmallVector<std::int64_t, kInlineRank> idx(nd, 0);
    for (;;) {
        if (!fn(in, out)) {
            return false;
        }
        std::size_t k = nd;
        for (;;) {
            if (k == 0) {
                return true;
            }
            --k;
            if (k == axis) {
                continue;
            }
            if (++idx[k] < il.shape[k]) {
                in += il.strides[k];
                out += ol.strides[k];
                break;
            }
            in -= (il.shape[k] - 1) * il.strides[k];
            out -= (ol.shape[k] - 1) * ol.strides[k];
            idx[k] = 0;
        }
    }
}

// "add.reduce" (the reduction loop's name) -> "add.<method>"
std::string method_name(std::string n, const char* method) {
    const std::size_t p = n.rfind(".reduce");
    return p == std::string::npos ? n + "." + method : n.replace(p, 7, std::string(".") + method);
}

class AccumulateKernel : public Kernel {
  public:
    AccumulateKernel(const UfuncLoop* loop, int axis) : loop_(loop), axis_(axis) {}
    const char* name() const override { return name_.c_str(); }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Layout& il = inputs[0].layout();
        const Layout& ol = outputs[0].layout();
        if (ol.size() == 0) {
            return {};
        }
        const std::size_t isz = itemsize(ol.dtype);
        const auto ax = static_cast<std::size_t>(axis_);
        const auto n = static_cast<std::intptr_t>(il.shape[ax]);
        const auto sin = static_cast<std::intptr_t>(il.strides[ax]);
        const auto sout = static_cast<std::intptr_t>(ol.strides[ax]);
        const std::intptr_t strides[3] = {sout, sin, sout};
        const bool ok = for_each_lane(reinterpret_cast<char*>(inputs[0].first()), il,
                                      reinterpret_cast<char*>(outputs[0].first()), ol, ax,
                                      [&](char* in, char* out) {
                                          std::memmove(out, in, isz);
                                          if (n <= 1) {
                                              return true;
                                          }
                                          std::intptr_t count = n - 1;
                                          char* data[3] = {out, in + sin, out + sout};
                                          return loop_->call(data, &count, strides) == 0;
                                      });
        return ok ? KernelResult{} : KernelResult{1, take_loop_error(), 0};
    }

  private:
    const UfuncLoop* loop_;
    int axis_;
    std::string name_ = method_name(loop_->name(), "accumulate");

  public:
    const char* fp_name() const override {
        return loop_->reports_fp_errors() ? "accumulate" : nullptr;
    }
};

class ReduceatKernel : public Kernel {
  public:
    ReduceatKernel(const UfuncLoop* loop, int axis, std::vector<std::int64_t> indices)
        : loop_(loop), axis_(axis), indices_(std::move(indices)) {}
    const char* name() const override { return name_.c_str(); }
    const char* fp_name() const override {
        return loop_->reports_fp_errors() ? "reduceat" : nullptr;
    }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Layout& il = inputs[0].layout();
        const Layout& ol = outputs[0].layout();
        if (ol.size() == 0) {
            return {};
        }
        const std::size_t isz = itemsize(ol.dtype);
        const auto ax = static_cast<std::size_t>(axis_);
        const std::int64_t n = il.shape[ax];
        const auto sin = static_cast<std::intptr_t>(il.strides[ax]);
        const auto sout = static_cast<std::intptr_t>(ol.strides[ax]);
        // the accumulator does not advance: the loop reduces (pairwise sums)
        const std::intptr_t strides[3] = {0, sin, 0};
        const std::size_t m = indices_.size();
        const bool ok = for_each_lane(
            reinterpret_cast<char*>(inputs[0].first()), il,
            reinterpret_cast<char*>(outputs[0].first()), ol, ax, [&](char* in, char* out) {
                for (std::size_t i = 0; i < m; ++i) {
                    const std::int64_t start = indices_[i];
                    const std::int64_t end = i + 1 == m ? n : indices_[i + 1];
                    char* o = out + static_cast<std::intptr_t>(i) * sout;
                    char* first = in + static_cast<std::intptr_t>(start) * sin;
                    std::memmove(o, first, isz);
                    if (end - start > 1) {
                        std::intptr_t count = static_cast<std::intptr_t>(end - start - 1);
                        char* data[3] = {o, first + sin, o};
                        if (loop_->call(data, &count, strides) != 0) {
                            return false;
                        }
                    }
                }
                return true;
            });
        return ok ? KernelResult{} : KernelResult{1, take_loop_error(), 0};
    }

  private:
    const UfuncLoop* loop_;
    int axis_;
    std::vector<std::int64_t> indices_;
    std::string name_ = method_name(loop_->name(), "reduceat");
};

} // namespace

std::shared_ptr<Kernel> make_accumulate_kernel(const UfuncLoop* loop, int axis) {
    return std::make_shared<AccumulateKernel>(loop, axis);
}

std::shared_ptr<Kernel> make_reduceat_kernel(const UfuncLoop* loop, int axis,
                                             std::vector<std::int64_t> indices) {
    return std::make_shared<ReduceatKernel>(loop, axis, std::move(indices));
}

namespace {

// op_axes entry of a reduction axis (NumPy's NPY_ITER_REDUCTION_AXIS, which
// its public headers do not export; the encoding is part of the nditer API).
constexpr int reduction_axis(int axis) { return axis + (1 << (NPY_BITSOF_INT - 2)); }

// Reduction templates kept per LoopCache (all dropped when full).
constexpr std::size_t kMaxReduceTemplates = 1024;
// Largest itemsize of a viproc dtype (complex128).
constexpr std::size_t kMaxItemsize = 16;

// Start values converted once per object: immutable ones only (a 0-d array
// passed as initial= could change between calls).
bool immutable_scalar(PyObject* v) {
    return PyLong_CheckExact(v) || PyFloat_CheckExact(v) || PyBool_Check(v) ||
           PyComplex_CheckExact(v) || PyArray_IsScalar(v, Generic);
}

} // namespace

struct ReduceTemplate {
    std::shared_ptr<IterTemplate> iters;
    std::unique_ptr<UfuncLoop> loop; // for the template's inner strides
    DeferRelease defer;
    PyObject* dtype = nullptr;        // dtype= of the key (owned: keeps its address)
    PyArray_Descr* descr0 = nullptr;  // the accumulator's descriptor (owned)
    PyObject* last_initial = nullptr; // owned; its bytes in initial_bytes
    std::array<std::byte, kMaxItemsize> initial_bytes{};
    Layout result; // NumPy's result layout (offset 0)
    npy_intp result_size = 0;
    npy_intp itemsize = 0;
    bool has_initial = false;
    // First-element copy (no initial): kept input axes (shape, input stride,
    // result stride).
    std::vector<std::array<npy_intp, 3>> kept;

    ~ReduceTemplate() {
        PyObject* objs[3] = {dtype, reinterpret_cast<PyObject*>(descr0), last_initial};
        UfuncLoop* l = loop.release();
        defer([objs, l] {
            for (PyObject* o : objs) {
                Py_XDECREF(o);
            }
            delete l; // releases the call info capsule
        });
    }
};

namespace {

// NumPy's reduction over a copy of a template iterator: initial values as
// PyUFunc_ReduceWrapper sets them, then NumPy's reduce_loop over the
// iterator's blocks (casts happen in its buffer copies). Runs without the
// GIL.
class IterReduceKernel : public Kernel {
  public:
    explicit IterReduceKernel(std::shared_ptr<ReduceTemplate> t) : t_(std::move(t)) {}

    const char* name() const override { return t_->loop->name(); }
    const char* fp_name() const override {
        return t_->loop->reports_fp_errors() ? "reduce" : nullptr;
    }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        char* const in = reinterpret_cast<char*>(inputs[0].first());
        char* const out = reinterpret_cast<char*>(outputs[0].first());
        const npy_intp isz = t_->itemsize;
        if (t_->has_initial) {
            for (npy_intp i = 0; i < t_->result_size; ++i) {
                std::memcpy(out + i * isz, initial_.data(), static_cast<std::size_t>(isz));
            }
        } else {
            copy_first(in, out);
        }
        char* base[2] = {out, in};
        PooledIter it;
        std::string error;
        if (!t_->iters->acquire(it, error)) {
            return {1, std::move(error), 0};
        }
        char* errmsg = nullptr;
        KernelResult r;
        // Points the iterator at the buffers (no GIL: errmsg variant); with
        // buffering it also copies the first block.
        if (NpyIter_ResetBasePointers(it.iter, base, &errmsg) != NPY_SUCCEED) {
            r = {1, errmsg != nullptr ? errmsg : "NpyIter_ResetBasePointers failed", 0};
        } else if (!reduce_loop(it)) {
            r = {1, "NumPy reduction loop failed", 0};
        }
        t_->iters->give_back(it, r.status == 0);
        return r;
    }

    // Set up by LoopCache::iter_reduction().
    std::array<std::byte, kMaxItemsize> initial_{};

  private:
    void copy_first(char* in, char* out) const {
        const auto& kept = t_->kept;
        const auto isz = static_cast<std::size_t>(t_->itemsize);
        const std::size_t n = kept.size();
        SmallVector<npy_intp, kInlineRank> idx(n, 0);
        for (;;) {
            std::memcpy(out, in, isz);
            std::size_t k = n;
            for (;;) {
                if (k == 0) {
                    return;
                }
                --k;
                if (++idx[k] < kept[k][0]) {
                    in += kept[k][1];
                    out += kept[k][2];
                    break;
                }
                in -= (kept[k][0] - 1) * kept[k][1];
                out -= (kept[k][0] - 1) * kept[k][2];
                idx[k] = 0;
            }
        }
    }

    // NumPy's reduce_loop (ufunc_object.c), unmasked.
    bool reduce_loop(const PooledIter& it) {
        char** const dataptr_ = it.dataptr;
        npy_intp* const strideptr_ = it.strideptr;
        npy_intp* const countptr_ = it.countptr;
        const UfuncLoop* loop = t_->loop.get();
        npy_intp skip = t_->has_initial ? 0 : t_->result_size;
        char* data[3];
        npy_intp strides[3];
        if (skip > 0) {
            for (;;) {
                npy_intp count = *countptr_;
                char* in = dataptr_[1];
                if (NpyIter_IsFirstVisit(it.iter, 0)) {
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
                    if (loop->call(data, &count, strides) < 0) {
                        return false;
                    }
                }
                if (!it.iternext(it.iter)) {
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
            if (loop->call(data, countptr_, strides) < 0) {
                return false;
            }
        } while (it.iternext(it.iter));
        return true;
    }

    std::shared_ptr<ReduceTemplate> t_;
};

// Builds the template of a reduction (see LoopCache::iter_reduction()).
std::shared_ptr<ReduceTemplate> build_reduce_template(PyObject* ufunc, const Layout& in,
                                                      std::span<const bool> reduced, bool keepdims,
                                                      PyObject* dtype, bool has_initial,
                                                      long buffersize, DeferRelease defer,
                                                      std::string& error) {
    const int nd = static_cast<int>(in.shape.size());
    PyArrayObject* operand = template_array(in, template_address(), false);
    if (operand == nullptr) {
        error = fetch_python_error();
        return nullptr;
    }
    PyObject* in_descr = reinterpret_cast<PyObject*>(PyArray_DESCR(operand));
    PyObject* info = resolve_reduction_info(ufunc, in_descr, dtype, error);
    if (info == nullptr) {
        Py_DECREF(operand);
        return nullptr;
    }
    PyObject* descrs = PyTuple_GetItem(info, 0);
    PyObject* capsule = PyTuple_GetItem(info, 1);
    auto* descr0 = reinterpret_cast<PyArray_Descr*>(PyTuple_GetItem(descrs, 0));
    auto* descr1 = reinterpret_cast<PyArray_Descr*>(PyTuple_GetItem(descrs, 1));

    // The iterator, as PyUFunc_ReduceWrapper sets it up, without
    // COPY_IF_OVERLAP: the template's operands live at a dummy address, and
    // viproc's result never overlaps the input. NumPy allocates the
    // template's result; its layout is the one viproc gives its result.
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
    PyArrayObject* op[2] = {nullptr, operand};
    PyArray_Descr* op_dtypes[2] = {descr0, descr1};
    npy_uint32 op_flags[2] = {NPY_ITER_READWRITE | NPY_ITER_ALIGNED | NPY_ITER_ALLOCATE |
                                  NPY_ITER_NO_SUBTYPE,
                              NPY_ITER_READONLY | NPY_ITER_ALIGNED | NPY_ITER_NO_BROADCAST};
    const npy_uint32 flags = NPY_ITER_BUFFERED | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_GROWINNER |
                             NPY_ITER_ZEROSIZE_OK | NPY_ITER_REFS_OK | NPY_ITER_DELAY_BUFALLOC |
                             NPY_ITER_DONT_NEGATE_STRIDES;
    NpyIter* iter =
        NpyIter_AdvancedNew(2, op, flags, NPY_KEEPORDER, NPY_UNSAFE_CASTING, op_flags, op_dtypes,
                            nd, op_axes, nullptr, static_cast<npy_intp>(buffersize));
    Py_DECREF(operand); // the iterator holds it
    if (iter == nullptr) {
        error = fetch_python_error();
        Py_DECREF(info);
        return nullptr;
    }
    auto t = std::make_shared<ReduceTemplate>();
    t->defer = defer;
    t->iters = new_iter_template();
    t->iters->templ = iter;
    t->iters->defer = std::move(defer);
    auto fail = [&](const char* why) {
        error = why;
        PyErr_Clear();
        Py_DECREF(info);
        return nullptr;
    };
    if (NpyIter_GetIterSize(iter) == 0) {
        return fail("empty reduction");
    }
    if (NpyIter_GetOperandArray(iter)[1] != operand) {
        // NumPy cast the input into a copy at construction (it does so for
        // 0-d inputs): the copy, not the input, would be read.
        return fail("the iterator copies the input");
    }
    if (NpyIter_IterationNeedsAPI(iter)) {
        return fail("iteration needs the Python API");
    }
    if (!has_initial &&
        (descr0->type_num != type_num(in.dtype) || descr1->type_num != type_num(in.dtype))) {
        // First elements are copied as they are: only without a cast.
        return fail("a reduction without initial value that casts");
    }
    PyArrayObject* result = NpyIter_GetOperandArray(iter)[0];
    void* data = nullptr;
    std::size_t nbytes = 0;
    if (!ndarray_layout(reinterpret_cast<PyObject*>(result), t->result, data, nbytes) ||
        t->result.offset != 0 || PyArray_ITEMSIZE(result) > static_cast<npy_intp>(kMaxItemsize)) {
        return fail("unsupported result dtype");
    }
    t->result_size = PyArray_SIZE(result);
    t->itemsize = PyArray_ITEMSIZE(result);
    npy_intp* strideptr = NpyIter_GetInnerStrideArray(iter);
    const std::intptr_t fixed[3] = {strideptr[0], strideptr[1], strideptr[0]};
    t->loop = UfuncLoop::from_call_info(ufunc, capsule, fixed, error);
    if (!t->loop) {
        PyErr_Clear();
        Py_DECREF(info);
        return nullptr;
    }
    Py_INCREF(descr0);
    t->descr0 = descr0;
    Py_DECREF(info);
    t->has_initial = has_initial;
    if (!has_initial) {
        int j = 0;
        for (int i = 0; i < nd; ++i) {
            if (!reduced[static_cast<std::size_t>(i)]) {
                t->kept.push_back({in.shape[static_cast<std::size_t>(i)],
                                   in.strides[static_cast<std::size_t>(i)],
                                   PyArray_STRIDES(result)[j]});
                ++j;
            } else if (keepdims) {
                ++j;
            }
        }
    }
    Py_INCREF(dtype);
    t->dtype = dtype;
    return t;
}

} // namespace

std::shared_ptr<Kernel> LoopCache::iter_reduction(PyObject* ufunc, const Layout& in,
                                                  std::span<const bool> reduced, bool keepdims,
                                                  PyObject* dtype, PyObject* initial,
                                                  DeferRelease defer, Layout& result,
                                                  std::string& error) {
    if (in.size() == 0) {
        error = "empty reduction";
        return nullptr;
    }
    const bool has_initial = initial != Py_None;
    const long bufsize = buffersize();
    std::string key;
    auto put = [&key](std::int64_t v) { key.append(reinterpret_cast<const char*>(&v), sizeof v); };
    put(reinterpret_cast<std::intptr_t>(ufunc));
    put(reinterpret_cast<std::intptr_t>(dtype));
    put((has_initial ? 1 : 0) | (keepdims ? 2 : 0));
    put(bufsize);
    put(static_cast<std::int64_t>(in.dtype));
    put(static_cast<std::int64_t>(in.shape.size()));
    for (std::size_t d = 0; d < in.shape.size(); ++d) {
        put(in.shape[d]);
        put(in.strides[d]);
        put(reduced[d] ? 1 : 0);
    }
    std::shared_ptr<ReduceTemplate> t;
    if (auto it = reduce_templates_.find(key); it != reduce_templates_.end()) {
        t = it->second;
    } else {
        if (in.shape.empty()) {
            // NumPy casts a 0-d input into a copy when it builds the
            // iterator (not in its buffers), which a template cannot reuse.
            // The same iteration over a 1-element 1-d input buffers it; the
            // result is that one element either way.
            Layout in1 = in;
            in1.shape = {1};
            in1.strides = {static_cast<std::int64_t>(itemsize(in.dtype))};
            const bool kept[1] = {false};
            t = build_reduce_template(ufunc, in1, kept, false, dtype, has_initial, bufsize,
                                      std::move(defer), error);
            if (t) {
                t->result.shape.clear();
                t->result.strides.clear();
            }
        } else {
            t = build_reduce_template(ufunc, in, reduced, keepdims, dtype, has_initial, bufsize,
                                      std::move(defer), error);
        }
        if (!t) {
            return nullptr;
        }
        if (reduce_templates_.size() >= kMaxReduceTemplates) {
            reduce_templates_.clear();
        }
        reduce_templates_.emplace(std::move(key), t);
    }
    // The start value in the accumulator's dtype, converted by NumPy.
    if (has_initial && initial != t->last_initial) {
        Py_INCREF(t->descr0);
        PyObject* v = PyArray_FromAny(initial, t->descr0, 0, 0, 0, nullptr); // steals descr0
        if (v == nullptr) {
            error = fetch_python_error();
            return nullptr;
        }
        std::memcpy(t->initial_bytes.data(), PyArray_DATA(reinterpret_cast<PyArrayObject*>(v)),
                    static_cast<std::size_t>(t->itemsize));
        Py_DECREF(v);
        Py_XDECREF(t->last_initial);
        t->last_initial = nullptr;
        if (immutable_scalar(initial)) {
            Py_INCREF(initial);
            t->last_initial = initial;
        }
    }
    if (!t->iters->reserve()) {
        error = fetch_python_error();
        return nullptr;
    }
    auto kernel = std::make_shared<IterReduceKernel>(t);
    kernel->initial_ = t->initial_bytes;
    result = t->result;
    return kernel;
}

namespace {

class ArgReduceKernel : public Kernel {
  public:
    ArgReduceKernel(PyArray_ArgFunc* fn, bool argmax) : fn_(fn), argmax_(argmax) {}
    const char* name() const override { return argmax_ ? "numpy.argmax" : "numpy.argmin"; }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Layout& l = inputs[0].layout();
        const npy_intp m = l.shape.empty() ? 1 : static_cast<npy_intp>(l.shape.back());
        const npy_intp rows = m == 0 ? 0 : static_cast<npy_intp>(l.size()) / m;
        const auto step = static_cast<npy_intp>(itemsize(l.dtype)) * m;
        char* in = reinterpret_cast<char*>(inputs[0].first());
        auto* out = reinterpret_cast<npy_intp*>(outputs[0].first());
        for (npy_intp r = 0; r < rows; ++r) {
            // The last argument (the array) is unused for builtin dtypes.
            if (fn_(in + r * step, m, out + r, nullptr) < 0) {
                return {1, "NumPy argmin/argmax failed", 0};
            }
        }
        return {};
    }

  private:
    PyArray_ArgFunc* fn_;
    bool argmax_;
};

} // namespace

std::shared_ptr<Kernel> make_arg_reduce_kernel(DType dtype, bool argmax, std::string& error) {
    PyArray_Descr* descr = PyArray_DescrFromType(type_num(dtype));
    PyArray_ArgFunc* fn =
        argmax ? PyDataType_GetArrFuncs(descr)->argmax : PyDataType_GetArrFuncs(descr)->argmin;
    Py_DECREF(descr);
    if (fn == nullptr) {
        error = "no argmin/argmax function for this dtype";
        return nullptr;
    }
    return std::make_shared<ArgReduceKernel>(fn, argmax);
}

} // namespace viproc::numpy
