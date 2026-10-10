#include "kernels/numpy_ufunc.hpp"

#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL viproc_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL viproc_UFUNC_API
#include <numpy/arrayobject.h>
#include <numpy/dtype_api.h>
#include <numpy/ufuncobject.h>

#include <cstring>

namespace viproc::numpy {

namespace {

// Prefix of NumPy's `ufunc_call_info` (capsule "numpy_1.24_ufunc_call_info").
// Only the leading fields are mirrored; see "Calling NumPy kernels" in CLAUDE.md.
struct UfuncCallInfoPrefix {
    PyArrayMethod_StridedLoop* strided_loop;
    PyArrayMethod_Context* context;
    NpyAuxData* auxdata;
    npy_bool requires_pyapi;
    npy_bool no_floatingpoint_errors;
};

constexpr const char* kCallInfoCapsule = "numpy_1.24_ufunc_call_info";

int init_numpy_api() {
    import_array1(-1);
    import_umath1(-1);
    return 0;
}

std::string fetch_python_error() {
    PyObject* exc = PyErr_GetRaisedException();
    if (exc == nullptr) {
        return "unknown error";
    }
    PyObject* s = PyObject_Str(exc);
    std::string msg = s != nullptr ? PyUnicode_AsUTF8(s) : "unknown error";
    Py_XDECREF(s);
    Py_DECREF(exc);
    return msg;
}

bool dtype_of(int type_num, DType& out) {
    switch (type_num) {
    case NPY_BOOL:
        out = DType::Bool;
        return true;
    case NPY_INT32:
        out = DType::Int32;
        return true;
    case NPY_INT64:
        out = DType::Int64;
        return true;
    case NPY_FLOAT32:
        out = DType::Float32;
        return true;
    case NPY_FLOAT64:
        out = DType::Float64;
        return true;
    case NPY_COMPLEX64:
        out = DType::Complex64;
        return true;
    case NPY_COMPLEX128:
        out = DType::Complex128;
        return true;
    default:
        return false;
    }
}

} // namespace

bool init() { return init_numpy_api() == 0; }

int type_num(DType dtype) {
    switch (dtype) {
    case DType::Bool:
        return NPY_BOOL;
    case DType::Int32:
        return NPY_INT32;
    case DType::Int64:
        return NPY_INT64;
    case DType::Float32:
        return NPY_FLOAT32;
    case DType::Float64:
        return NPY_FLOAT64;
    case DType::Complex64:
        return NPY_COMPLEX64;
    case DType::Complex128:
        return NPY_COMPLEX128;
    }
    return NPY_NOTYPE;
}

UfuncLoop::~UfuncLoop() { Py_XDECREF(capsule_); }

std::unique_ptr<UfuncLoop> UfuncLoop::resolve(PyObject* ufunc, std::span<const DType> dtypes,
                                              std::string& error) {
    auto* uf = reinterpret_cast<PyUFuncObject*>(ufunc);
    if (!PyObject_TypeCheck(ufunc, &PyUFunc_Type) ||
        static_cast<std::size_t>(uf->nargs) != dtypes.size()) {
        error = "not a ufunc, or wrong number of dtypes";
        return nullptr;
    }
    PyObject* requested = PyTuple_New(static_cast<Py_ssize_t>(dtypes.size()));
    for (std::size_t i = 0; i < dtypes.size(); ++i) {
        PyTuple_SET_ITEM(requested, static_cast<Py_ssize_t>(i),
                         reinterpret_cast<PyObject*>(PyArray_DescrFromType(type_num(dtypes[i]))));
    }
    PyObject* name = PyUnicode_FromString("_resolve_dtypes_and_context");
    PyObject* res = PyObject_CallMethodOneArg(ufunc, name, requested);
    Py_DECREF(name);
    if (res == nullptr) {
        Py_DECREF(requested);
        error = fetch_python_error();
        return nullptr;
    }
    PyObject* resolved = PyTuple_GetItem(res, 0);
    bool exact = true;
    for (std::size_t i = 0; i < dtypes.size(); ++i) {
        auto* d =
            reinterpret_cast<PyArray_Descr*>(PyTuple_GetItem(resolved, static_cast<Py_ssize_t>(i)));
        exact = exact && d->type_num == type_num(dtypes[i]);
    }
    Py_DECREF(requested);
    if (!exact) {
        Py_DECREF(res);
        error = "NumPy resolves a different dtype signature (casting not supported yet)";
        return nullptr;
    }
    std::unique_ptr<UfuncLoop> loop(new UfuncLoop());
    loop->capsule_ = PyTuple_GetItem(res, 1);
    Py_INCREF(loop->capsule_);
    Py_DECREF(res);

    name = PyUnicode_FromString("_get_strided_loop");
    PyObject* filled = PyObject_CallMethodOneArg(ufunc, name, loop->capsule_);
    Py_DECREF(name);
    if (filled == nullptr) {
        error = fetch_python_error();
        return nullptr;
    }
    Py_DECREF(filled);
    loop->info_ = PyCapsule_GetPointer(loop->capsule_, kCallInfoCapsule);
    if (loop->info_ == nullptr) {
        error = fetch_python_error();
        return nullptr;
    }
    loop->nin_ = uf->nin;
    loop->nout_ = uf->nout;
    if (loop->requires_pyapi()) {
        error = "loop requires the Python API; cannot run asynchronously";
        return nullptr;
    }
    loop->kernel_ = make_elementwise_kernel(loop.get());
    return loop;
}

bool UfuncLoop::requires_pyapi() const {
    return static_cast<const UfuncCallInfoPrefix*>(info_)->requires_pyapi != 0;
}

int UfuncLoop::call(char* const* data, const std::intptr_t* dims,
                    const std::intptr_t* strides) const {
    const auto* info = static_cast<const UfuncCallInfoPrefix*>(info_);
    return info->strided_loop(info->context, data, dims, strides, info->auxdata);
}

const UfuncLoop* LoopCache::get(PyObject* ufunc, std::span<const DType> dtypes,
                                std::string& error) {
    auto key = std::make_pair(ufunc, std::vector<DType>(dtypes.begin(), dtypes.end()));
    auto it = loops_.find(key);
    if (it != loops_.end()) {
        return it->second.get();
    }
    std::unique_ptr<UfuncLoop> loop = UfuncLoop::resolve(ufunc, dtypes, error);
    if (!loop) {
        return nullptr;
    }
    Py_INCREF(ufunc); // keep the key valid; released never (ufuncs are module globals)
    return loops_.emplace(std::move(key), std::move(loop)).first->second.get();
}

namespace {

class ElementwiseKernel : public Kernel {
  public:
    explicit ElementwiseKernel(const UfuncLoop* loop) : loop_(loop) {}

    // Only the first nin() inputs go to the loop; further inputs are in/out
    // arrays the runtime lists as read (see Runtime::issue).
    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Shape& shape = outputs[0].layout.shape;
        const auto nin = static_cast<std::size_t>(loop_->nin());
        if (nin + outputs.size() <= kMaxFlat) {
            // Fast path: every operand covers `shape` C-contiguously or is a
            // single element (stride 0). Then one loop call covers all
            // elements, as in NumPy (element-wise results do not depend on
            // how the elements are split into calls).
            const std::int64_t n = outputs[0].layout.size();
            char* ptrs[kMaxFlat];
            std::intptr_t strides[kMaxFlat];
            std::size_t k = 0;
            auto flat = [&](const Operand& op) {
                const Layout& l = op.layout;
                ptrs[k] = reinterpret_cast<char*>(op.first());
                if (l.shape == shape && l.c_contiguous()) {
                    strides[k++] = static_cast<std::intptr_t>(itemsize(l.dtype));
                    return true;
                }
                strides[k++] = 0;
                return l.size() == 1;
            };
            bool ok = true;
            for (const Operand& op : inputs.first(nin)) {
                ok = ok && flat(op);
            }
            for (const Operand& op : outputs) {
                ok = ok && flat(op) && strides[k - 1] != 0; // a broadcast output is not flat
            }
            if (ok) {
                if (n == 0) {
                    return {};
                }
                const auto count = static_cast<std::intptr_t>(n);
                return loop_->call(ptrs, &count, strides) == 0
                           ? KernelResult{}
                           : KernelResult{1, "NumPy loop failed", 0};
            }
        }
        std::vector<Layout> layouts;
        layouts.reserve(nin + outputs.size());
        OperandPointers base;
        for (const Operand& op : inputs.first(nin)) {
            layouts.push_back(broadcast_to(op.layout, shape));
            base.push_back(reinterpret_cast<char*>(op.first()));
        }
        for (const Operand& op : outputs) {
            layouts.push_back(op.layout);
            base.push_back(reinterpret_cast<char*>(op.first()));
        }
        const bool ok = for_each_row(shape, layouts, std::move(base),
                                     [this](char** ptrs, std::intptr_t n, const std::intptr_t* st) {
                                         return loop_->call(ptrs, &n, st) == 0;
                                     });
        return ok ? KernelResult{} : KernelResult{1, "NumPy loop failed", 0};
    }

  private:
    static constexpr std::size_t kMaxFlat = 8;
    const UfuncLoop* loop_;
};

} // namespace

std::shared_ptr<Kernel> make_elementwise_kernel(const UfuncLoop* loop) {
    return std::make_shared<ElementwiseKernel>(loop);
}

bool supported_ndarray(PyObject* obj) {
    if (!PyArray_Check(obj)) {
        return false;
    }
    auto* arr = reinterpret_cast<PyArrayObject*>(obj);
    DType d;
    return dtype_of(PyArray_TYPE(arr), d) && PyArray_ISALIGNED(arr) && PyArray_ISNOTSWAPPED(arr);
}

ArrayPtr wrap_ndarray(Runtime& rt, PyObject* obj, std::string& error, std::shared_ptr<void> owner) {
    if (!PyArray_Check(obj)) {
        error = "not a NumPy array";
        return nullptr;
    }
    auto* arr = reinterpret_cast<PyArrayObject*>(obj);
    Layout l;
    if (!dtype_of(PyArray_TYPE(arr), l.dtype) || !PyArray_ISALIGNED(arr) ||
        !PyArray_ISNOTSWAPPED(arr)) {
        error = "unsupported dtype or memory layout";
        return nullptr;
    }
    const int nd = PyArray_NDIM(arr);
    l.shape.assign(PyArray_SHAPE(arr), PyArray_SHAPE(arr) + nd);
    l.strides.assign(PyArray_STRIDES(arr), PyArray_STRIDES(arr) + nd);
    // Memory extent covered by the view (strides may be negative or zero).
    std::int64_t lo = 0;
    std::int64_t hi = static_cast<std::int64_t>(itemsize(l.dtype));
    for (int i = 0; i < nd; ++i) {
        const std::int64_t span = (l.shape[i] - 1) * l.strides[i];
        (span < 0 ? lo : hi) += l.shape[i] > 0 ? span : 0;
    }
    l.offset = -lo;
    auto buffer = std::make_shared<Buffer>(PyArray_BYTES(arr) + lo,
                                           static_cast<std::size_t>(hi - lo), std::move(owner));
    return rt.wrap(std::move(l), std::move(buffer));
}

ArrayPtr scalar_array(Runtime& rt, PyObject* value, DType dtype) {
    PyArray_Descr* descr = PyArray_DescrFromType(type_num(dtype));
    PyObject* arr = PyArray_FromAny(value, descr, 0, 0, 0, nullptr); // steals descr
    if (arr == nullptr) {
        return nullptr;
    }
    Layout l = Layout::contiguous(dtype, {});
    auto buffer = std::make_shared<Buffer>(l.nbytes());
    buffer->ensure_allocated();
    std::memcpy(buffer->data(), PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)), l.nbytes());
    Py_DECREF(arr);
    return rt.wrap(std::move(l), std::move(buffer));
}

PyObject* to_ndarray(const Array& a) {
    const Layout& l = a.layout();
    std::vector<npy_intp> shape(l.shape.begin(), l.shape.end());
    std::vector<npy_intp> strides(l.strides.begin(), l.strides.end());
    PyObject* view =
        PyArray_New(&PyArray_Type, static_cast<int>(shape.size()), shape.data(), type_num(l.dtype),
                    strides.data(), a.buffer()->data() + l.offset, 0, 0, nullptr);
    if (view == nullptr) {
        return nullptr;
    }
    PyObject* copy = PyArray_NewCopy(reinterpret_cast<PyArrayObject*>(view), NPY_CORDER);
    Py_DECREF(view);
    return copy;
}

} // namespace viproc::numpy
