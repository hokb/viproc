#include "kernels/numpy_ufunc.hpp"

#include "runtime/pool.hpp"

#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
// NumPy >= 2.0 is required anyway; 2.0 adds PyArray_Pack to the C API.
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL viproc_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL viproc_UFUNC_API
#include <numpy/arrayobject.h>
#include <numpy/dtype_api.h>
#include <numpy/ufuncobject.h>

#include "kernels/numpy_iter.hpp"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <mutex>

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
        break;
    }
    // The other C integer type of the same width (Linux: `longlong` next to
    // `long` for int64; Windows: `int` next to `long` for int32): the same
    // dtype for NumPy (np.dtype("q") == np.dtype("l")), only another number.
    std::size_t size = 0;
    if (type_num == NPY_INT) {
        size = sizeof(int);
    } else if (type_num == NPY_LONG) {
        size = sizeof(long);
    } else if (type_num == NPY_LONGLONG) {
        size = sizeof(long long);
    }
    if (size == 4) {
        out = DType::Int32;
        return true;
    }
    if (size == 8) {
        out = DType::Int64;
        return true;
    }
    return false;
}

// NumPy's descriptor `type_num` is viproc's `dtype` (NPY_INT and NPY_LONG
// are both int32 on Windows: NumPy may resolve either, e.g. frexp's `d->di`).
bool same_dtype(int type_num, DType dtype) {
    DType d{};
    return dtype_of(type_num, d) && d == dtype;
}

} // namespace

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
        exact = exact && same_dtype(d->type_num, dtypes[i]);
    }
    Py_DECREF(requested);
    if (!exact) {
        Py_DECREF(res);
        error = "NumPy resolves a different dtype signature (casting not supported yet)";
        return nullptr;
    }
    std::unique_ptr<UfuncLoop> loop(new UfuncLoop());
    loop->ufunc_ = ufunc;
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
    loop->init_info("");
    loop->kernel_ = make_elementwise_kernel(loop.get());
    return loop;
}

std::unique_ptr<UfuncLoop> UfuncLoop::resolve_reduction(PyObject* ufunc, DType dtype,
                                                        const std::array<std::intptr_t, 3>& fixed,
                                                        std::string& error) {
    auto* uf = reinterpret_cast<PyUFuncObject*>(ufunc);
    if (!PyObject_TypeCheck(ufunc, &PyUFunc_Type) || uf->nin != 2 || uf->nout != 1) {
        error = "not a binary ufunc";
        return nullptr;
    }
    // As NumPy's reduction: (None, operand dtype, None), reduction=True.
    PyObject* descr = reinterpret_cast<PyObject*>(PyArray_DescrFromType(type_num(dtype)));
    PyObject* requested = PyTuple_Pack(3, Py_None, descr, Py_None);
    Py_DECREF(descr);
    PyObject* meth = PyObject_GetAttrString(ufunc, "_resolve_dtypes_and_context");
    PyObject* args = PyTuple_Pack(1, requested);
    PyObject* kwargs = Py_BuildValue("{s:O}", "reduction", Py_True);
    PyObject* res = meth != nullptr && args != nullptr && kwargs != nullptr
                        ? PyObject_Call(meth, args, kwargs)
                        : nullptr;
    Py_XDECREF(meth);
    Py_XDECREF(args);
    Py_XDECREF(kwargs);
    Py_DECREF(requested);
    if (res == nullptr) {
        error = fetch_python_error();
        return nullptr;
    }
    PyObject* resolved = PyTuple_GetItem(res, 0);
    bool exact = true;
    for (Py_ssize_t i = 0; i < 3; ++i) {
        auto* d = reinterpret_cast<PyArray_Descr*>(PyTuple_GetItem(resolved, i));
        exact = exact && same_dtype(d->type_num, dtype);
    }
    if (!exact) {
        Py_DECREF(res);
        error = "the reduction casts (casting not supported yet)";
        return nullptr;
    }
    std::unique_ptr<UfuncLoop> loop(new UfuncLoop());
    loop->ufunc_ = ufunc;
    loop->capsule_ = PyTuple_GetItem(res, 1);
    Py_INCREF(loop->capsule_);
    Py_DECREF(res);

    PyObject* get = PyObject_GetAttrString(ufunc, "_get_strided_loop");
    PyObject* get_args = PyTuple_Pack(1, loop->capsule_);
    PyObject* strides =
        Py_BuildValue("(nnn)", static_cast<Py_ssize_t>(fixed[0]), static_cast<Py_ssize_t>(fixed[1]),
                      static_cast<Py_ssize_t>(fixed[2]));
    PyObject* get_kwargs = Py_BuildValue("{s:O}", "fixed_strides", strides);
    PyObject* filled = get != nullptr && get_args != nullptr && get_kwargs != nullptr
                           ? PyObject_Call(get, get_args, get_kwargs)
                           : nullptr;
    Py_XDECREF(get);
    Py_XDECREF(get_args);
    Py_XDECREF(strides);
    Py_XDECREF(get_kwargs);
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
    loop->nin_ = 2;
    loop->nout_ = 1;
    if (loop->requires_pyapi()) {
        error = "loop requires the Python API; cannot run asynchronously";
        return nullptr;
    }
    loop->init_info(".reduce");
    return loop;
}

bool UfuncLoop::requires_pyapi() const {
    return static_cast<const UfuncCallInfoPrefix*>(info_)->requires_pyapi != 0;
}

namespace {

thread_local std::string t_loop_error;

// The Python thread state of this thread, for loops that raise through the
// Python API. The main thread has one. A worker gets its own on first use
// (PyThreadState_New needs no GIL) and keeps it: then the GIL state the
// loop's npy_gil_error() takes (PyGILState_Ensure) finds it, and the error
// stays on it after PyGILState_Release (a temporary thread state would be
// deleted, and the error with it). Deleted when the worker exits; the
// runtime is destroyed with the GIL released (vp_runtime_destroy), so the
// worker can take it then.
struct WorkerThreadState {
    PyThreadState* ts = nullptr;
    bool own = false;
    ~WorkerThreadState() {
        if (own && Py_IsInitialized() != 0 && !python_finalizing()) {
            PyEval_RestoreThread(ts);
            PyThreadState_Clear(ts);
            PyThreadState_DeleteCurrent();
        }
    }
    static bool python_finalizing() {
#if PY_VERSION_HEX >= 0x030D0000
        return Py_IsFinalizing() != 0;
#else
        return _Py_IsFinalizing() != 0;
#endif
    }
};
thread_local WorkerThreadState t_python;

PyThreadState* this_thread_state(PyInterpreterState* interp) {
    if (t_python.ts == nullptr) {
        t_python.ts = PyGILState_GetThisThreadState();
        if (t_python.ts == nullptr) {
            t_python.ts = PyThreadState_New(interp);
            t_python.own = t_python.ts != nullptr;
        }
    }
    return t_python.ts;
}

// The raised exception as "TypeName: message" (builtin types by their name).
std::string fetch_typed_python_error() {
    PyObject* exc = PyErr_GetRaisedException();
    if (exc == nullptr) {
        return "unknown error";
    }
    std::string msg = Py_TYPE(exc)->tp_name;
    PyObject* s = PyObject_Str(exc);
    const char* text = s != nullptr ? PyUnicode_AsUTF8(s) : nullptr;
    msg += ": ";
    msg += text != nullptr ? text : "unknown error";
    Py_XDECREF(s);
    Py_DECREF(exc);
    PyErr_Clear();
    return msg;
}

} // namespace

std::string take_loop_error() {
    std::string e = std::move(t_loop_error);
    t_loop_error.clear();
    return e.empty() ? std::string("NumPy loop failed") : e;
}

int UfuncLoop::call(char* const* data, const std::intptr_t* dims,
                    const std::intptr_t* strides) const {
    if (raises_through_pyapi_) {
        return call_watched(data, dims, strides);
    }
    const auto* info = static_cast<const UfuncCallInfoPrefix*>(info_);
    return info->strided_loop(info->context, data, dims, strides, info->auxdata);
}

int UfuncLoop::call_watched(char* const* data, const std::intptr_t* dims,
                            const std::intptr_t* strides) const {
    // The loop sets its exception on this thread's Python thread state. That
    // state belongs to this thread alone: reading its current exception
    // needs no GIL. The GIL is taken only if there is one (the error path,
    // where npy_gil_error() took it already).
    PyThreadState* ts = this_thread_state(interp_);
    if (ts == nullptr) {
        t_loop_error = "cannot create a Python thread state for this loop";
        return -1;
    }
    PyObject* const before = ts->current_exception;
    const auto* info = static_cast<const UfuncCallInfoPrefix*>(info_);
    int r = info->strided_loop(info->context, data, dims, strides, info->auxdata);
    if (ts->current_exception != before) {
        const PyGILState_STATE g = PyGILState_Ensure();
        t_loop_error = fetch_typed_python_error();
        PyGILState_Release(g);
        r = -1;
    }
    return r;
}

void UfuncLoop::init_info(const char* suffix) {
    auto* uf = reinterpret_cast<PyUFuncObject*>(ufunc_);
    name_ = std::string("numpy.") + (uf->name != nullptr ? uf->name : "?") + suffix;
    interp_ = PyInterpreterState_Get();
    // NumPy's own `power` with a signed integer exponent only.
    static PyObject* power = [] {
        PyObject* umath = PyImport_ImportModule("numpy._core.umath");
        PyObject* p = umath != nullptr ? PyObject_GetAttrString(umath, "power") : nullptr;
        Py_XDECREF(umath);
        PyErr_Clear();
        return p; // kept for the process
    }();
    if (ufunc_ == power && power != nullptr && nin_ == 2) {
        const auto* info = static_cast<const UfuncCallInfoPrefix*>(info_);
        raises_through_pyapi_ = PyTypeNum_ISSIGNED(info->context->descriptors[1]->type_num);
    }
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

PyObject* resolve_reduction_info(PyObject* ufunc, PyObject* in_descr, PyObject* dtype,
                                 std::string& error) {
    PyObject* dtypes = PyTuple_Pack(3, Py_None, in_descr, Py_None);
    PyObject* signature = PyTuple_Pack(3, dtype, Py_None, Py_None);
    PyObject* meth = PyObject_GetAttrString(ufunc, "_resolve_dtypes_and_context");
    PyObject* args = dtypes != nullptr ? PyTuple_Pack(1, dtypes) : nullptr;
    PyObject* kwargs = signature != nullptr
                           ? Py_BuildValue("{s:O,s:O,s:s}", "signature", signature, "reduction",
                                           Py_True, "casting", "unsafe")
                           : nullptr;
    PyObject* res = meth != nullptr && args != nullptr && kwargs != nullptr
                        ? PyObject_Call(meth, args, kwargs)
                        : nullptr;
    Py_XDECREF(dtypes);
    Py_XDECREF(signature);
    Py_XDECREF(meth);
    Py_XDECREF(args);
    Py_XDECREF(kwargs);
    if (res == nullptr) {
        error = fetch_python_error();
    }
    return res;
}

std::unique_ptr<UfuncLoop> UfuncLoop::from_call_info(PyObject* ufunc, PyObject* capsule,
                                                     std::span<const std::intptr_t> fixed,
                                                     std::string& error) {
    std::unique_ptr<UfuncLoop> loop(new UfuncLoop());
    loop->ufunc_ = ufunc;
    loop->capsule_ = capsule;
    Py_INCREF(capsule);
    PyObject* strides = PyTuple_New(static_cast<Py_ssize_t>(fixed.size()));
    for (std::size_t i = 0; strides != nullptr && i < fixed.size(); ++i) {
        PyTuple_SET_ITEM(strides, static_cast<Py_ssize_t>(i),
                         PyLong_FromSsize_t(static_cast<Py_ssize_t>(fixed[i])));
    }
    PyObject* get = PyObject_GetAttrString(ufunc, "_get_strided_loop");
    PyObject* get_args = PyTuple_Pack(1, capsule);
    PyObject* get_kwargs =
        strides != nullptr ? Py_BuildValue("{s:O}", "fixed_strides", strides) : nullptr;
    PyObject* filled = get != nullptr && get_args != nullptr && get_kwargs != nullptr
                           ? PyObject_Call(get, get_args, get_kwargs)
                           : nullptr;
    Py_XDECREF(strides);
    Py_XDECREF(get);
    Py_XDECREF(get_args);
    Py_XDECREF(get_kwargs);
    if (filled == nullptr) {
        error = fetch_python_error();
        return nullptr;
    }
    Py_DECREF(filled);
    loop->info_ = PyCapsule_GetPointer(capsule, kCallInfoCapsule);
    if (loop->info_ == nullptr) {
        error = fetch_python_error();
        return nullptr;
    }
    auto* uf = reinterpret_cast<PyUFuncObject*>(ufunc);
    loop->nin_ = uf->nin;
    loop->nout_ = uf->nout;
    if (loop->requires_pyapi()) {
        error = "loop requires the Python API; cannot run asynchronously";
        return nullptr;
    }
    loop->init_info(".reduce"); // only reductions on NumPy's iterator use it
    return loop;
}

const UfuncLoop* LoopCache::get_reduction(PyObject* ufunc, DType dtype,
                                          const std::array<std::intptr_t, 3>& fixed,
                                          std::string& error) {
    auto key = std::make_tuple(ufunc, dtype, fixed);
    auto it = reduction_loops_.find(key);
    if (it != reduction_loops_.end()) {
        return it->second.get();
    }
    std::unique_ptr<UfuncLoop> loop = UfuncLoop::resolve_reduction(ufunc, dtype, fixed, error);
    if (!loop) {
        return nullptr;
    }
    Py_INCREF(ufunc); // as in get()
    return reduction_loops_.emplace(std::move(key), std::move(loop)).first->second.get();
}

namespace {

// Dense with positive strides: the elements fill a block starting at the
// first element (Layout::dense() without the offset condition).
bool dense_from_first(const Layout& l) {
    Layout at0 = l;
    at0.offset = 0;
    return at0.dense();
}

// Same shape and the same strides along every axis of extent > 1.
bool same_order(const Layout& a, const Layout& b) {
    if (a.shape != b.shape) {
        return false;
    }
    for (std::size_t i = 0; i < a.shape.size(); ++i) {
        if (a.shape[i] != 1 && a.strides[i] != b.strides[i]) {
            return false;
        }
    }
    return true;
}

class ElementwiseKernel : public Kernel {
  public:
    explicit ElementwiseKernel(const UfuncLoop* loop) : loop_(loop) {}
    const char* name() const override { return loop_->name(); }

    // Element-wise loops read element i before writing it: an output may
    // share an input's memory with the same layout (NumPy's out= / elision).
    bool in_place_ok() const override { return true; }

    // Only the first nin() inputs go to the loop; further inputs are in/out
    // arrays the runtime lists as read (see Runtime::issue).
    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Shape& shape = outputs[0].layout().shape;
        const auto nin = static_cast<std::size_t>(loop_->nin());
        if (nin + outputs.size() <= kMaxFlat && outputs.size() == 1) {
            const Layout* layouts[kMaxFlat];
            char* ptrs[kMaxFlat];
            for (std::size_t i = 0; i < nin; ++i) {
                layouts[i] = &inputs[i].layout();
                ptrs[i] = reinterpret_cast<char*>(inputs[i].first());
            }
            ptrs[nin] = reinterpret_cast<char*>(outputs[0].first());
            std::intptr_t strides[kMaxFlat];
            if (single_call(std::span(layouts, nin), outputs[0].layout(), strides)) {
                const auto count = static_cast<std::intptr_t>(outputs[0].layout().size());
                if (count == 0) {
                    return {};
                }
                return loop_->call(ptrs, &count, strides) == 0
                           ? KernelResult{}
                           : KernelResult{1, take_loop_error(), 0};
            }
        }
        std::vector<Layout> layouts;
        layouts.reserve(nin + outputs.size());
        OperandPointers base;
        for (const Operand& op : inputs.first(nin)) {
            layouts.push_back(broadcast_to(op.layout(), shape));
            base.push_back(reinterpret_cast<char*>(op.first()));
        }
        for (const Operand& op : outputs) {
            layouts.push_back(op.layout());
            base.push_back(reinterpret_cast<char*>(op.first()));
        }
        const bool ok = for_each_row(shape, layouts, std::move(base),
                                     [this](char** ptrs, std::intptr_t n, const std::intptr_t* st) {
                                         return loop_->call(ptrs, &n, st) == 0;
                                     });
        return ok ? KernelResult{} : KernelResult{1, take_loop_error(), 0};
    }

  private:
    static constexpr std::size_t kMaxFlat = 8;
    const UfuncLoop* loop_;
};

// ufunc.at(a, idx, b) with a 1-d index into axis 0 (see make_at_kernel).
class AtKernel : public Kernel {
  public:
    explicit AtKernel(const UfuncLoop* loop)
        : loop_(loop), name_(std::string(loop->name()) + ".at") {}
    const char* name() const override { return name_.c_str(); }
    bool writes_all() const override { return false; }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const Layout& al = outputs[0].layout();
        const Layout& xl = inputs[0].layout();
        const std::int64_t n = xl.shape[0];
        const std::int64_t extent = al.shape[0];
        const char* xp = reinterpret_cast<const char*>(inputs[0].first());
        const bool i64 = xl.dtype == DType::Int64;
        auto index = [&](std::int64_t i) {
            const char* p = xp + i * xl.strides[0];
            std::int64_t v;
            if (i64) {
                std::memcpy(&v, p, sizeof v);
            } else {
                std::int32_t w;
                std::memcpy(&w, p, sizeof w);
                v = w;
            }
            return v;
        };
        // NumPy checks every index before it changes anything
        // (PyArray_MapIterCheckIndices)
        for (std::int64_t i = 0; i < n; ++i) {
            const std::int64_t v = index(i);
            if (v < -extent || v >= extent) {
                return {1,
                        "IndexError: index " + std::to_string(v) +
                            " is out of bounds for axis 0 with size " + std::to_string(extent),
                        0};
            }
        }
        // the subspace: a's other axes, in C order; b broadcast to idx.shape + it
        Shape sub(al.shape.begin() + 1, al.shape.end());
        Shape full = al.shape; // idx.shape + a.shape[1:]
        full[0] = n;
        const bool binary = loop_->nin() == 2;
        Layout bl;
        const char* bp = nullptr;
        if (binary) {
            bl = broadcast_to(inputs[1].layout(), full);
            bp = reinterpret_cast<const char*>(inputs[1].first());
        }
        char* ap = reinterpret_cast<char*>(outputs[0].first());
        const std::size_t ns = sub.size();
        std::int64_t total = 1;
        for (std::int64_t e : sub) {
            total *= e;
        }
        SmallVector<std::int64_t, kInlineRank> idx(ns, 0);
        const std::intptr_t strides[3] = {0, 0, 0};
        std::intptr_t one = 1;
        for (std::int64_t i = 0; i < n; ++i) {
            const std::int64_t v = index(i);
            char* a = ap + (v < 0 ? v + extent : v) * al.strides[0];
            char* b = binary ? const_cast<char*>(bp) + i * bl.strides[0] : a;
            for (std::int64_t j = 0; j < total; ++j) {
                // ufunc_at__fast_iter: one element per call, a is in and out
                char* data[3] = {a, b, a};
                if (loop_->call(data, &one, strides) != 0) {
                    return {1, take_loop_error(), 0};
                }
                for (std::size_t k = ns; k-- > 0;) {
                    if (++idx[k] < sub[k]) {
                        a += al.strides[k + 1];
                        b += binary ? bl.strides[k + 1] : al.strides[k + 1];
                        break;
                    }
                    a -= (sub[k] - 1) * al.strides[k + 1];
                    b -= (sub[k] - 1) * (binary ? bl.strides[k + 1] : al.strides[k + 1]);
                    idx[k] = 0;
                }
            }
        }
        return {};
    }

  private:
    const UfuncLoop* loop_;
    std::string name_;
};

// NumPy's C/F contiguity flags (extent-1 dimensions may have any stride).
bool f_contiguous(const Layout& l) {
    std::int64_t step = static_cast<std::int64_t>(itemsize(l.dtype));
    for (std::size_t i = 0; i < l.shape.size(); ++i) {
        if (l.shape[i] == 0) {
            return true;
        }
        if (l.shape[i] != 1) {
            if (l.strides[i] != step) {
                return false;
            }
            step *= l.shape[i];
        }
    }
    return true;
}

int contiguity(const Layout& l) { return (l.c_contiguous() ? 1 : 0) | (f_contiguous(l) ? 2 : 0); }

// Operands of an element-wise op (NPY_MAXARGS is not a constant in NumPy 2).
constexpr std::size_t kMaxOps = 32;
// Iterator templates kept per LoopCache (all dropped when full).
constexpr std::size_t kMaxIterTemplates = 1024;

} // namespace

namespace {

// NumPy's element-wise ufunc loop over its iterator (execute_ufunc_loop).
class IterElementwiseKernel : public Kernel {
  public:
    IterElementwiseKernel(const UfuncLoop* loop, NpyIter* iter, std::shared_ptr<IterTemplate> pool,
                          bool masked)
        : loop_(loop), iter_(iter), pool_(std::move(pool)), masked_(masked) {}

    ~IterElementwiseKernel() override { pool_->give_back(iter_, clean_); }
    const char* name() const override { return loop_->name(); }

    bool in_place_ok() const override { return true; }
    // where=: masked-off elements of the output keep their values
    bool writes_all() const override { return !masked_; }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        const auto nin = static_cast<std::size_t>(loop_->nin());
        char* base[kMaxOps];
        for (std::size_t i = 0; i < nin; ++i) {
            base[i] = reinterpret_cast<char*>(inputs[i].first());
        }
        const auto nop = nin + static_cast<std::size_t>(loop_->nout());
        for (std::size_t j = nin; j < nop; ++j) {
            base[j] = reinterpret_cast<char*>(outputs[j - nin].first());
        }
        if (masked_) {
            base[nop] = reinterpret_cast<char*>(inputs[nin].first()); // the where= mask
        }
        char* errmsg = nullptr;
        clean_ = false;
        // Points the iterator at the buffers (no GIL: errmsg variant); with
        // buffering it also copies the first block.
        if (NpyIter_ResetBasePointers(iter_, base, &errmsg) != NPY_SUCCEED) {
            return {1, errmsg != nullptr ? errmsg : "NpyIter_ResetBasePointers failed", 0};
        }
        do {
            const int r = masked_ ? masked_call(nop) : loop_->call(dataptr_, countptr_, strideptr_);
            if (r != 0) {
                return {1, take_loop_error(), 0};
            }
        } while (iternext_(iter_));
        clean_ = true;
        return {};
    }

    // NumPy's generic_masked_strided_loop (array_method.c): the loop is
    // called once per run of True mask values, the False ones are skipped.
    int masked_call(std::size_t nop) const {
        char* ptrs[kMaxOps];
        std::copy(dataptr_, dataptr_ + nop, ptrs);
        const char* mask = dataptr_[nop];
        const npy_intp ms = strideptr_[nop];
        npy_intp n = *countptr_;
        while (n > 0) {
            npy_intp skip = 0;
            while (skip < n && *mask == 0) {
                mask += ms;
                ++skip;
            }
            n -= skip;
            npy_intp run = 0;
            while (run < n && *mask != 0) {
                mask += ms;
                ++run;
            }
            for (std::size_t i = 0; i < nop; ++i) {
                ptrs[i] += skip * strideptr_[i];
            }
            if (run > 0) {
                if (loop_->call(ptrs, &run, strideptr_) != 0) {
                    return -1;
                }
                for (std::size_t i = 0; i < nop; ++i) {
                    ptrs[i] += run * strideptr_[i];
                }
                n -= run;
            }
        }
        return 0;
    }

    NpyIter_IterNextFunc* iternext_ = nullptr;
    char** dataptr_ = nullptr;
    npy_intp* strideptr_ = nullptr;
    npy_intp* countptr_ = nullptr;

  private:
    const UfuncLoop* loop_;
    NpyIter* iter_;
    std::shared_ptr<IterTemplate> pool_;
    bool clean_ = true; // a fresh copy has nothing in its buffers
    bool masked_;
};

npy_intp numpy_buffersize() {
    npy_intp buffersize = 0;
    if (PyObject* np = PyImport_ImportModule("numpy")) {
        if (PyObject* bs = PyObject_CallMethod(np, "getbufsize", nullptr)) {
            buffersize = PyLong_AsSsize_t(bs);
            Py_DECREF(bs);
        }
        Py_DECREF(np);
    }
    PyErr_Clear();
    return buffersize;
}

} // namespace

PyArrayObject* template_array(const Layout& l, char* data, bool writable) {
    npy_intp shape[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    for (std::size_t i = 0; i < l.shape.size(); ++i) {
        shape[i] = static_cast<npy_intp>(l.shape[i]);
        strides[i] = static_cast<npy_intp>(l.strides[i]);
    }
    PyArray_Descr* descr = PyArray_DescrFromType(type_num(l.dtype));
    return reinterpret_cast<PyArrayObject*>(PyArray_NewFromDescr(
        &PyArray_Type, descr, static_cast<int>(l.shape.size()), shape, strides, data,
        NPY_ARRAY_ALIGNED | (writable ? NPY_ARRAY_WRITEABLE : 0), nullptr)); // steals descr
}

char* template_address() {
    alignas(64) static char dummy[64];
    return dummy;
}

std::shared_ptr<Kernel> make_at_kernel(const UfuncLoop* loop) {
    return std::make_shared<AtKernel>(loop);
}

std::shared_ptr<Kernel> make_elementwise_kernel(const UfuncLoop* loop) {
    return std::make_shared<ElementwiseKernel>(loop);
}

bool single_call(std::span<const Layout* const> inputs, const Layout& out, std::intptr_t* strides) {
    const std::size_t nin = inputs.size();
    const auto isz = [](const Layout& l) { return static_cast<std::intptr_t>(itemsize(l.dtype)); };
    // NumPy's trivial loop (try_trivial_single_output_loop): 0-d inputs with
    // stride 0; all other operands with the output's shape, either 1-d (with
    // their own strides) or all with the same C/F contiguity flags.
    // The common case first: n-d operands of the output's shape, all C- and
    // none F-contiguous (no extent-1 dimension): the trivial loop with
    // stride = itemsize, as below, without the contiguity flags.
    if (out.shape.size() >= 2 &&
        std::find(out.shape.begin(), out.shape.end(), 1) == out.shape.end()) {
        bool c_all = out.c_contiguous();
        for (std::size_t i = 0; i < nin && c_all; ++i) {
            c_all = inputs[i]->shape == out.shape && inputs[i]->c_contiguous();
        }
        if (c_all) {
            for (std::size_t i = 0; i <= nin; ++i) {
                strides[i] = isz(i < nin ? *inputs[i] : out);
            }
            return true;
        }
    }
    bool trivial = true;
    int order = -1;
    const bool one_d = out.shape.size() == 1;
    for (std::size_t i = 0; i <= nin && trivial; ++i) {
        const Layout& l = i < nin ? *inputs[i] : out;
        if (i < nin && l.shape.empty()) {
            strides[i] = 0;
        } else if (l.shape != out.shape) {
            trivial = false;
        } else if (one_d) {
            strides[i] = static_cast<std::intptr_t>(l.strides[0]);
        } else {
            const int c = contiguity(l);
            trivial = c != 0 && (order < 0 || order == c);
            order = c;
            strides[i] = isz(l);
        }
    }
    if (trivial) {
        return !(one_d && out.strides[0] < static_cast<std::int64_t>(isz(out)) &&
                 out.strides[0] != 0); // self-overlapping output: NumPy's iterator
    }
    // NumPy's iterator, where it coalesces all axes into one and buffers
    // nothing: every operand dense with the output's strides, or a single
    // element (stride 0).
    if (!(out.c_contiguous() || dense_from_first(out))) {
        return false;
    }
    for (std::size_t i = 0; i < nin; ++i) {
        const Layout& l = *inputs[i];
        if (same_order(l, out)) {
            strides[i] = isz(l);
        } else if (l.size() == 1) {
            strides[i] = 0;
        } else {
            return false;
        }
    }
    strides[nin] = isz(out);
    return true;
}

namespace {

// A cast with NumPy's cast functions: an iterator over {input, output}
// requesting the output dtype for both casts the input block by block into
// its buffer; the kernel copies each block into the output (no arithmetic:
// the cast happened in NumPy's buffer copy).
class CastKernel : public Kernel {
  public:
    CastKernel(NpyIter* iter, DeferRelease defer) : iter_(iter), defer_(std::move(defer)) {}
    const char* name() const override { return "astype"; }

    ~CastKernel() override {
        NpyIter* iter = iter_;
        defer_([iter] { NpyIter_Deallocate(iter); });
    }

    KernelResult run(std::span<const Operand> inputs, std::span<const Operand> outputs) override {
        char* base[2] = {reinterpret_cast<char*>(inputs[0].first()),
                         reinterpret_cast<char*>(outputs[0].first())};
        char* errmsg = nullptr;
        if (NpyIter_ResetBasePointers(iter_, base, &errmsg) != NPY_SUCCEED) {
            return {1, errmsg != nullptr ? errmsg : "NpyIter_ResetBasePointers failed", 0};
        }
        const auto isz = static_cast<std::size_t>(itemsize_);
        do {
            const npy_intp n = *countptr_;
            char* src = dataptr_[0];
            char* dst = dataptr_[1];
            if (strideptr_[0] == itemsize_ && strideptr_[1] == itemsize_) {
                std::memcpy(dst, src, static_cast<std::size_t>(n) * isz);
            } else {
                for (npy_intp i = 0; i < n; ++i) {
                    std::memcpy(dst + i * strideptr_[1], src + i * strideptr_[0], isz);
                }
            }
        } while (iternext_(iter_));
        return {};
    }

    NpyIter_IterNextFunc* iternext_ = nullptr;
    char** dataptr_ = nullptr;
    npy_intp* strideptr_ = nullptr;
    npy_intp* countptr_ = nullptr;
    npy_intp itemsize_ = 0;

  private:
    NpyIter* iter_;
    DeferRelease defer_;
};

// A 0-d layout as one element of a 1-d layout: NumPy's iterator does not
// buffer (and so would not cast) a 0-d operand.
Layout at_least_1d(Layout l) {
    if (l.shape.empty()) {
        l.shape.push_back(1);
        l.strides.push_back(static_cast<std::int64_t>(itemsize(l.dtype)));
    }
    return l;
}

} // namespace

std::shared_ptr<Kernel> make_cast_kernel(const Layout& in, const Layout& out, long buffersize,
                                         DeferRelease defer, std::string& error) {
    char* const dummy = template_address();
    PyArrayObject* op[2] = {template_array(at_least_1d(in), dummy, false),
                            template_array(at_least_1d(out), dummy, true)};
    PyArray_Descr* descr = PyArray_DescrFromType(type_num(out.dtype));
    PyArray_Descr* descrs[2] = {descr, descr};
    npy_uint32 op_flags[2] = {NPY_ITER_READONLY | NPY_ITER_ALIGNED,
                              NPY_ITER_WRITEONLY | NPY_ITER_ALIGNED | NPY_ITER_NO_BROADCAST};
    NpyIter* iter = nullptr;
    if (op[0] != nullptr && op[1] != nullptr) {
        // Any element order works: each element is cast on its own.
        iter = NpyIter_AdvancedNew(2, op,
                                   NPY_ITER_BUFFERED | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_GROWINNER |
                                       NPY_ITER_ZEROSIZE_OK | NPY_ITER_DELAY_BUFALLOC,
                                   NPY_KEEPORDER, NPY_UNSAFE_CASTING, op_flags, descrs, -1, nullptr,
                                   nullptr, static_cast<npy_intp>(buffersize));
    }
    Py_XDECREF(op[0]);
    Py_XDECREF(op[1]);
    Py_DECREF(descr);
    if (iter == nullptr) {
        error = fetch_python_error();
        return nullptr;
    }
    if (NpyIter_IterationNeedsAPI(iter) || NpyIter_GetIterSize(iter) == 0) {
        NpyIter_Deallocate(iter);
        error = "cast needs the Python API or is empty";
        return nullptr;
    }
    auto kernel = std::make_shared<CastKernel>(iter, std::move(defer));
    kernel->iternext_ = NpyIter_GetIterNext(iter, nullptr);
    if (kernel->iternext_ == nullptr) {
        error = fetch_python_error();
        return nullptr;
    }
    kernel->dataptr_ = NpyIter_GetDataPtrArray(iter);
    kernel->strideptr_ = NpyIter_GetInnerStrideArray(iter);
    kernel->countptr_ = NpyIter_GetInnerLoopSizePtr(iter);
    kernel->itemsize_ = static_cast<npy_intp>(itemsize(out.dtype));
    return kernel;
}

LoopCache::~LoopCache() {
    iter_templates_.clear(); // kernels still holding one release it later
    reduce_templates_.clear();
    Py_XDECREF(extobj_);
    Py_XDECREF(extobj_var_);
}

long LoopCache::buffersize() {
    // np.getbufsize() costs ~0.5 us; NumPy keeps it (with the error mode) in
    // the value of a context variable, which np.setbufsize/np.errstate
    // replace. Re-read only when that value changes (the reference held in
    // extobj_ keeps its address from being reused).
    if (extobj_var_ == nullptr) {
        if (PyObject* m = PyImport_ImportModule("numpy._core._multiarray_umath")) {
            extobj_var_ = PyObject_GetAttrString(m, "_extobj_contextvar");
            Py_DECREF(m);
        }
        if (extobj_var_ != nullptr && !PyContextVar_CheckExact(extobj_var_)) {
            Py_CLEAR(extobj_var_);
        }
        PyErr_Clear();
    }
    PyObject* value = nullptr;
    if (extobj_var_ != nullptr && PyContextVar_Get(extobj_var_, nullptr, &value) == 0 &&
        value != nullptr && value == extobj_) {
        Py_DECREF(value);
        return buffersize_;
    }
    PyErr_Clear();
    buffersize_ = static_cast<long>(numpy_buffersize());
    Py_XDECREF(extobj_);
    extobj_ = value; // the new reference from PyContextVar_Get (or nullptr)
    return buffersize_;
}

std::shared_ptr<Kernel> LoopCache::iter_elementwise(const UfuncLoop* loop,
                                                    std::span<const DType> dtypes,
                                                    std::span<const Layout> inputs,
                                                    std::span<const Layout> outs, bool out_given,
                                                    DeferRelease defer, std::string& error,
                                                    const Layout* mask) {
    const std::size_t nin = inputs.size();
    const std::size_t nop = nin + outs.size();
    const std::size_t nit = nop + (mask != nullptr ? 1 : 0);
    if (nit > kMaxOps || static_cast<int>(nin) != loop->nin() ||
        static_cast<int>(outs.size()) != loop->nout()) {
        error = "unsupported number of operands";
        return nullptr;
    }
    const long bufsize = buffersize();
    std::string key;
    auto put = [&key](std::int64_t v) { key.append(reinterpret_cast<const char*>(&v), sizeof v); };
    put(reinterpret_cast<std::intptr_t>(loop));
    put(out_given ? 1 : 0);
    put(bufsize);
    put(mask != nullptr ? 1 : 0);
    for (std::size_t i = 0; i < nit; ++i) {
        const Layout& l = i < nin ? inputs[i] : i < nop ? outs[i - nin] : *mask;
        put(static_cast<std::int64_t>(l.shape.size()));
        put(static_cast<std::int64_t>(l.dtype));
        for (std::size_t d = 0; d < l.shape.size(); ++d) {
            put(l.shape[d]);
            put(l.strides[d]);
        }
    }
    std::shared_ptr<IterTemplate> pool;
    if (auto it = iter_templates_.find(key); it != iter_templates_.end()) {
        pool = it->second;
    } else {
        NpyIter* templ = nullptr;
        // Templates over a dummy address: nothing is read or written before
        // a kernel resets the base pointers, and without COPY_IF_OVERLAP the
        // iterator never compares addresses (viproc's copy-on-write already
        // separated outputs from overlapping inputs). Alignment is the only
        // property of the address it uses: element-aligned, as viproc
        // buffers.
        char* const dummy = template_address();
        PyArrayObject* op[kMaxOps] = {};
        PyArray_Descr* descrs[kMaxOps] = {};
        npy_uint32 op_flags[kMaxOps];
        bool ok = true;
        for (std::size_t i = 0; i < nit && ok; ++i) {
            const Layout& l = i < nin ? inputs[i] : i < nop ? outs[i - nin] : *mask;
            op[i] = template_array(l, dummy, i >= nin && i < nop);
            ok = op[i] != nullptr;
            if (i == nop) { // the where= mask: its own dtype (bool), not buffered for a cast
                descrs[i] = nullptr;
                op_flags[i] = NPY_ITER_READONLY | NPY_ITER_ARRAYMASK;
                continue;
            }
            // The loop's dtypes; operands of another dtype are cast in the
            // iterator's buffers, as in NumPy.
            descrs[i] = PyArray_DescrFromType(type_num(dtypes[i]));
            // NumPy's default operand flags for ufuncs (ufunc_object.c,
            // NPY_UFUNC_DEFAULT_INPUT_FLAGS / _OUTPUT_FLAGS with WRITEONLY;
            // ALLOCATE does nothing for a given output).
            // With where=, PyUFunc_GenericFunctionInternal adds WRITEMASKED
            // and execute_ufunc_loop READWRITE for a given output (WRITEONLY
            // for one it allocates).
            const npy_uint32 out_access = mask == nullptr ? NPY_ITER_WRITEONLY
                                          : out_given ? NPY_ITER_READWRITE | NPY_ITER_WRITEMASKED
                                                      : NPY_ITER_WRITEONLY | NPY_ITER_WRITEMASKED;
            op_flags[i] =
                i < nin ? NPY_ITER_READONLY | NPY_ITER_ALIGNED | NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE
                        : out_access | NPY_ITER_ALIGNED | NPY_ITER_NO_BROADCAST |
                              NPY_ITER_NO_SUBTYPE | NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;
        }
        PyObject* ufunc = loop->ufunc();
        const npy_uint32 uflags =
            ufunc != nullptr ? reinterpret_cast<PyUFuncObject*>(ufunc)->iter_flags : 0;
        // execute_ufunc_loop's flags, without COPY_IF_OVERLAP (see
        // above). An output NumPy allocates is never reversed (no
        // negative strides): DONT_NEGATE_STRIDES keeps it that way with
        // viproc's output.
        const npy_uint32 flags = uflags | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                 NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED | NPY_ITER_GROWINNER |
                                 NPY_ITER_DELAY_BUFALLOC |
                                 (out_given ? 0 : NPY_ITER_DONT_NEGATE_STRIDES);
        if (ok) {
            templ = NpyIter_AdvancedNew(static_cast<int>(nit), op, flags, NPY_KEEPORDER,
                                        NPY_UNSAFE_CASTING, op_flags, descrs, -1, nullptr, nullptr,
                                        bufsize);
        }
        pool = std::make_shared<IterTemplate>();
        if (templ != nullptr && mask != nullptr) {
            // masked: new iterators from the same arguments (IterTemplate)
            std::vector<PyArrayObject*> ops(op, op + nit);
            std::vector<PyArray_Descr*> ds(descrs, descrs + nit);
            std::vector<npy_uint32> fl(op_flags, op_flags + nit);
            for (std::size_t i = 0; i < nit; ++i) {
                Py_XINCREF(op[i]);
                Py_XINCREF(descrs[i]);
                pool->owned.push_back(reinterpret_cast<PyObject*>(op[i]));
                pool->owned.push_back(reinterpret_cast<PyObject*>(descrs[i]));
            }
            pool->fresh = [ops, ds, fl, flags, bufsize]() mutable {
                return NpyIter_AdvancedNew(static_cast<int>(ops.size()), ops.data(), flags,
                                           NPY_KEEPORDER, NPY_UNSAFE_CASTING, fl.data(), ds.data(),
                                           -1, nullptr, nullptr, bufsize);
            };
        }
        for (std::size_t i = 0; i < nit; ++i) {
            Py_XDECREF(op[i]); // the iterator holds its own references
            Py_XDECREF(descrs[i]);
        }
        if (templ == nullptr) {
            error = fetch_python_error();
            return nullptr;
        }
        if (NpyIter_IterationNeedsAPI(templ)) {
            NpyIter_Deallocate(templ);
            error = "iteration needs the Python API";
            return nullptr;
        }
        if (iter_templates_.size() >= kMaxIterTemplates) {
            iter_templates_.clear();
        }
        pool->templ = templ;
        pool->defer = std::move(defer);
        iter_templates_.emplace(std::move(key), pool);
    }
    NpyIter* iter = pool->get();
    if (iter == nullptr) {
        error = fetch_python_error();
        return nullptr;
    }
    auto kernel = std::make_shared<IterElementwiseKernel>(loop, iter, pool, mask != nullptr);
    kernel->iternext_ = NpyIter_GetIterNext(iter, nullptr);
    if (kernel->iternext_ == nullptr) {
        error = fetch_python_error();
        return nullptr; // the kernel releases the iterator
    }
    kernel->dataptr_ = NpyIter_GetDataPtrArray(iter);
    kernel->strideptr_ = NpyIter_GetInnerStrideArray(iter);
    kernel->countptr_ = NpyIter_GetInnerLoopSizePtr(iter);
    return kernel;
}

PyObject* scalar_object(DType dtype, const void* data) {
    PyArray_Descr* descr = PyArray_DescrFromType(type_num(dtype));
    PyObject* r = PyArray_Scalar(const_cast<void*>(data), descr, nullptr);
    Py_DECREF(descr);
    return r;
}

bool supported_ndarray(PyObject* obj) {
    if (!PyArray_Check(obj)) {
        return false;
    }
    auto* arr = reinterpret_cast<PyArrayObject*>(obj);
    DType d;
    return dtype_of(PyArray_TYPE(arr), d) && PyArray_ISALIGNED(arr) && PyArray_ISNOTSWAPPED(arr);
}

bool ndarray_layout(PyObject* obj, Layout& l, void*& data, std::size_t& nbytes) {
    if (!supported_ndarray(obj)) {
        return false;
    }
    auto* arr = reinterpret_cast<PyArrayObject*>(obj);
    dtype_of(PyArray_TYPE(arr), l.dtype);
    const int nd = PyArray_NDIM(arr);
    l.shape.assign(PyArray_SHAPE(arr), PyArray_SHAPE(arr) + nd);
    l.strides.assign(PyArray_STRIDES(arr), PyArray_STRIDES(arr) + nd);
    std::int64_t lo = 0;
    std::int64_t hi = static_cast<std::int64_t>(itemsize(l.dtype));
    for (int i = 0; i < nd; ++i) {
        const std::int64_t span = (l.shape[i] - 1) * l.strides[i];
        (span < 0 ? lo : hi) += l.shape[i] > 0 ? span : 0;
    }
    l.offset = -lo;
    data = PyArray_BYTES(arr) + lo;
    nbytes = static_cast<std::size_t>(hi - lo);
    return true;
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
    auto buffer = std::make_shared<CountableArray>(
        PyArray_BYTES(arr) + lo, static_cast<std::size_t>(hi - lo), std::move(owner));
    return rt.wrap(std::move(l), std::move(buffer));
}

ArrayPtr scalar_array(Runtime& rt, PyObject* value, DType dtype) {
    PyArray_Descr* descr = PyArray_DescrFromType(type_num(dtype));
    PyObject* arr = PyArray_FromAny(value, descr, 0, 0, 0, nullptr); // steals descr
    if (arr == nullptr) {
        return nullptr;
    }
    Layout l = Layout::contiguous(dtype, {});
    auto buffer = make_pooled<CountableArray>(l.nbytes());
    std::memcpy(buffer->data(), PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)), l.nbytes());
    Py_DECREF(arr);
    return rt.wrap(std::move(l), std::move(buffer));
}

ArrayPtr item_array(Runtime& rt, PyObject* value, DType dtype) {
    PyArray_Descr* descr = PyArray_DescrFromType(type_num(dtype));
    Layout l = Layout::contiguous(dtype, {});
    auto buffer = make_pooled<CountableArray>(l.nbytes());
    const int status = PyArray_Pack(descr, buffer->data(), value);
    Py_DECREF(descr);
    if (status < 0) {
        return nullptr;
    }
    return rt.wrap(std::move(l), std::move(buffer));
}

bool scalar_bytes(PyObject* value, DType dtype, std::vector<std::byte>& out) {
    PyArray_Descr* descr = PyArray_DescrFromType(type_num(dtype));
    PyObject* arr = PyArray_FromAny(value, descr, 0, 0, 0, nullptr); // steals descr
    if (arr == nullptr) {
        return false;
    }
    const auto* p =
        static_cast<const std::byte*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)));
    out.assign(p, p + itemsize(dtype));
    Py_DECREF(arr);
    return true;
}

namespace {

// Base object of exported NumPy views: keeps the buffer alive and, for a
// read-only export, holds one host_views count on it until the last NumPy view
// is gone. Created and released with the GIL (main thread).
constexpr const char* kExportCapsule = "viproc.export";

struct ExportHold {
    std::shared_ptr<CountableArray> buffer;
    bool counted; // holds a host_views count
};

void release_export(PyObject* capsule) {
    std::unique_ptr<ExportHold> hold(
        static_cast<ExportHold*>(PyCapsule_GetPointer(capsule, kExportCapsule)));
    if (hold && hold->counted) {
        --hold->buffer->host_views;
    }
}

} // namespace

namespace {

PyObject* ndarray_over(const Array& a, bool writable) {
    const Layout& l = a.layout();
    npy_intp shape[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    for (std::size_t i = 0; i < l.shape.size(); ++i) {
        shape[i] = static_cast<npy_intp>(l.shape[i]);
        strides[i] = static_cast<npy_intp>(l.strides[i]);
    }
    // Read-only unless the caller prepared the buffer for writing (see header).
    const int flags = writable ? NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE : NPY_ARRAY_ALIGNED;
    return PyArray_New(&PyArray_Type, static_cast<int>(l.shape.size()), shape, type_num(l.dtype),
                       strides, a.buffer()->data() + l.offset, 0, flags, nullptr);
}

// Makes `view` (over a's buffer, no base yet) an export: its base keeps the
// buffer alive and, read-only, counts as a reader. Returns false (view
// untouched, Python error set) on failure.
bool attach_export(const Array& a, PyObject* view, bool writable) {
    auto hold = std::make_unique<ExportHold>(ExportHold{a.buffer(), !writable});
    PyObject* capsule = PyCapsule_New(hold.get(), kExportCapsule, release_export);
    if (capsule == nullptr) {
        return false;
    }
    if (hold->counted) {
        ++hold->buffer->host_views;
    }
    hold.release(); // owned by the capsule now
    if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(view), capsule) < 0) {
        Py_DECREF(capsule);
        return false;
    }
    return true; // the capsule reference went to the view
}

} // namespace

PyObject* export_ndarray(const Array& a, bool writable) {
    PyObject* view = ndarray_over(a, writable);
    if (view == nullptr || !attach_export(a, view, writable)) {
        Py_XDECREF(view);
        return nullptr;
    }
    return view;
}

// Writable as NumPy's own arrays are (NumPy copies read-only inputs in some
// reductions, e.g. argmax), for the reductions the bridge calls: they read
// it only. Read-only once pinned.
PyObject* borrow_ndarray(const Array& a) { return ndarray_over(a, true); }

int ndarray_dtype_code(PyObject* nd) {
    if (!PyArray_Check(nd)) {
        return -1;
    }
    PyArray_Descr* d = PyArray_DESCR(reinterpret_cast<PyArrayObject*>(nd));
    DType t;
    if (!PyArray_ISNBO(d->byteorder) || !dtype_of(d->type_num, t)) {
        return -1;
    }
    return static_cast<int>(t);
}

int ndarray_ndim_dense(PyObject* nd, bool* dense) {
    if (!PyArray_Check(nd)) {
        return -1;
    }
    PyArrayObject* a = reinterpret_cast<PyArrayObject*>(nd);
    *dense = (PyArray_IS_C_CONTIGUOUS(a) || PyArray_IS_F_CONTIGUOUS(a)) && PyArray_ISALIGNED(a);
    return PyArray_NDIM(a);
}

bool ndarrays_overlap(PyObject* a, PyObject* b) {
    auto extent = [](PyArrayObject* x, char** lo, char** hi) {
        *lo = *hi = PyArray_BYTES(x);
        for (int i = 0; i < PyArray_NDIM(x); ++i) {
            if (PyArray_DIM(x, i) == 0) {
                *hi = *lo; // empty
                return;
            }
            const npy_intp step = PyArray_STRIDE(x, i) * (PyArray_DIM(x, i) - 1);
            (step < 0 ? *lo : *hi) += step;
        }
        *hi += PyArray_ITEMSIZE(x);
    };
    char *alo, *ahi, *blo, *bhi;
    extent(reinterpret_cast<PyArrayObject*>(a), &alo, &ahi);
    extent(reinterpret_cast<PyArrayObject*>(b), &blo, &bhi);
    return alo < ahi && blo < bhi && alo < bhi && blo < ahi;
}

bool ndarray_shape(PyObject* nd, int* ndim, std::int64_t* shape) {
    if (!PyArray_Check(nd)) {
        return false;
    }
    PyArrayObject* a = reinterpret_cast<PyArrayObject*>(nd);
    *ndim = PyArray_NDIM(a);
    for (int i = 0; i < *ndim; ++i) {
        shape[i] = PyArray_DIM(a, i);
    }
    return true;
}

PyObject* run_trivial_into_ndarray(const UfuncLoop& loop, const DType* dtypes,
                                   const Array* const* ins, PyObject* const* nds, int nin,
                                   int ndim, const std::int64_t* shape) {
    if (loop.raises_through_pyapi() || nin > 3) {
        return nullptr;
    }
    npy_intp n = 1;
    npy_intp dims[NPY_MAXDIMS];
    for (int i = 0; i < ndim; ++i) {
        dims[i] = static_cast<npy_intp>(shape[i]);
        n *= dims[i];
    }
    char* data[4];
    npy_intp strides[4];
    for (int i = 0; i < nin; ++i) {
        if (nds != nullptr && nds[i] != nullptr) {
            PyArrayObject* x = reinterpret_cast<PyArrayObject*>(nds[i]);
            const int xd = PyArray_NDIM(x);
            if (!same_dtype(PyArray_DESCR(x)->type_num, dtypes[i]) || !PyArray_ISALIGNED(x) ||
                !PyArray_ISNOTSWAPPED(x)) {
                return nullptr;
            }
            if (xd == 0) {
                strides[i] = 0;
            } else if (xd == ndim && PyArray_IS_C_CONTIGUOUS(x) &&
                       std::equal(PyArray_DIMS(x), PyArray_DIMS(x) + xd, shape)) {
                strides[i] = static_cast<npy_intp>(itemsize(dtypes[i]));
            } else {
                return nullptr;
            }
            data[i] = static_cast<char*>(PyArray_DATA(x));
            continue;
        }
        const Array& a = *ins[i];
        const Layout& l = a.layout();
        if (l.dtype != dtypes[i]) {
            return nullptr; // a cast: NumPy's iterator
        }
        if (l.shape.empty()) {
            strides[i] = 0;
        } else if (static_cast<int>(l.shape.size()) == ndim &&
                   std::equal(l.shape.begin(), l.shape.end(), shape) && l.c_contiguous()) {
            strides[i] = static_cast<npy_intp>(itemsize(l.dtype));
        } else {
            return nullptr;
        }
        data[i] = reinterpret_cast<char*>(a.buffer()->data() + l.offset);
    }
    PyObject* out = PyArray_SimpleNew(ndim, dims, type_num(dtypes[nin]));
    if (out == nullptr) {
        return nullptr;
    }
    data[nin] = static_cast<char*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out)));
    strides[nin] = static_cast<npy_intp>(itemsize(dtypes[nin]));
    std::feclearexcept(FE_ALL_EXCEPT);
    const npy_intp count[1] = {n};
    const int st = n > 0 ? loop.call(data, count, strides) : 0;
    if (st != 0 || std::fetestexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID | FE_UNDERFLOW)) {
        Py_DECREF(out);
        PyErr_Clear();
        return nullptr; // NumPy runs it again: its errors and warnings
    }
    return out;
}

bool pin_borrowed(const Array& a, PyObject* view) {
    PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(view), NPY_ARRAY_WRITEABLE);
    return attach_export(a, view, false);
}

} // namespace viproc::numpy
