// Implementation of the public C ABI (include/viproc/viproc.h).

#include "viproc/viproc.h"

#include "kernels/numpy_ufunc.hpp"
#include "runtime/assign.hpp"
#include "runtime/runtime.hpp"

#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

using namespace viproc;

namespace {

// Python objects whose last C++ reference died on a thread without the GIL.
// They are released on the main thread at the next C ABI call.
struct Graveyard {
    std::mutex mutex;
    std::vector<PyObject*> objects;

    void drain() {
        std::vector<PyObject*> dead;
        {
            std::lock_guard lock(mutex);
            dead.swap(objects);
        }
        for (PyObject* o : dead) {
            Py_DECREF(o);
        }
    }
};

void set_error(char* err, std::size_t errlen, const std::string& msg) {
    if (err != nullptr && errlen > 0) {
        std::snprintf(err, errlen, "%s", msg.c_str());
    }
}

static_assert(static_cast<int>(DType::Bool) == VP_BOOL &&
                  static_cast<int>(DType::Int32) == VP_INT32 &&
                  static_cast<int>(DType::Int64) == VP_INT64 &&
                  static_cast<int>(DType::Float32) == VP_FLOAT32 &&
                  static_cast<int>(DType::Float64) == VP_FLOAT64 &&
                  static_cast<int>(DType::Complex64) == VP_COMPLEX64 &&
                  static_cast<int>(DType::Complex128) == VP_COMPLEX128,
              "vp_dtype and DType must match");

DType to_dtype(vp_dtype d) { return static_cast<DType>(d); }
vp_dtype to_vp(DType d) { return static_cast<vp_dtype>(d); }

} // namespace

struct vp_runtime {
    std::unique_ptr<Runtime> rt;
    // Issue sites, interned: tasks keep a `const char*` to them. Grows with
    // the number of distinct source locations only.
    std::unordered_set<std::string> sites;
    const char* intern(const char* site) { return sites.emplace(site).first->c_str(); }
    std::unique_ptr<numpy::LoopCache> loops;
    std::shared_ptr<Graveyard> graveyard = std::make_shared<Graveyard>();
};

struct vp_array {
    ArrayPtr array;
};

extern "C" {

vp_runtime* vp_runtime_create(size_t workers, size_t max_active_tasks, char* err, size_t errlen) {
    if (!numpy::init()) {
        PyErr_Clear();
        set_error(err, errlen, "cannot import the NumPy C API");
        return nullptr;
    }
    RuntimeOptions options;
    options.workers = workers;
    if (max_active_tasks != 0) {
        options.max_active_tasks = max_active_tasks;
    }
    auto* r = new vp_runtime();
    r->rt = std::make_unique<Runtime>(options);
    r->loops = std::make_unique<numpy::LoopCache>();
    return r;
}

void vp_wait_all(vp_runtime* rt) { rt->rt->wait_all(); }

void vp_runtime_destroy(vp_runtime* rt) {
    Py_BEGIN_ALLOW_THREADS;
    rt->rt.reset(); // waits for all tasks, stops the workers
    Py_END_ALLOW_THREADS;
    rt->loops.reset();
    rt->graveyard->drain();
    delete rt;
}

size_t vp_worker_count(const vp_runtime* rt) { return rt->rt->worker_count(); }
size_t vp_active_tasks(const vp_runtime* rt) { return rt->rt->active_tasks(); }

int vp_array_from_ndarray(vp_runtime* rt, PyObject* ndarray, vp_array** out, char* err,
                          size_t errlen) {
    rt->graveyard->drain();
    if (!numpy::supported_ndarray(ndarray)) {
        set_error(err, errlen, "unsupported dtype or memory layout");
        return VP_UNSUPPORTED;
    }
    Py_INCREF(ndarray);
    std::weak_ptr<Graveyard> weak = rt->graveyard;
    std::shared_ptr<void> owner(ndarray, [weak](void* p) {
        auto* o = static_cast<PyObject*>(p);
        if (PyGILState_Check()) {
            Py_DECREF(o);
        } else if (auto g = weak.lock()) {
            std::lock_guard lock(g->mutex);
            g->objects.push_back(o);
        } // else: runtime gone, interpreter shutting down; leak
    });
    std::string e;
    ArrayPtr a = numpy::wrap_ndarray(*rt->rt, ndarray, e, std::move(owner));
    if (!a) {
        set_error(err, errlen, e);
        return VP_UNSUPPORTED;
    }
    *out = new vp_array{std::move(a)};
    return VP_OK;
}

int vp_array_from_scalar(vp_runtime* rt, PyObject* value, vp_dtype dtype, vp_array** out) {
    ArrayPtr a = numpy::scalar_array(*rt->rt, value, to_dtype(dtype));
    if (!a) {
        return VP_PYERR;
    }
    *out = new vp_array{std::move(a)};
    return VP_OK;
}

vp_array* vp_array_view(vp_runtime* rt, const vp_array* base, int ndim, const int64_t* shape,
                        const int64_t* strides, int64_t offset) {
    Layout l;
    l.dtype = base->array->layout().dtype;
    l.shape.assign(shape, shape + ndim);
    l.strides.assign(strides, strides + ndim);
    l.offset = base->array->layout().offset + offset;
    return new vp_array{rt->rt->view(base->array, std::move(l))};
}

void vp_array_release(vp_array* a) { delete a; }

int vp_array_ndim(const vp_array* a) { return static_cast<int>(a->array->layout().shape.size()); }
const int64_t* vp_array_shape(const vp_array* a) { return a->array->layout().shape.data(); }
const int64_t* vp_array_strides(const vp_array* a) { return a->array->layout().strides.data(); }
vp_dtype vp_array_dtype(const vp_array* a) { return to_vp(a->array->layout().dtype); }
int vp_array_ready(vp_array* a) { return a->array->ready() ? 1 : 0; }
int vp_array_same_storage(const vp_array* a, const vp_array* b) {
    return a->array->storage() == b->array->storage() ? 1 : 0;
}

int vp_array_wait(vp_runtime* rt, vp_array* a, char* err, size_t errlen) {
    std::optional<TaskError> e = rt->rt->wait(*a->array);
    if (e) {
        set_error(err, errlen, e->message + " (in op issued at " + e->issue_site + ")");
        return VP_ERROR;
    }
    return VP_OK;
}

PyObject* vp_array_to_ndarray(const vp_array* a) { return numpy::to_ndarray(*a->array); }

int vp_ufunc(vp_runtime* rt, PyObject* ufunc, int nin, vp_array* const* inputs, vp_dtype out_dtype,
             int out_ndim, const int64_t* out_shape, vp_array* out, vp_array** result,
             const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    std::vector<DType> dtypes;
    std::vector<ArrayPtr> ins;
    for (int i = 0; i < nin; ++i) {
        dtypes.push_back(inputs[i]->array->layout().dtype);
        ins.push_back(inputs[i]->array);
    }
    dtypes.push_back(to_dtype(out_dtype));
    std::string e;
    const numpy::UfuncLoop* loop = rt->loops->get(ufunc, dtypes, e);
    if (loop == nullptr) {
        PyErr_Clear();
        set_error(err, errlen, e);
        return VP_UNSUPPORTED;
    }
    if (loop->nin() != nin || loop->nout() != 1) {
        set_error(err, errlen, "only ufuncs with one output are supported");
        return VP_UNSUPPORTED;
    }
    Shape shape(out_shape, out_shape + out_ndim);
    const std::shared_ptr<Kernel>& kernel = loop->elementwise_kernel();
    const char* interned = rt->intern(site);
    if (out != nullptr) {
        const Layout& ol = out->array->layout();
        if (ol.shape != shape || ol.dtype != to_dtype(out_dtype)) {
            set_error(err, errlen, "out= array has the wrong shape or dtype");
            return VP_INVALID;
        }
        std::vector<ArrayPtr> io{out->array};
        rt->rt->issue(kernel, ins, io, {}, interned);
        *result = nullptr;
        return VP_OK;
    }
    Layout l = Layout::contiguous(to_dtype(out_dtype), std::move(shape));
    std::vector<ArrayPtr> res = rt->rt->issue(kernel, ins, {}, std::span(&l, 1), interned);
    *result = new vp_array{std::move(res[0])};
    return VP_OK;
}

int vp_assign(vp_runtime* rt, vp_array* dst, vp_array* src, const char* site, char* err,
              size_t errlen) {
    rt->graveyard->drain();
    const Layout& d = dst->array->layout();
    const Layout& s = src->array->layout();
    Shape b;
    if (d.dtype != s.dtype || !broadcast_shapes(d.shape, s.shape, b) || b != d.shape) {
        set_error(err, errlen, "assign: dtype mismatch or shapes do not broadcast");
        return VP_INVALID;
    }
    std::vector<ArrayPtr> ins{src->array};
    std::vector<ArrayPtr> io{dst->array};
    static const std::shared_ptr<Kernel> assign = make_assign_kernel();
    rt->rt->issue(assign, ins, io, {}, rt->intern(site));
    return VP_OK;
}

} // extern "C"
