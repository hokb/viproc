// Implementation of the public C ABI (include/viproc/viproc.h).

#include "viproc/viproc.h"

#include "kernels/numpy_matmul.hpp"
#include "kernels/numpy_reduce.hpp"
#include "kernels/numpy_ufunc.hpp"
#include "runtime/assign.hpp"
#include "runtime/element_pool.hpp"
#include "runtime/index.hpp"
#include "runtime/runtime.hpp"

#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace viproc;

namespace {

// Python objects whose last C++ reference died on a thread without the GIL.
// They are released on the main thread at the next C ABI call.
struct Graveyard {
    std::mutex mutex;
    std::vector<PyObject*> objects;
    std::vector<std::function<void()>> releases; // kernels' Python state

    void drain() {
        std::vector<PyObject*> dead;
        std::vector<std::function<void()>> todo;
        {
            std::lock_guard lock(mutex);
            dead.swap(objects);
            todo.swap(releases);
        }
        for (PyObject* o : dead) {
            Py_DECREF(o);
        }
        for (auto& f : todo) {
            f();
        }
    }
};

// Releases `f` now if this thread holds the GIL, else at the next C ABI
// call (main thread). After the runtime is gone (interpreter shutdown) the
// objects leak.
numpy::DeferRelease defer_release(const std::shared_ptr<Graveyard>& graveyard) {
    std::weak_ptr<Graveyard> weak = graveyard;
    return [weak](std::function<void()> f) {
        if (PyGILState_Check()) {
            f();
        } else if (auto g = weak.lock()) {
            std::lock_guard lock(g->mutex);
            g->releases.push_back(std::move(f));
        }
    };
}

// Keeps `obj` (a new reference) alive as a buffer owner; dropped with the GIL.
std::shared_ptr<void> python_owner(const std::shared_ptr<Graveyard>& graveyard, PyObject* obj) {
    std::weak_ptr<Graveyard> weak = graveyard;
    return std::shared_ptr<void>(obj, [weak](void* p) {
        auto* o = static_cast<PyObject*>(p);
        if (PyGILState_Check()) {
            Py_DECREF(o);
        } else if (auto g = weak.lock()) {
            std::lock_guard lock(g->mutex);
            g->objects.push_back(o);
        } // else: runtime gone, interpreter shutting down; leak
    });
}

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

struct vp_loop {
    const numpy::UfuncLoop* loop; // owned by vp_runtime::loops
    std::vector<DType> dtypes;    // inputs, then the output
};

struct vp_runtime {
    std::unique_ptr<Runtime> rt;
    // Issue sites, interned: tasks keep a `const char*` to them. Grows with
    // the number of distinct source locations only.
    std::unordered_set<std::string> sites;
    const char* intern(const char* site) { return sites.emplace(site).first->c_str(); }
    std::unique_ptr<numpy::LoopCache> loops;
    // One handle per resolved loop (node-based: stable addresses).
    std::unordered_map<const numpy::UfuncLoop*, vp_loop> handles;
    std::shared_ptr<Graveyard> graveyard = std::make_shared<Graveyard>();
    // One matmul kernel per resolved loop.
    std::unordered_map<const numpy::UfuncLoop*, std::shared_ptr<Kernel>> matmul_kernels;
};

// A vp_array* is an Array* holding one reference (no wrapper allocation).
namespace {
Array& arr(const vp_array* a) { return *reinterpret_cast<Array*>(const_cast<vp_array*>(a)); }
ArrayPtr ref(const vp_array* a) { return ArrayPtr(&arr(a)); } // a new reference
vp_array* hand_out(ArrayPtr a) { return reinterpret_cast<vp_array*>(a.release()); }
} // namespace

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
    ElementPool::instance().trim(); // cached blocks are not kept across runtimes
}

size_t vp_worker_count(const vp_runtime* rt) { return rt->rt->worker_count(); }
size_t vp_active_tasks(const vp_runtime* rt) { return rt->rt->active_tasks(); }

void vp_set_offload_policy(vp_runtime* rt, int64_t sync_below, int adaptive) {
    OffloadOptions o = rt->rt->offload_policy().options();
    o.sync_below = sync_below;
    o.adaptive = adaptive != 0;
    rt->rt->set_offload_options(o);
}

uint64_t vp_direct_count(const vp_runtime* rt) { return rt->rt->direct_count(); }

void vp_offload_counts(const vp_runtime* rt, uint64_t* inline_ops, uint64_t* offloaded_ops) {
    *inline_ops = rt->rt->offload_policy().inline_count();
    *offloaded_ops = rt->rt->offload_policy().offload_count();
}

void vp_pool_get_stats(vp_pool_stats* out) {
    const ElementPool::Stats s = ElementPool::instance().stats();
    *out = {s.hits, s.misses, s.cached, s.dropped, ElementPool::instance().limit()};
}

void vp_pool_trim(void) { ElementPool::instance().trim(); }

void vp_pool_set_limit(uint64_t bytes) {
    ElementPool::instance().set_limit(static_cast<std::size_t>(bytes));
}

int vp_array_from_ndarray(vp_runtime* rt, PyObject* ndarray, vp_array** out, char* err,
                          size_t errlen) {
    rt->graveyard->drain();
    if (!numpy::supported_ndarray(ndarray)) {
        set_error(err, errlen, "unsupported dtype or memory layout");
        return VP_UNSUPPORTED;
    }
    Py_INCREF(ndarray);
    std::shared_ptr<void> owner = python_owner(rt->graveyard, ndarray);
    std::string e;
    ArrayPtr a = numpy::wrap_ndarray(*rt->rt, ndarray, e, std::move(owner));
    if (!a) {
        set_error(err, errlen, e);
        return VP_UNSUPPORTED;
    }
    *out = hand_out(std::move(a));
    return VP_OK;
}

const char* vp_intern(vp_runtime* rt, const char* site) { return rt->intern(site); }

int vp_array_from_scalar(vp_runtime* rt, PyObject* value, vp_dtype dtype, vp_array** out) {
    ArrayPtr a = numpy::scalar_array(*rt->rt, value, to_dtype(dtype));
    if (!a) {
        return VP_PYERR;
    }
    *out = hand_out(std::move(a));
    return VP_OK;
}

PyObject* vp_array_scalar(const vp_array* a) {
    const Array& x = arr(a);
    return numpy::scalar_object(x.layout().dtype, x.buffer()->data() + x.layout().offset);
}

vp_array* vp_array_reshape(vp_runtime* rt, const vp_array* a, int ndim, const int64_t* shape,
                           char order) {
    const Layout& l = arr(a).layout();
    Layout out;
    if (!reshape_view(l, Shape(shape, shape + ndim), order, out)) {
        return nullptr;
    }
    return hand_out(rt->rt->view(ref(a), std::move(out)));
}

vp_array* vp_array_view(vp_runtime* rt, const vp_array* base, vp_dtype dtype, int ndim,
                        const int64_t* shape, const int64_t* strides, int64_t offset) {
    Layout l;
    l.dtype = to_dtype(dtype);
    l.shape.assign(shape, shape + ndim);
    l.strides.assign(strides, strides + ndim);
    l.offset = arr(base).layout().offset + offset;
    return hand_out(rt->rt->view(ref(base), std::move(l)));
}

void vp_array_release(vp_array* a) { ArrayPtr::adopt(&arr(a)).reset(); }

int vp_array_ndim(const vp_array* a) { return static_cast<int>(arr(a).layout().shape.size()); }
const int64_t* vp_array_shape(const vp_array* a) { return arr(a).layout().shape.data(); }
const int64_t* vp_array_strides(const vp_array* a) { return arr(a).layout().strides.data(); }
vp_dtype vp_array_dtype(const vp_array* a) { return to_vp(arr(a).layout().dtype); }
int vp_array_ready(vp_array* a) { return arr(a).ready() ? 1 : 0; }
int vp_array_same_storage(const vp_array* a, const vp_array* b) {
    return arr(a).storage() == arr(b).storage() ? 1 : 0;
}

size_t vp_array_reusable_bytes(const vp_array* a) {
    const Array& x = arr(a);
    const Buffer& b = *x.buffer();
    const Layout& l = x.layout();
    const bool ok = x.ref_count() == 1 && x.storage().use_count() == 1 && b.capacity() != 0 &&
                    !b.has_readers() && l.offset == 0 && l.c_contiguous() &&
                    l.nbytes() == b.nbytes();
    return ok ? b.nbytes() : 0;
}

int vp_array_wait(vp_runtime* rt, vp_array* a, char* err, size_t errlen) {
    std::optional<TaskError> e = rt->rt->wait(arr(a));
    if (e) {
        set_error(err, errlen, e->message + " (in op issued at " + e->issue_site + ")");
        return VP_ERROR;
    }
    return VP_OK;
}

PyObject* vp_array_export(const vp_array* a, int writable) {
    return numpy::export_ndarray(arr(a), writable != 0);
}

PyObject* vp_array_borrow(const vp_array* a) { return numpy::borrow_ndarray(arr(a)); }

int vp_array_pin(const vp_array* a, PyObject* view) {
    return numpy::pin_borrowed(arr(a), view) ? VP_OK : VP_PYERR;
}

int vp_array_prepare_write(vp_runtime* rt, vp_array* a, char* err, size_t errlen) {
    std::optional<TaskError> e = rt->rt->prepare_host_write(arr(a));
    if (e) {
        set_error(err, errlen, e->message + " (in op issued at " + e->issue_site + ")");
        return VP_ERROR;
    }
    return VP_OK;
}

int vp_ufunc_loop(vp_runtime* rt, PyObject* ufunc, int nin, const vp_dtype* dtypes,
                  const vp_loop** out, char* err, size_t errlen) {
    std::vector<DType> d;
    for (int i = 0; i <= nin; ++i) {
        d.push_back(to_dtype(dtypes[i]));
    }
    std::string e;
    const numpy::UfuncLoop* loop = rt->loops->get(ufunc, d, e);
    if (loop == nullptr) {
        PyErr_Clear();
        set_error(err, errlen, e);
        return VP_UNSUPPORTED;
    }
    if (loop->nin() != nin || loop->nout() != 1) {
        set_error(err, errlen, "only ufuncs with one output are supported");
        return VP_UNSUPPORTED;
    }
    *out = &rt->handles.try_emplace(loop, vp_loop{loop, std::move(d)}).first->second;
    return VP_OK;
}

} // extern "C"

namespace {

// F-contiguous as NumPy's flag: extent-1 dimensions may have any stride.
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

// The kernel of an element-wise op: the loop's shared kernel where NumPy
// calls the loop once over all elements, else one driven by NumPy's iterator
// (LoopCache::iter_elementwise), which also casts inputs of another dtype.
// nullptr if NumPy cannot run it so.
std::shared_ptr<Kernel> elementwise_kernel(vp_runtime* rt, const vp_loop& loop,
                                           std::span<const ArrayPtr> ins, const Layout& out,
                                           bool out_given, char* err, size_t errlen) {
    const Layout* ptrs[3];
    bool cast = false;
    for (std::size_t i = 0; i < ins.size(); ++i) {
        ptrs[i] = &ins[i]->layout();
        cast = cast || ptrs[i]->dtype != loop.dtypes[i];
    }
    std::intptr_t strides[4];
    if (out.size() == 0 ||
        (!cast && numpy::single_call(std::span(ptrs, ins.size()), out, strides))) {
        return loop.loop->elementwise_kernel();
    }
    Layout layouts[3];
    for (std::size_t i = 0; i < ins.size(); ++i) {
        layouts[i] = ins[i]->layout();
    }
    const std::span<const Layout> ls(layouts, ins.size());
    std::string e;
    std::shared_ptr<Kernel> k = rt->loops->iter_elementwise(
        loop.loop, loop.dtypes, ls, out, out_given, defer_release(rt->graveyard), e);
    if (!k) {
        PyErr_Clear();
        set_error(err, errlen, e);
    }
    return k;
}

// Inputs of another dtype than the loop's: NumPy's ufuncs copy 0-d and short
// 1-d ones (up to the buffer size) into a cast array first and keep the
// single loop call (check_for_trivial_loop); longer and n-d ones are cast
// in the iterator's buffers (elementwise_kernel). Replaces the first kind by
// the result of an issued cast task. False if NumPy cannot cast.
bool cast_small_inputs(vp_runtime* rt, const vp_loop& loop, std::span<ArrayPtr> ins,
                       const char* site, char* err, size_t errlen) {
    long bufsize = -1;
    for (std::size_t i = 0; i < ins.size(); ++i) {
        const Layout& l = ins[i]->layout();
        const DType to = loop.dtypes[i];
        if (l.dtype == to || l.size() == 0 || l.shape.size() > 1) {
            continue;
        }
        if (bufsize < 0) {
            bufsize = rt->loops->buffersize();
        }
        if (l.shape.size() == 1 && l.shape[0] > bufsize) {
            continue;
        }
        std::string e;
        const Layout dst = Layout::contiguous(to, l.shape);
        std::shared_ptr<Kernel> k =
            numpy::make_cast_kernel(l, dst, bufsize, defer_release(rt->graveyard), e);
        if (!k) {
            PyErr_Clear();
            set_error(err, errlen, e);
            return false;
        }
        ArrayPtr res;
        const ArrayPtr in[] = {ins[i]};
        rt->rt->issue(k, in, {}, std::span(&dst, 1), site, std::span(&res, 1));
        ins[i] = std::move(res);
    }
    return true;
}

} // namespace

extern "C" {

int vp_ufunc_issue(vp_runtime* rt, const vp_loop* loop, vp_array* const* inputs, int out_ndim,
                   const int64_t* out_shape, vp_array* out, vp_array** result, const char* site,
                   char* err, size_t errlen) {
    rt->graveyard->drain();
    const int nin = loop->loop->nin();
    ArrayPtr ins[3];
    if (nin > 3) {
        set_error(err, errlen, "too many inputs");
        return VP_UNSUPPORTED;
    }
    for (int i = 0; i < nin; ++i) {
        ins[i] = ref(inputs[i]);
    }
    if (!cast_small_inputs(rt, *loop, std::span(ins, static_cast<std::size_t>(nin)), site, err,
                           errlen)) {
        return VP_UNSUPPORTED;
    }
    const DType out_dtype = loop->dtypes[static_cast<std::size_t>(nin)];
    Shape shape(out_shape, out_shape + out_ndim);
    const std::span<const ArrayPtr> in_span(ins, static_cast<std::size_t>(nin));
    if (out != nullptr) {
        const Layout& ol = arr(out).layout();
        if (ol.shape != shape || ol.dtype != out_dtype) {
            set_error(err, errlen, "out= array has the wrong shape or dtype");
            return VP_INVALID;
        }
        const ArrayPtr o = ref(out);
        std::shared_ptr<Kernel> k = elementwise_kernel(rt, *loop, in_span, ol, true, err, errlen);
        if (!k) {
            return VP_UNSUPPORTED;
        }
        rt->rt->issue(k, in_span, std::span(&o, 1), {}, site);
        *result = nullptr;
        return VP_OK;
    }
    // NumPy's result layout (order='K'); C order unless an input is not.
    bool all_c = true;
    for (int i = 0; i < nin; ++i) {
        const Layout& il = ins[i]->layout();
        all_c = all_c && (il.c_contiguous() || il.size() <= 1);
    }
    // NumPy's trivial loop allocates F order when every input is a 0-d array
    // or F- (and not C-) contiguous with the result's shape, without a cast; that fixes the
    // strides of extent-1 dimensions too, which keep_order() would not.
    bool all_f = !all_c && out_ndim > 1;
    for (int i = 0; i < nin && all_f; ++i) {
        const Layout& il = ins[i]->layout();
        all_f = il.shape.empty() || (il.shape == shape && f_contiguous(il) && !il.c_contiguous() &&
                                     il.dtype == loop->dtypes[i]);
    }
    Layout l;
    if (all_c) {
        l = Layout::contiguous(out_dtype, std::move(shape));
    } else if (all_f) {
        l.dtype = out_dtype;
        l.strides.resize(shape.size());
        std::int64_t step = static_cast<std::int64_t>(itemsize(out_dtype));
        for (std::size_t i = 0; i < shape.size(); ++i) {
            l.strides[i] = step;
            step *= shape[i];
        }
        l.shape = std::move(shape);
    } else {
        Layout in_layouts[3];
        for (int i = 0; i < nin; ++i) {
            in_layouts[i] = ins[i]->layout();
        }
        l = Layout::keep_order(out_dtype, std::move(shape),
                               std::span<const Layout>(in_layouts, static_cast<std::size_t>(nin)));
    }
    std::shared_ptr<Kernel> k = elementwise_kernel(rt, *loop, in_span, l, false, err, errlen);
    if (!k) {
        return VP_UNSUPPORTED;
    }
    ArrayPtr res;
    rt->rt->issue(k, in_span, {}, std::span(&l, 1), site, std::span(&res, 1));
    *result = hand_out(std::move(res));
    return VP_OK;
}

int vp_ufunc(vp_runtime* rt, PyObject* ufunc, int nin, vp_array* const* inputs, vp_dtype out_dtype,
             int out_ndim, const int64_t* out_shape, vp_array* out, vp_array** result,
             const char* site, char* err, size_t errlen) {
    std::vector<vp_dtype> dtypes;
    for (int i = 0; i < nin; ++i) {
        dtypes.push_back(to_vp(arr(inputs[i]).layout().dtype));
    }
    dtypes.push_back(out_dtype);
    const vp_loop* loop = nullptr;
    const int st = vp_ufunc_loop(rt, ufunc, nin, dtypes.data(), &loop, err, errlen);
    if (st != VP_OK) {
        return st;
    }
    return vp_ufunc_issue(rt, loop, inputs, out_ndim, out_shape, out, result, site, err, errlen);
}

int vp_astype(vp_runtime* rt, vp_array* in, vp_dtype dtype, char order, vp_array** result,
              const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    const Layout& l = arr(in).layout();
    const DType to = to_dtype(dtype);
    Layout out;
    if (order == 'F') {
        out.dtype = to;
        out.shape = l.shape;
        out.strides.resize(l.shape.size());
        std::int64_t step = static_cast<std::int64_t>(itemsize(to));
        for (std::size_t i = 0; i < l.shape.size(); ++i) {
            out.strides[i] = step;
            if (l.shape[i] != 0) {
                step *= l.shape[i];
            }
        }
    } else if (order == 'C' || l.c_contiguous() || l.shape.size() <= 1) {
        out = Layout::contiguous(to, l.shape);
    } else if (order == 'K') {
        // PyArray_NewLikeArray(KEEPORDER): F order if `in` is F-contiguous,
        // else axes by decreasing |stride| (stable), C order among equals.
        out.dtype = to;
        out.shape = l.shape;
        out.strides.resize(l.shape.size());
        std::vector<std::size_t> perm(l.shape.size());
        for (std::size_t i = 0; i < perm.size(); ++i) {
            perm[i] = i;
        }
        if (!f_contiguous(l)) {
            std::stable_sort(perm.begin(), perm.end(), [&](std::size_t a, std::size_t b) {
                return std::llabs(l.strides[a]) > std::llabs(l.strides[b]);
            });
        } else {
            std::reverse(perm.begin(), perm.end());
        }
        std::int64_t step = static_cast<std::int64_t>(itemsize(to));
        for (std::size_t k = perm.size(); k-- > 0;) {
            out.strides[perm[k]] = step;
            step *= l.shape[perm[k]];
        }
    } else {
        set_error(err, errlen, "unsupported order");
        return VP_UNSUPPORTED;
    }
    if (l.size() == 0) {
        set_error(err, errlen, "empty");
        return VP_UNSUPPORTED;
    }
    std::string e;
    std::shared_ptr<Kernel> k =
        numpy::make_cast_kernel(l, out, rt->loops->buffersize(), defer_release(rt->graveyard), e);
    if (!k) {
        PyErr_Clear(); // e.g. a ComplexWarning raised as an error: NumPy raises it again
        set_error(err, errlen, e);
        return VP_UNSUPPORTED;
    }
    const ArrayPtr ins[] = {ref(in)};
    ArrayPtr res;
    rt->rt->issue(k, ins, {}, std::span(&out, 1), site, std::span(&res, 1));
    *result = hand_out(std::move(res));
    return VP_OK;
}

} // extern "C"

namespace {

// The IndexSpec of an indexing call and the shape of its result.
bool index_spec(const Layout& view, int nidx, vp_array* const* idx, const int* idx_axes, int nres,
                const int* result_axes, IndexSpec& spec, Shape& shape, char* err, size_t errlen) {
    const int nd = static_cast<int>(view.shape.size());
    for (int j = 0; j < nidx; ++j) {
        const Layout& il = arr(idx[j]).layout();
        Shape b;
        if ((il.dtype != DType::Int32 && il.dtype != DType::Int64) || idx_axes[j] < 0 ||
            idx_axes[j] >= nd || !broadcast_shapes(spec.bshape, il.shape, b)) {
            set_error(err, errlen, "index arrays: wrong dtype, axis or shape");
            return false;
        }
        spec.bshape = std::move(b);
        spec.index_axes.push_back(idx_axes[j]);
    }
    for (int k = 0; k < nres; ++k) {
        const int a = result_axes[k];
        if (a >= nd || (a < 0 && -1 - a >= static_cast<int>(spec.bshape.size()))) {
            set_error(err, errlen, "result axes out of range");
            return false;
        }
        spec.result_axes.push_back(a);
        shape.push_back(a >= 0 ? view.shape[static_cast<std::size_t>(a)]
                               : spec.bshape[static_cast<std::size_t>(-1 - a)]);
    }
    return true;
}

} // namespace

extern "C" {

int vp_gather(vp_runtime* rt, vp_array* src, int nidx, vp_array* const* idx, const int* idx_axes,
              int nres, const int* result_axes, const int64_t* result_strides, vp_array** result,
              const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    const Layout& v = arr(src).layout();
    IndexSpec spec;
    Shape shape;
    if (!index_spec(v, nidx, idx, idx_axes, nres, result_axes, spec, shape, err, errlen)) {
        return VP_INVALID;
    }
    Layout out = Layout::contiguous(v.dtype, std::move(shape));
    if (result_strides != nullptr) {
        out.strides.assign(result_strides, result_strides + nres);
        if (!out.dense() && out.size() != 0) {
            set_error(err, errlen, "gather: result strides are not dense");
            return VP_INVALID;
        }
    }
    std::vector<ArrayPtr> ins{ref(src)};
    for (int j = 0; j < nidx; ++j) {
        ins.push_back(ref(idx[j]));
    }
    ArrayPtr res;
    rt->rt->issue(make_gather_kernel(std::move(spec)), ins, {}, std::span(&out, 1), site,
                  std::span(&res, 1));
    *result = hand_out(std::move(res));
    return VP_OK;
}

int vp_scatter(vp_runtime* rt, vp_array* dst, int nidx, vp_array* const* idx, const int* idx_axes,
               int nres, const int* result_axes, vp_array* values, const char* site, char* err,
               size_t errlen) {
    rt->graveyard->drain();
    const Layout& v = arr(dst).layout();
    IndexSpec spec;
    Shape shape;
    if (!index_spec(v, nidx, idx, idx_axes, nres, result_axes, spec, shape, err, errlen)) {
        return VP_INVALID;
    }
    const Layout& vl = arr(values).layout();
    Shape b;
    if (vl.dtype != v.dtype || !broadcast_shapes(shape, vl.shape, b) || b != shape) {
        set_error(err, errlen, "scatter: values have the wrong dtype or shape");
        return VP_INVALID;
    }
    std::vector<ArrayPtr> ins{ref(values)};
    for (int j = 0; j < nidx; ++j) {
        ins.push_back(ref(idx[j]));
    }
    const ArrayPtr io[] = {ref(dst)};
    rt->rt->issue(make_scatter_kernel(std::move(spec)), ins, io, {}, site);
    return VP_OK;
}

int vp_select(vp_runtime* rt, vp_array* cond, vp_array* x, vp_array* y, vp_array** result,
              const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    const Layout ls[] = {arr(cond).layout(), arr(x).layout(), arr(y).layout()};
    Shape a, b;
    if (ls[0].dtype != DType::Bool || ls[1].dtype != ls[2].dtype ||
        !broadcast_shapes(ls[0].shape, ls[1].shape, a) || !broadcast_shapes(a, ls[2].shape, b)) {
        set_error(err, errlen, "where: wrong dtypes or shapes");
        return VP_INVALID;
    }
    Layout out = Layout::keep_order(ls[1].dtype, std::move(b), ls);
    const ArrayPtr ins[] = {ref(cond), ref(x), ref(y)};
    ArrayPtr res;
    static const std::shared_ptr<Kernel> kernel = make_select_kernel();
    rt->rt->issue(kernel, ins, {}, std::span(&out, 1), site, std::span(&res, 1));
    *result = hand_out(std::move(res));
    return VP_OK;
}

int vp_mask_assign(vp_runtime* rt, vp_array* dst, vp_array* mask, vp_array* value, const char* site,
                   char* err, size_t errlen) {
    rt->graveyard->drain();
    const Layout& d = arr(dst).layout();
    const Layout& m = arr(mask).layout();
    Shape b;
    if (m.dtype != DType::Bool || arr(value).layout().dtype != d.dtype ||
        arr(value).layout().size() != 1 || !broadcast_shapes(d.shape, m.shape, b) || b != d.shape) {
        set_error(err, errlen, "mask assignment: wrong dtypes or shapes");
        return VP_INVALID;
    }
    const ArrayPtr ins[] = {ref(value), ref(mask)};
    const ArrayPtr io[] = {ref(dst)};
    static const std::shared_ptr<Kernel> kernel = make_mask_assign_kernel();
    rt->rt->issue(kernel, ins, io, {}, site);
    return VP_OK;
}

namespace {
// True if NumPy's gufunc iterator would take `a` as it is (NPY_ITER_ALIGNED):
// the first element and every stride along an axis of extent > 1 are
// multiples of the dtype's alignment.
bool numpy_aligned(const Array& a) {
    const Layout& l = a.layout();
    const std::size_t align = l.dtype == DType::Complex64 || l.dtype == DType::Complex128
                                  ? itemsize(l.dtype) / 2
                                  : itemsize(l.dtype);
    // A deferred buffer has no address yet; its pool block will be aligned
    // (element_pool.cpp: alignment_for), so only the offset counts.
    const std::byte* data = a.buffer()->data();
    std::uintptr_t bits =
        reinterpret_cast<std::uintptr_t>(data) + static_cast<std::uintptr_t>(l.offset);
    for (std::size_t i = 0; i < l.shape.size(); ++i) {
        if (l.shape[i] > 1) {
            bits |= static_cast<std::uintptr_t>(l.strides[i]);
        }
    }
    return bits % align == 0;
}
} // namespace

int vp_matmul(vp_runtime* rt, PyObject* ufunc, vp_array* a, vp_array* b, vp_array** result,
              const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    const Layout& la = arr(a).layout();
    const Layout& lb = arr(b).layout();
    Layout out;
    std::int64_t work = 0;
    if (!numpy::matmul_layout(la, lb, out, work)) {
        set_error(err, errlen, "matmul: dtypes or shapes NumPy does not take as they are");
        return VP_UNSUPPORTED;
    }
    if (!numpy_aligned(arr(a)) || !numpy_aligned(arr(b))) {
        set_error(err, errlen, "matmul: misaligned operand (NumPy copies it)");
        return VP_UNSUPPORTED;
    }
    std::string e;
    const DType d[] = {la.dtype, la.dtype, la.dtype};
    const numpy::UfuncLoop* loop = rt->loops->get(ufunc, d, e);
    if (loop == nullptr) {
        PyErr_Clear();
        set_error(err, errlen, e);
        return VP_UNSUPPORTED;
    }
    std::shared_ptr<Kernel>& kernel = rt->matmul_kernels[loop];
    if (!kernel) {
        kernel = numpy::make_matmul_kernel(loop);
    }
    const ArrayPtr ins[] = {ref(a), ref(b)};
    ArrayPtr res;
    rt->rt->issue(kernel, ins, {}, std::span(&out, 1), site, std::span(&res, 1), work);
    *result = hand_out(std::move(res));
    return VP_OK;
}

int vp_arg_reduce(vp_runtime* rt, int argmax, vp_array* in, int out_ndim, const int64_t* out_shape,
                  vp_array** result, const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    const Layout& l = arr(in).layout();
    Layout out = Layout::contiguous(DType::Int64, Shape(out_shape, out_shape + out_ndim));
    if (l.shape.empty() || !l.c_contiguous() || l.shape.back() == 0 ||
        out.size() * l.shape.back() != l.size() || sizeof(void*) != 8) {
        set_error(err, errlen, "argmin/argmax: unsupported input");
        return VP_UNSUPPORTED;
    }
    std::string e;
    std::shared_ptr<Kernel> k = numpy::make_arg_reduce_kernel(l.dtype, argmax != 0, e);
    if (!k) {
        set_error(err, errlen, e);
        return VP_UNSUPPORTED;
    }
    const ArrayPtr ins[] = {ref(in)};
    ArrayPtr res;
    rt->rt->issue(k, ins, {}, std::span(&out, 1), site, std::span(&res, 1));
    *result = hand_out(std::move(res));
    return VP_OK;
}

int vp_reduce(vp_runtime* rt, PyObject* ufunc, vp_array* in, const unsigned char* axes,
              int keepdims, PyObject* dtype, PyObject* identity, vp_array** result,
              const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    const Layout& l = arr(in).layout();
    if (l.size() == 0) {
        set_error(err, errlen, "empty reduction"); // NumPy's identity / error handling
        return VP_UNSUPPORTED;
    }
    const std::size_t nd = l.shape.size();
    std::vector<char> flags(nd); // std::vector<bool> has no data()
    Shape out_shape;
    for (std::size_t i = 0; i < nd; ++i) {
        flags[i] = axes[i] != 0;
        if (!flags[i]) {
            out_shape.push_back(l.shape[i]);
        } else if (keepdims) {
            out_shape.push_back(1);
        }
    }
    Layout out_layout = Layout::contiguous(l.dtype, std::move(out_shape));
    const std::span<const bool> reduced(reinterpret_cast<const bool*>(flags.data()), nd);
    if (dtype == nullptr) {
        dtype = Py_None;
    }
    if (identity == nullptr) {
        identity = Py_None;
    }
    numpy::ReducePlan plan = numpy::plan_reduction(l, reduced, out_layout);
    std::string e;
    // Fast path: no cast, a dense input (in any axis order) whose reduced axes
    // coalesce: the loop calls follow from the plan. Everything else, NumPy's
    // iterator drives: it may buffer strided inputs even without a cast, and
    // a loop called on a buffer can round differently than on strided data
    // (complex multiply has a SIMD path for contiguous operands only).
    Layout dense_check = l;
    dense_check.offset = 0;
    const numpy::UfuncLoop* loop =
        dtype == Py_None && plan.supported() && dense_check.dense()
            ? rt->loops->get_reduction(ufunc, l.dtype, plan.inner_strides(), e)
            : nullptr;
    if (loop == nullptr) {
        PyErr_Clear();
        Layout rl;
        std::shared_ptr<Kernel> k = rt->loops->iter_reduction(
            ufunc, l, reduced, keepdims != 0, dtype, identity, defer_release(rt->graveyard), rl, e);
        if (!k) {
            PyErr_Clear();
            set_error(err, errlen, e);
            return VP_UNSUPPORTED;
        }
        const ArrayPtr ins[] = {ref(in)};
        ArrayPtr res;
        rt->rt->issue(k, ins, {}, std::span(&rl, 1), site, std::span(&res, 1), l.size());
        *result = hand_out(std::move(res));
        return VP_OK;
    }
    std::vector<std::byte> init;
    if (identity != nullptr && identity != Py_None &&
        !numpy::scalar_bytes(identity, l.dtype, init)) {
        return VP_PYERR;
    }
    const ArrayPtr ins[] = {ref(in)};
    ArrayPtr res;
    rt->rt->issue(numpy::make_reduce_kernel(loop, std::move(plan), std::move(init)), ins, {},
                  std::span(&out_layout, 1), site, std::span(&res, 1), l.size());
    *result = hand_out(std::move(res));
    return VP_OK;
}

int vp_assign(vp_runtime* rt, vp_array* dst, vp_array* src, const char* site, char* err,
              size_t errlen) {
    rt->graveyard->drain();
    const Layout& d = arr(dst).layout();
    const Layout& s = arr(src).layout();
    Shape b;
    if (!broadcast_shapes(d.shape, s.shape, b) || b != d.shape) {
        set_error(err, errlen, "assign: shapes do not broadcast");
        return VP_INVALID;
    }
    const ArrayPtr ins[] = {ref(src)};
    const ArrayPtr io[] = {ref(dst)};
    if (d.dtype != s.dtype) {
        // NumPy casts the values in its assignment (unsafe casting): the cast
        // kernel writes them straight into dst.
        if (d.size() == 0) {
            return VP_OK;
        }
        std::string e;
        std::shared_ptr<Kernel> k = numpy::make_cast_kernel(
            broadcast_to(s, d.shape), d, rt->loops->buffersize(), defer_release(rt->graveyard), e);
        if (!k) {
            PyErr_Clear();
            set_error(err, errlen, e);
            return VP_UNSUPPORTED;
        }
        rt->rt->issue(k, ins, io, {}, site);
        return VP_OK;
    }
    static const std::shared_ptr<Kernel> assign = make_assign_kernel();
    rt->rt->issue(assign, ins, io, {}, site);
    return VP_OK;
}

} // extern "C"
