// Implementation of the public C ABI (include/viproc/viproc.h).

#include "viproc/viproc.h"

#include "kernels/numpy_gufunc.hpp"
#include "kernels/numpy_matmul.hpp"
#include "kernels/numpy_reduce.hpp"
#include "kernels/numpy_sort.hpp"
#include "kernels/numpy_ufunc.hpp"
#include "runtime/assign.hpp"
#include "runtime/cpu.hpp"
#include "runtime/element_pool.hpp"
#include "runtime/index.hpp"
#include "runtime/runtime.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <tuple>
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

// "<message> (in <op>, issued at <site>)": the message keeps a leading
// "TypeName: " for the bridge to raise that exception type.
std::string describe(const TaskError& e) {
    return e.message + " (in " + (e.op.empty() ? std::string("array op") : e.op) + ", issued at " +
           e.issue_site + ")";
}

DType to_dtype(vp_dtype d) { return static_cast<DType>(d); }
vp_dtype to_vp(DType d) { return static_cast<vp_dtype>(d); }

} // namespace

struct vp_loop {
    const numpy::UfuncLoop* loop; // owned by vp_runtime::loops
    std::vector<DType> dtypes;    // inputs, then the output
};

namespace {

// A reduction's identity as a cache key: exact Python bools, ints (within
// int64) and floats only; their value alone gives the bytes NumPy converts
// them to. Other objects (0-d arrays as `initial=`, which may be written
// later, NumPy scalars, big ints) are converted per op.
struct IdentityKey {
    int kind = 0; // 0 none, 1 bool, 2 int, 3 float
    std::uint64_t bits = 0;
    bool operator==(const IdentityKey&) const = default;
};

bool identity_key(PyObject* v, IdentityKey& k) {
    if (v == Py_None) {
        k = {0, 0};
    } else if (PyBool_Check(v)) {
        k = {1, v == Py_True ? 1u : 0u};
    } else if (PyLong_CheckExact(v)) {
        int overflow = 0;
        const long long x = PyLong_AsLongLongAndOverflow(v, &overflow);
        if (overflow != 0 || (x == -1 && PyErr_Occurred())) {
            PyErr_Clear();
            return false;
        }
        k = {2, static_cast<std::uint64_t>(x)};
    } else if (PyFloat_CheckExact(v)) {
        const double d = PyFloat_AS_DOUBLE(v);
        k.kind = 3;
        std::memcpy(&k.bits, &d, sizeof d); // -0.0 and NaN payloads differ
    } else {
        return false;
    }
    return true;
}

// The reduce kernel of a fast-path reduction (ReducePlan, no cast) depends
// on the ufunc, the input layout without its offset, the reduced axes and
// the identity only: one kernel per such key, shared by its ops, as the
// element-wise kernel per loop. A loop over columns (`np.sum(a[:, i],
// axis=0)`) planned, converted the identity (an ndarray per op) and built a
// kernel for every op: ~2,700 instructions.
struct ReduceKey {
    PyObject* ufunc; // alive: LoopCache holds a reference to every ufunc
    Layout layout;   // offset 0
    std::uint64_t axes;
    IdentityKey identity;
    bool operator==(const ReduceKey& o) const {
        return ufunc == o.ufunc && axes == o.axes && identity == o.identity &&
               layout.dtype == o.layout.dtype && layout.shape == o.layout.shape &&
               layout.strides == o.layout.strides;
    }
};

struct ReduceKeyHash {
    std::size_t operator()(const ReduceKey& k) const {
        std::size_t h = std::hash<const void*>()(k.ufunc);
        auto mix = [&h](std::uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
        mix(static_cast<std::uint64_t>(k.layout.dtype));
        for (std::size_t i = 0; i < k.layout.shape.size(); ++i) {
            mix(static_cast<std::uint64_t>(k.layout.shape[i]));
            mix(static_cast<std::uint64_t>(k.layout.strides[i]));
        }
        mix(k.axes);
        mix(static_cast<std::uint64_t>(k.identity.kind));
        mix(k.identity.bits);
        return h;
    }
};

} // namespace

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
    // One sort kernel per (dtype, kind, arg, axis).
    std::map<std::tuple<DType, int, bool, int>, std::shared_ptr<Kernel>> sort_kernels;
    // One gufunc kernel per (loop, core axes, invalid_error, options).
    std::map<std::tuple<const numpy::UfuncLoop*, std::vector<std::vector<int>>, std::string, int>,
             std::shared_ptr<Kernel>>
        gufunc_kernels;
    // Fast-path reduce kernels (ReduceKey), cleared when it grows large.
    std::unordered_map<ReduceKey, std::shared_ptr<Kernel>, ReduceKeyHash> reduce_kernels;
};

// A vp_array* is an Array* holding one reference (no wrapper allocation).
namespace {
Array& arr(const vp_array* a) { return *reinterpret_cast<Array*>(const_cast<vp_array*>(a)); }
ArrayPtr ref(const vp_array* a) { return ArrayPtr(&arr(a)); } // a new reference
vp_array* hand_out(ArrayPtr a) { return reinterpret_cast<vp_array*>(a.release()); }
} // namespace

extern "C" {

namespace {

// RuntimeOptions::block_enter / block_leave: the GIL is released while the
// main thread blocks inside an issue (a task may need it).
thread_local PyThreadState* t_blocked = nullptr;

void release_gil() {
    if (PyGILState_Check()) {
        t_blocked = PyEval_SaveThread();
    }
}

void reacquire_gil() {
    if (t_blocked != nullptr) {
        PyEval_RestoreThread(t_blocked);
        t_blocked = nullptr;
    }
}

} // namespace

vp_runtime* vp_runtime_create(size_t workers, size_t max_active_tasks, char* err, size_t errlen) {
    if (!numpy::init()) {
        PyErr_Clear();
        set_error(err, errlen, "cannot import the NumPy C API");
        return nullptr;
    }
    RuntimeOptions options;
    options.workers = workers;
    options.block_enter = release_gil;
    options.block_leave = reacquire_gil;
    if (max_active_tasks != 0) {
        options.max_active_tasks = max_active_tasks;
    }
    auto* r = new vp_runtime();
    r->rt = std::make_unique<Runtime>(options);
    r->loops = std::make_unique<numpy::LoopCache>();
    numpy::set_kernel_threads(r->rt->worker_count() + 1);
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

void vp_set_error_notify(vp_runtime* rt, void (*fn)(void*), void* arg) {
    rt->rt->errors().set_notify(fn, arg);
}

int vp_error_pending(const vp_runtime* rt) { return rt->rt->errors().pending() ? 1 : 0; }

int vp_take_error(vp_runtime* rt, char* err, size_t errlen) {
    std::size_t more = 0;
    std::optional<TaskError> e = rt->rt->errors().take(&more);
    if (!e) {
        return VP_OK;
    }
    std::string msg = describe(*e);
    if (more > 0) {
        msg += " [" + std::to_string(more) + " more failed op" + (more > 1 ? "s" : "") +
               " not reported]";
    }
    set_error(err, errlen, msg);
    return VP_ERROR;
}

void vp_set_fp_mode(vp_runtime* rt, int mode, int ctx) {
    rt->rt->set_fp_mode(FpMode{static_cast<std::uint16_t>(mode), static_cast<std::uint16_t>(ctx)});
}

int vp_take_fp_events(vp_runtime* rt, PyObject** out) {
    std::vector<FpEvent> events = rt->rt->errors().take_fp();
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(events.size()));
    if (list == nullptr) {
        return -1;
    }
    for (std::size_t i = 0; i < events.size(); ++i) {
        const FpEvent& e = events[i];
        PyObject* t = Py_BuildValue("(iiiiss)", e.report, e.status, static_cast<int>(e.mode.mode),
                                    static_cast<int>(e.mode.ctx), e.name, e.issue_site.c_str());
        if (t == nullptr) {
            Py_DECREF(list);
            return -1;
        }
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), t);
    }
    *out = list;
    return static_cast<int>(events.size());
}

size_t vp_worker_count(const vp_runtime* rt) { return rt->rt->worker_count(); }

size_t vp_physical_core_count(void) { return physical_core_count(); }

double vp_direct_budget_ns(const vp_runtime* rt) {
    return rt->rt->offload_policy().options().direct_budget_ns;
}

void vp_set_direct_budget_ns(vp_runtime* rt, double ns) {
    OffloadOptions o = rt->rt->offload_policy().options();
    o.direct_budget_ns = ns;
    rt->rt->set_offload_options(o);
}
void vp_set_chain_ns(vp_runtime* rt, double ns) {
    OffloadOptions o = rt->rt->offload_policy().options();
    o.chain_ns = ns;
    rt->rt->set_offload_options(o);
}
size_t vp_active_tasks(const vp_runtime* rt) { return rt->rt->active_tasks(); }

void vp_set_offload_policy(vp_runtime* rt, int64_t sync_below, int adaptive) {
    OffloadOptions o = rt->rt->offload_policy().options();
    o.sync_below = sync_below;
    o.adaptive = adaptive != 0;
    rt->rt->set_offload_options(o);
}

uint64_t vp_direct_count(const vp_runtime* rt) { return rt->rt->direct_count(); }

uint64_t vp_task_count(const vp_runtime* rt) { return rt->rt->task_count(); }

uint64_t vp_sync_count(const vp_runtime* rt) { return rt->rt->sync_count(); }

uint64_t vp_wake_count(const vp_runtime* rt) { return rt->rt->wake_count(); }

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

int vp_array_from_item(vp_runtime* rt, PyObject* value, vp_dtype dtype, vp_array** out) {
    ArrayPtr a = numpy::item_array(*rt->rt, value, to_dtype(dtype));
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
int64_t vp_array_offset(const vp_array* a) { return arr(a).layout().offset; }
int vp_array_ready(vp_array* a) { return arr(a).ready() ? 1 : 0; }
int vp_array_same_storage(const vp_array* a, const vp_array* b) {
    return arr(a).group() == arr(b).group() ? 1 : 0;
}

size_t vp_array_reusable_bytes(const vp_array* a) {
    const Array& x = arr(a);
    const CountableArray& b = *x.buffer();
    const Layout& l = x.layout();
    const bool ok = x.ref_count() == 1 && x.group().use_count() == 1 && b.capacity() != 0 &&
                    !b.has_readers() && l.offset == 0 && l.c_contiguous() &&
                    l.nbytes() == b.nbytes();
    return ok ? b.nbytes() : 0;
}

int vp_array_wait(vp_runtime* rt, vp_array* a, char* err, size_t errlen) {
    ErrorRef e = rt->rt->wait(arr(a));
    if (e) {
        set_error(err, errlen, describe(rt->rt->errors().raise_at_sync_point(e)));
        return VP_ERROR;
    }
    return VP_OK;
}

PyObject* vp_array_export(const vp_array* a, int writable) {
    return numpy::export_ndarray(arr(a), writable != 0);
}

int vp_array_host_writable(vp_array* a) {
    Array& x = arr(a);
    return x.ready() && !x.group()->last_error() && !x.buffer()->has_readers() ? 1 : 0;
}

PyObject* vp_array_borrow(const vp_array* a) { return numpy::borrow_ndarray(arr(a)); }

int vp_array_pin(const vp_array* a, PyObject* view) {
    return numpy::pin_borrowed(arr(a), view) ? VP_OK : VP_PYERR;
}

int vp_array_prepare_write(vp_runtime* rt, vp_array* a, char* err, size_t errlen) {
    ErrorRef e = rt->rt->prepare_host_write(arr(a));
    if (e) {
        set_error(err, errlen, describe(rt->rt->errors().raise_at_sync_point(e)));
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
    // an out= array of another dtype: NumPy casts into it in its iterator's
    // buffers (check_for_trivial_loop: no trivial loop)
    cast = cast || out.dtype != loop.dtypes[ins.size()];
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
        loop.loop, loop.dtypes, ls, std::span(&out, 1), out_given, defer_release(rt->graveyard), e);
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
        if (ol.shape != shape) {
            set_error(err, errlen, "out= array has the wrong shape");
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

int vp_ufunc_issue_masked(vp_runtime* rt, const vp_loop* loop, vp_array* const* inputs,
                          vp_array* mask, int out_ndim, const int64_t* out_shape, vp_array* out,
                          vp_array** result, const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    const int nin = loop->loop->nin();
    if (nin > 3 || loop->loop->nout() != 1 || arr(mask).layout().dtype != DType::Bool) {
        set_error(err, errlen, "unsupported masked ufunc");
        return VP_UNSUPPORTED;
    }
    // no copies of small inputs: NumPy's masked execution skips
    // check_for_trivial_loop
    ArrayPtr ins[4];
    Layout layouts[4];
    for (int i = 0; i < nin; ++i) {
        ins[i] = ref(inputs[i]);
        layouts[i] = ins[i]->layout();
    }
    ins[nin] = ref(mask);
    layouts[nin] = ins[nin]->layout();
    const auto ni = static_cast<std::size_t>(nin);
    const DType out_dtype = loop->dtypes[ni];
    const Shape shape(out_shape, out_shape + out_ndim);
    Layout ol;
    if (out != nullptr) {
        ol = arr(out).layout();
        if (ol.shape != shape) {
            set_error(err, errlen, "out= array has the wrong shape");
            return VP_INVALID;
        }
    } else {
        // the iterator allocates the output in order K over every operand,
        // the mask included
        bool all_c = true;
        for (std::size_t i = 0; i <= ni; ++i) {
            all_c = all_c && (layouts[i].c_contiguous() || layouts[i].size() <= 1);
        }
        ol = all_c ? Layout::contiguous(out_dtype, shape)
                   : Layout::keep_order(out_dtype, shape, std::span(layouts, ni + 1));
    }
    std::string e;
    std::shared_ptr<Kernel> k = rt->loops->iter_elementwise(
        loop->loop, loop->dtypes, std::span(layouts, ni), std::span(&ol, 1), out != nullptr,
        defer_release(rt->graveyard), e, &layouts[ni]);
    if (!k) {
        PyErr_Clear();
        set_error(err, errlen, e);
        return VP_UNSUPPORTED;
    }
    const std::span<const ArrayPtr> in_span(ins, ni + 1);
    if (out != nullptr) {
        const ArrayPtr o = ref(out);
        rt->rt->issue(k, in_span, std::span(&o, 1), {}, site);
        *result = nullptr;
        return VP_OK;
    }
    ArrayPtr res;
    rt->rt->issue(k, in_span, {}, std::span(&ol, 1), site, std::span(&res, 1));
    *result = hand_out(std::move(res));
    return VP_OK;
}

int vp_ufunc_multi(vp_runtime* rt, PyObject* ufunc, int nin, vp_array* const* inputs, int nout,
                   const vp_dtype* dtypes, int out_ndim, const int64_t* out_shape,
                   vp_array** results, const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    constexpr int kMax = 8;
    if (nin < 1 || nout < 1 || nin + nout > kMax) {
        set_error(err, errlen, "unsupported number of operands");
        return VP_UNSUPPORTED;
    }
    std::vector<DType> d;
    for (int i = 0; i < nin + nout; ++i) {
        d.push_back(to_dtype(dtypes[i]));
    }
    std::string e;
    const numpy::UfuncLoop* loop = rt->loops->get(ufunc, d, e);
    if (loop == nullptr || loop->nin() != nin || loop->nout() != nout) {
        PyErr_Clear();
        set_error(err, errlen, loop == nullptr ? e : "wrong number of operands");
        return VP_UNSUPPORTED;
    }
    ArrayPtr ins[kMax];
    for (int i = 0; i < nin; ++i) {
        ins[i] = ref(inputs[i]);
    }
    // check_for_trivial_loop runs for every ufunc: small inputs of another
    // dtype are cast into copies first here too
    const vp_loop handle{loop, d};
    if (!cast_small_inputs(rt, handle, std::span(ins, static_cast<std::size_t>(nin)), site, err,
                           errlen)) {
        return VP_UNSUPPORTED;
    }
    Layout in_layouts[kMax];
    bool all_c = true;
    for (int i = 0; i < nin; ++i) {
        in_layouts[i] = ins[i]->layout();
        all_c = all_c && (in_layouts[i].c_contiguous() || in_layouts[i].size() <= 1);
    }
    const auto ni = static_cast<std::size_t>(nin);
    const auto no = static_cast<std::size_t>(nout);
    // NumPy runs several outputs on its iterator (no trivial loop), which
    // allocates them in order K over the inputs (C if all inputs are)
    Layout outs[kMax];
    const Shape shape(out_shape, out_shape + out_ndim);
    for (std::size_t j = 0; j < no; ++j) {
        outs[j] = all_c ? Layout::contiguous(d[ni + j], shape)
                        : Layout::keep_order(d[ni + j], shape, std::span(in_layouts, ni));
    }
    std::shared_ptr<Kernel> k =
        rt->loops->iter_elementwise(loop, d, std::span(in_layouts, ni), std::span(outs, no), false,
                                    defer_release(rt->graveyard), e);
    if (!k) {
        PyErr_Clear();
        set_error(err, errlen, e);
        return VP_UNSUPPORTED;
    }
    ArrayPtr res[kMax];
    rt->rt->issue(k, std::span(ins, ni), {}, std::span(outs, no), site, std::span(res, no));
    for (std::size_t j = 0; j < no; ++j) {
        results[j] = hand_out(std::move(res[j]));
    }
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
              int nres, const int* result_axes, const int64_t* result_strides, int mode,
              vp_array** result, const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    const Layout& v = arr(src).layout();
    IndexSpec spec;
    Shape shape;
    if (!index_spec(v, nidx, idx, idx_axes, nres, result_axes, spec, shape, err, errlen)) {
        return VP_INVALID;
    }
    if (mode < VP_INDEX_RAISE || mode > VP_INDEX_CLIP) {
        set_error(err, errlen, "indexing: unknown index mode");
        return VP_INVALID;
    }
    spec.mode = static_cast<IndexMode>(mode);
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
               int nres, const int* result_axes, vp_array* values, int mode, const char* site,
               char* err, size_t errlen) {
    rt->graveyard->drain();
    const Layout& v = arr(dst).layout();
    IndexSpec spec;
    Shape shape;
    if (!index_spec(v, nidx, idx, idx_axes, nres, result_axes, spec, shape, err, errlen)) {
        return VP_INVALID;
    }
    if (mode < VP_INDEX_RAISE || mode > VP_INDEX_CLIP) {
        set_error(err, errlen, "indexing: unknown index mode");
        return VP_INVALID;
    }
    spec.mode = static_cast<IndexMode>(mode);
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

int vp_place(vp_runtime* rt, vp_dtype dtype, int ndim, const int64_t* shape, const int64_t* strides,
             int nparts, vp_array* const* srcs, const int* part_ndim, const int64_t* part_shapes,
             const int64_t* part_strides, const int64_t* part_offsets, vp_array** result,
             const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    Shape dims;
    dims.assign(shape, shape + ndim);
    Layout out = Layout::contiguous(to_dtype(dtype), std::move(dims));
    if (strides != nullptr) {
        out.strides.assign(strides, strides + ndim);
        if (!out.dense() && out.size() != 0) {
            set_error(err, errlen, "place: result strides are not dense");
            return VP_INVALID;
        }
    }
    const std::int64_t item = static_cast<std::int64_t>(itemsize(out.dtype));
    const std::int64_t nbytes = out.size() * item;
    std::vector<Part> parts(static_cast<std::size_t>(nparts));
    std::vector<ArrayPtr> ins;
    ins.reserve(parts.size());
    for (int i = 0; i < nparts; ++i) {
        Part& p = parts[i];
        p.shape.assign(part_shapes, part_shapes + part_ndim[i]);
        p.strides.assign(part_strides, part_strides + part_ndim[i]);
        p.offset = part_offsets[i];
        part_shapes += part_ndim[i];
        part_strides += part_ndim[i];
        // The part must lie within the result's elements.
        std::int64_t lo = p.offset, hi = p.offset, size = 1;
        for (int d = 0; d < part_ndim[i]; ++d) {
            size *= p.shape[d];
            const std::int64_t span = (p.shape[d] - 1) * p.strides[d];
            (span < 0 ? lo : hi) += span;
        }
        const Layout& s = arr(srcs[i]).layout();
        Shape b;
        if (s.dtype != out.dtype || !broadcast_shapes(p.shape, s.shape, b) || b != p.shape ||
            (size != 0 && (lo < 0 || hi + item > nbytes))) {
            set_error(err, errlen, "place: a part has the wrong dtype, shape or position");
            return VP_INVALID;
        }
        ins.push_back(ref(srcs[i]));
    }
    ArrayPtr res;
    rt->rt->issue(make_place_kernel(std::move(parts)), ins, {}, std::span(&out, 1), site,
                  std::span(&res, 1));
    *result = hand_out(std::move(res));
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

int vp_gufunc(vp_runtime* rt, PyObject* ufunc, int nin, vp_array* const* inputs, int nout,
              vp_array* const* outs, const vp_dtype* dtypes, const int* core_n,
              const int* core_axes, int nlabels, int64_t* sizes, int flags, int order_n,
              const int* out_order, const char* invalid_error, int64_t work, vp_array** results,
              const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    constexpr int kMax = 8;
    if (nin < 1 || nout < 1 || nin + nout > kMax || nlabels < 0) {
        set_error(err, errlen, "unsupported number of operands");
        return VP_UNSUPPORTED;
    }
    if (order_n > 0) { // a permutation of the outputs' axes
        std::vector<bool> seen(static_cast<std::size_t>(order_n), false);
        for (int k = 0; k < order_n; ++k) {
            if (out_order[k] < 0 || out_order[k] >= order_n ||
                seen[static_cast<std::size_t>(out_order[k])]) {
                set_error(err, errlen, "gufunc: out_order is no permutation");
                return VP_UNSUPPORTED;
            }
            seen[static_cast<std::size_t>(out_order[k])] = true;
        }
    }
    const auto ni = static_cast<std::size_t>(nin);
    const auto no = static_cast<std::size_t>(nout);
    const std::size_t nop = ni + no;
    std::vector<DType> d;
    for (std::size_t i = 0; i < nop; ++i) {
        d.push_back(to_dtype(dtypes[i]));
    }
    std::string e;
    const numpy::UfuncLoop* loop = rt->loops->get(ufunc, d, e);
    if (loop == nullptr || loop->nin() != nin || loop->nout() != nout) {
        PyErr_Clear();
        set_error(err, errlen, loop == nullptr ? e : "wrong number of operands");
        return VP_UNSUPPORTED;
    }
    numpy::GufuncCore core;
    core.nlabels = nlabels;
    for (std::size_t i = 0, k = 0; i < nop; ++i) {
        core.axes.emplace_back(core_axes + k, core_axes + k + core_n[i]);
        k += static_cast<std::size_t>(core_n[i]);
    }
    const bool given = outs != nullptr && outs[0] != nullptr;
    const Layout* ls[kMax];
    for (std::size_t i = 0; i < nop; ++i) {
        if (i >= ni && !given) {
            ls[i] = nullptr;
            continue;
        }
        const Array& a = arr(i < ni ? inputs[i] : outs[i - ni]);
        ls[i] = &a.layout();
        if (!numpy_aligned(a)) {
            set_error(err, errlen, "gufunc: misaligned operand (NumPy copies it)");
            return VP_UNSUPPORTED;
        }
        if (i >= ni && ls[i]->dtype != d[i]) {
            set_error(err, errlen, "gufunc: out= of another dtype");
            return VP_UNSUPPORTED;
        }
        for (std::size_t j = 0; i >= ni && j < ni; ++j) {
            if (arr(inputs[j]).group() == a.group()) {
                set_error(err, errlen, "gufunc: out= overlaps an input (NumPy copies it)");
                return VP_UNSUPPORTED;
            }
        }
    }
    // Core dimension sizes (no broadcasting of core dimensions) and the
    // broadcast batch shape.
    Shape batch;
    for (std::size_t i = 0; i < nop; ++i) {
        if (ls[i] == nullptr) {
            continue;
        }
        const Layout& l = *ls[i];
        const std::size_t nc = core.present(i);
        if (l.shape.size() < nc) {
            set_error(err, errlen, "gufunc: operand has too few dimensions");
            return VP_UNSUPPORTED;
        }
        std::size_t ax = l.shape.size() - nc;
        for (int label : core.axes[i]) {
            if (label < 0) {
                continue;
            }
            int64_t& size = sizes[label];
            if (size >= 0 && size != l.shape[ax]) {
                set_error(err, errlen, "gufunc: core dimension mismatch");
                return VP_UNSUPPORTED;
            }
            size = l.shape[ax++];
        }
        Shape b;
        if (!broadcast_shapes(batch, Shape(l.shape.begin(), l.shape.end() - nc), b)) {
            set_error(err, errlen, "gufunc: batch shapes do not broadcast");
            return VP_UNSUPPORTED;
        }
        batch = std::move(b);
    }
    for (int i = 0; i < nlabels; ++i) {
        if (sizes[i] < 0) {
            set_error(err, errlen, "gufunc: output core dimension of unknown size");
            return VP_UNSUPPORTED;
        }
    }
    const std::size_t nb = batch.size();
    // The iteration space of NumPy's iterator: the batch axes, then each
    // output's core axes. Inputs have stride 0 along all output core axes;
    // given outputs along the other outputs' ones.
    Shape iter_shape = batch;
    std::vector<std::size_t> out_first(no);
    for (std::size_t j = 0; j < no; ++j) {
        out_first[j] = iter_shape.size();
        for (int label : core.axes[ni + j]) {
            if (label >= 0) {
                iter_shape.push_back(sizes[label]);
            }
        }
    }
    const std::size_t nit = iter_shape.size();
    std::vector<Layout> iter_ops;
    for (std::size_t i = 0; i < nop; ++i) {
        if (ls[i] == nullptr) {
            continue;
        }
        const Layout& l = *ls[i];
        Layout it;
        it.dtype = l.dtype;
        it.shape = Shape(nit, 1);
        it.strides = Strides(nit, 0);
        const std::size_t lb = l.shape.size() - core.present(i);
        for (std::size_t a = 0; a < lb; ++a) {
            it.shape[nb - lb + a] = l.shape[a];
            it.strides[nb - lb + a] = l.strides[a];
        }
        if (i >= ni) {
            for (std::size_t a = lb; a < l.shape.size(); ++a) {
                it.shape[out_first[i - ni] + a - lb] = l.shape[a];
                it.strides[out_first[i - ni] + a - lb] = l.strides[a];
            }
        }
        if (i >= ni && Shape(l.shape.begin(), l.shape.begin() + lb) != batch) {
            set_error(err, errlen, "gufunc: out= does not have the batch shape");
            return VP_UNSUPPORTED;
        }
        iter_ops.push_back(std::move(it));
    }
    const std::vector<std::size_t> perm = Layout::iteration_order(iter_shape, iter_ops);
    // Inputs of another dtype: NumPy's iterator casts them into copies
    // (npyiter_new_temp_array: batch axes in the iteration order, core
    // axes C-contiguous innermost).
    ArrayPtr ins[kMax];
    for (std::size_t i = 0; i < ni; ++i) {
        ins[i] = ref(inputs[i]);
        const Layout& l = *ls[i];
        if (l.dtype == d[i]) {
            continue;
        }
        const std::size_t nd = l.shape.size();
        const std::size_t lb = nd - core.present(i);
        Layout dst;
        dst.dtype = d[i];
        dst.shape = l.shape;
        dst.strides = Strides(nd, 0);
        std::vector<bool> set(nd, false);
        std::int64_t step = static_cast<std::int64_t>(itemsize(d[i]));
        for (std::size_t ax : perm) {
            if (ax < nb && ax >= nb - lb) {
                const std::size_t a = ax - (nb - lb);
                dst.strides[a] = step;
                set[a] = true;
                step *= l.shape[a];
            }
        }
        std::int64_t factor = 1;
        for (std::size_t a = nd; a-- > 0;) {
            if (!set[a]) {
                dst.strides[a] = factor * static_cast<std::int64_t>(itemsize(d[i]));
                factor *= l.shape[a];
            }
        }
        for (std::size_t a = 0; a < nd; ++a) {
            if (set[a]) {
                dst.strides[a] *= factor;
            }
        }
        if (l.size() == 0) {
            set_error(err, errlen, "gufunc: empty cast");
            return VP_UNSUPPORTED;
        }
        std::shared_ptr<Kernel> k = numpy::make_cast_kernel(l, dst, rt->loops->buffersize(),
                                                            defer_release(rt->graveyard), e);
        if (!k) {
            PyErr_Clear();
            set_error(err, errlen, e);
            return VP_UNSUPPORTED;
        }
        ArrayPtr res;
        const ArrayPtr in[] = {ins[i]};
        rt->rt->issue(k, in, {}, std::span(&dst, 1), site, std::span(&res, 1));
        ins[i] = std::move(res);
    }
    const int options = flags & (VP_GUFUNC_CHECK_FINITE | VP_GUFUNC_SERIAL);
    static_assert(int{VP_GUFUNC_CHECK_FINITE} == int{numpy::kGufuncCheckFinite} &&
                  int{VP_GUFUNC_SERIAL} == int{numpy::kGufuncSerial});
    std::shared_ptr<Kernel>& kernel = rt->gufunc_kernels[{
        loop, core.axes, invalid_error != nullptr ? invalid_error : "", options}];
    if (!kernel) {
        kernel = numpy::make_gufunc_kernel(loop, core, invalid_error ? invalid_error : "", options);
    }
    if (given) {
        ArrayPtr io[kMax];
        for (std::size_t j = 0; j < no; ++j) {
            io[j] = ref(outs[j]);
            results[j] = nullptr;
        }
        rt->rt->issue(kernel, std::span(ins, ni), std::span(io, no), {}, site, {}, work);
        return VP_OK;
    }
    // New outputs: NumPy's iterator allocates each over the batch axes and
    // its own core axes in the iteration order (or in out_order).
    Layout ol[kMax];
    for (std::size_t j = 0; j < no; ++j) {
        Shape shape = batch;
        const std::size_t first = out_first[j];
        const std::size_t nc = core.present(ni + j);
        for (std::size_t a = 0; a < nc; ++a) {
            shape.push_back(iter_shape[first + a]);
        }
        if (order_n > 0) {
            if (static_cast<std::size_t>(order_n) != shape.size()) {
                set_error(err, errlen, "gufunc: out_order does not match the outputs");
                return VP_UNSUPPORTED;
            }
            Layout l;
            l.dtype = d[ni + j];
            l.strides = Strides(shape.size(), 0);
            std::int64_t step = static_cast<std::int64_t>(itemsize(l.dtype));
            for (int k = order_n - 1; k >= 0; --k) {
                const auto a = static_cast<std::size_t>(out_order[k]);
                l.strides[a] = step;
                step *= shape[a];
            }
            l.shape = std::move(shape);
            ol[j] = std::move(l);
            continue;
        }
        Layout l;
        l.dtype = d[ni + j];
        l.strides = Strides(shape.size(), 0);
        std::int64_t step = static_cast<std::int64_t>(itemsize(l.dtype));
        for (std::size_t ax : perm) {
            std::size_t a;
            if (ax < nb) {
                a = ax;
            } else if (ax >= first && ax < first + nc) {
                a = nb + ax - first;
            } else {
                continue;
            }
            l.strides[a] = step;
            step *= shape[a];
        }
        l.shape = std::move(shape);
        ol[j] = std::move(l);
    }
    ArrayPtr res[kMax];
    rt->rt->issue(kernel, std::span(ins, ni), {}, std::span(ol, no), site, std::span(res, no),
                  work);
    for (std::size_t j = 0; j < no; ++j) {
        results[j] = hand_out(std::move(res[j]));
    }
    return VP_OK;
}

int vp_sort(vp_runtime* rt, vp_array* in, int axis, int kind, int arg, int in_place, int order_n,
            const int* out_order, vp_array** result, const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    *result = nullptr;
    const Layout& l = arr(in).layout();
    const auto nd = static_cast<int>(l.shape.size());
    if (axis < 0 || axis >= nd || (in_place != 0 && arg != 0) ||
        (in_place == 0 && arg == 0 && order_n != nd && order_n != 0)) {
        set_error(err, errlen, "sort: unsupported arguments");
        return VP_UNSUPPORTED;
    }
    std::shared_ptr<Kernel>& k = rt->sort_kernels[{l.dtype, kind, arg != 0, axis}];
    if (!k) {
        std::string e;
        k = numpy::make_sort_kernel(l.dtype, kind, arg != 0, axis, e);
        if (!k) {
            set_error(err, errlen, e);
            return VP_UNSUPPORTED;
        }
    }
    if (in_place != 0) {
        const ArrayPtr io[] = {ref(in)};
        rt->rt->issue(k, {}, io, {}, site, {});
        return VP_OK;
    }
    Layout out;
    if (arg != 0) {
        out = Layout::contiguous(DType::Int64, l.shape); // NumPy's intp result, C order
    } else {
        out.dtype = l.dtype;
        out.shape = l.shape;
        out.strides = Strides(l.shape.size(), 0);
        std::vector<int> k_order;
        if (order_n == 0) { // empty_like(a), order K (PyArray_NewLikeArray)
            k_order.resize(l.shape.size());
            for (int i = 0; i < nd; ++i) {
                k_order[static_cast<std::size_t>(i)] = l.c_contiguous() ? i : nd - 1 - i;
            }
            if (!l.c_contiguous() && !l.f_contiguous()) {
                std::iota(k_order.begin(), k_order.end(), 0); // PyArray_CreateSortedStridePerm
                std::ranges::stable_sort(k_order, [&l](int x, int y) {
                    return std::abs(l.strides[static_cast<std::size_t>(x)]) >
                           std::abs(l.strides[static_cast<std::size_t>(y)]);
                });
            }
            order_n = nd;
            out_order = k_order.data();
        }
        std::int64_t step = static_cast<std::int64_t>(itemsize(l.dtype));
        std::vector<bool> seen(l.shape.size(), false);
        for (int i = order_n - 1; i >= 0; --i) {
            const int a = out_order[i];
            if (a < 0 || a >= nd || seen[static_cast<std::size_t>(a)]) {
                set_error(err, errlen, "sort: out_order is no permutation");
                return VP_UNSUPPORTED;
            }
            seen[static_cast<std::size_t>(a)] = true;
            out.strides[static_cast<std::size_t>(a)] = step;
            step *= l.shape[static_cast<std::size_t>(a)];
        }
    }
    const ArrayPtr ins[] = {ref(in)};
    ArrayPtr res;
    rt->rt->issue(k, ins, {}, std::span(&out, 1), site, std::span(&res, 1),
                  static_cast<std::int64_t>(l.size()) * 4);
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
    return vp_reduce_out(rt, ufunc, in, axes, keepdims, dtype, identity, nullptr, result, site, err,
                         errlen);
}

// The result layout NumPy would allocate, with out's dtype and the strides
// of every axis of extent > 1: then NumPy's iteration with `out` given is
// the one with the result it allocates.
static bool same_result_layout(const Layout& out, const Layout& want) {
    if (out.dtype != want.dtype || out.shape != want.shape) {
        return false;
    }
    for (std::size_t i = 0; i < out.shape.size(); ++i) {
        if (out.shape[i] != 1 && out.strides[i] != want.strides[i]) {
            return false;
        }
    }
    return true;
}

namespace {

// The fast path's kernel: no cast, a dense input (in any axis order) whose
// reduced axes coalesce: the loop calls follow from the plan. Null if the
// reduction does not qualify, or with a Python error set if the identity
// does not convert. Everything else, NumPy's iterator drives: it may buffer
// strided inputs even without a cast, and a loop called on a buffer can
// round differently than on strided data (complex multiply has a SIMD path
// for contiguous operands only).
std::shared_ptr<Kernel> make_fast_reduce_kernel(vp_runtime* rt, PyObject* ufunc, const Layout& l,
                                                std::span<const bool> reduced,
                                                const Layout& out_layout, PyObject* dtype,
                                                PyObject* identity) {
    if (dtype != Py_None) {
        return nullptr;
    }
    numpy::ReducePlan plan = numpy::plan_reduction(l, reduced, out_layout);
    Layout dense_check = l;
    dense_check.offset = 0;
    if (!plan.supported() || !dense_check.dense()) {
        return nullptr;
    }
    std::string e;
    const numpy::UfuncLoop* loop =
        rt->loops->get_reduction(ufunc, l.dtype, plan.inner_strides(), e);
    if (loop == nullptr) {
        PyErr_Clear();
        return nullptr;
    }
    std::vector<std::byte> init;
    if (identity != Py_None && !numpy::scalar_bytes(identity, l.dtype, init)) {
        return nullptr; // the Python error stays set
    }
    return numpy::make_reduce_kernel(loop, std::move(plan), std::move(init));
}

} // namespace

int vp_reduce_out(vp_runtime* rt, PyObject* ufunc, vp_array* in, const unsigned char* axes,
                  int keepdims, PyObject* dtype, PyObject* identity, vp_array* out,
                  vp_array** result, const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    const Layout& l = arr(in).layout();
    if (l.size() == 0) {
        set_error(err, errlen, "empty reduction"); // NumPy's identity / error handling
        return VP_UNSUPPORTED;
    }
    const std::size_t nd = l.shape.size();
    SmallVector<char, kInlineRank> flags(nd, 0); // no std::vector<bool>: data()
    Shape out_shape;
    std::uint64_t axis_bits = 0;
    for (std::size_t i = 0; i < nd; ++i) {
        flags[i] = axes[i] != 0;
        if (!flags[i]) {
            out_shape.push_back(l.shape[i]);
        } else {
            axis_bits |= std::uint64_t{1} << (i % 64);
            if (keepdims) {
                out_shape.push_back(1);
            }
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
    ReduceKey key{ufunc, l, axis_bits, {}};
    key.layout.offset = 0;
    const bool cacheable = dtype == Py_None && nd <= 64 && identity_key(identity, key.identity);
    std::shared_ptr<Kernel> kernel;
    if (cacheable) {
        auto it = rt->reduce_kernels.find(key);
        if (it != rt->reduce_kernels.end()) {
            kernel = it->second;
        }
    }
    if (!kernel) {
        kernel = make_fast_reduce_kernel(rt, ufunc, l, reduced, out_layout, dtype, identity);
        if (!kernel && PyErr_Occurred()) {
            return VP_PYERR;
        }
        if (kernel && cacheable) {
            if (rt->reduce_kernels.size() >= 4096) {
                rt->reduce_kernels.clear();
            }
            rt->reduce_kernels.emplace(std::move(key), kernel);
        }
    }
    if (!kernel) {
        std::string e;
        Layout rl;
        std::shared_ptr<Kernel> k = rt->loops->iter_reduction(
            ufunc, l, reduced, keepdims != 0, dtype, identity, defer_release(rt->graveyard), rl, e);
        if (!k) {
            PyErr_Clear();
            set_error(err, errlen, e);
            return VP_UNSUPPORTED;
        }
        const ArrayPtr ins[] = {ref(in)};
        if (out != nullptr) {
            if (!same_result_layout(arr(out).layout(), rl)) {
                set_error(err, errlen, "reduce: out= has another layout or dtype");
                return VP_UNSUPPORTED;
            }
            const ArrayPtr io[] = {ref(out)};
            rt->rt->issue(k, ins, io, {}, site, {}, l.size());
            *result = nullptr;
            return VP_OK;
        }
        ArrayPtr res;
        rt->rt->issue(k, ins, {}, std::span(&rl, 1), site, std::span(&res, 1), l.size());
        *result = hand_out(std::move(res));
        return VP_OK;
    }
    const ArrayPtr ins[] = {ref(in)};
    if (out != nullptr) {
        if (!same_result_layout(arr(out).layout(), out_layout)) {
            set_error(err, errlen, "reduce: out= has another layout or dtype");
            return VP_UNSUPPORTED;
        }
        const ArrayPtr io[] = {ref(out)};
        rt->rt->issue(kernel, ins, io, {}, site, {}, l.size());
        *result = nullptr;
        return VP_OK;
    }
    ArrayPtr res;
    rt->rt->issue(kernel, ins, {}, std::span(&out_layout, 1), site, std::span(&res, 1), l.size());
    *result = hand_out(std::move(res));
    return VP_OK;
}

int vp_accumulate(vp_runtime* rt, PyObject* ufunc, vp_array* in, int axis, vp_array** result,
                  const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    const Layout& l = arr(in).layout();
    const auto nd = static_cast<int>(l.shape.size());
    if (nd == 0 || axis < 0 || axis >= nd || l.size() == 0) {
        set_error(err, errlen, "accumulate: empty or 0-d input");
        return VP_UNSUPPORTED;
    }
    // NumPy allocates a 1-d result C-contiguous, an n-d one through its
    // iterator in order K
    Layout out = nd == 1 ? Layout::contiguous(l.dtype, l.shape)
                         : Layout::keep_order(l.dtype, l.shape, std::span(&l, 1));
    const auto ax = static_cast<std::size_t>(axis);
    const auto sout = static_cast<std::intptr_t>(out.strides[ax]);
    const auto sin = static_cast<std::intptr_t>(l.strides[ax]);
    std::string e;
    const numpy::UfuncLoop* loop = rt->loops->get_reduction(ufunc, l.dtype, {sout, sin, sout}, e);
    if (loop == nullptr) {
        PyErr_Clear();
        set_error(err, errlen, e);
        return VP_UNSUPPORTED;
    }
    const ArrayPtr ins[] = {ref(in)};
    ArrayPtr res;
    rt->rt->issue(numpy::make_accumulate_kernel(loop, axis), ins, {}, std::span(&out, 1), site,
                  std::span(&res, 1), l.size());
    *result = hand_out(std::move(res));
    return VP_OK;
}

int vp_reduceat(vp_runtime* rt, PyObject* ufunc, vp_array* in, int axis, const int64_t* indices,
                int nindices, const int64_t* out_strides, vp_array** result, const char* site,
                char* err, size_t errlen) {
    rt->graveyard->drain();
    const Layout& l = arr(in).layout();
    const auto nd = static_cast<int>(l.shape.size());
    if (nd == 0 || axis < 0 || axis >= nd || nindices <= 0) {
        set_error(err, errlen, "reduceat: 0-d input or no indices");
        return VP_UNSUPPORTED;
    }
    const auto ax = static_cast<std::size_t>(axis);
    std::vector<std::int64_t> ind(indices, indices + nindices);
    for (const std::int64_t i : ind) {
        if (i < 0 || i >= l.shape[ax]) {
            set_error(err, errlen, "reduceat: index out of range"); // NumPy raises it
            return VP_UNSUPPORTED;
        }
    }
    Shape shape = l.shape;
    shape[ax] = nindices;
    Layout out = Layout::contiguous(l.dtype, shape);
    if (out_strides != nullptr) {
        for (int i = 0; i < nd; ++i) {
            out.strides[static_cast<std::size_t>(i)] = out_strides[i];
        }
        if (!out.dense()) {
            set_error(err, errlen, "reduceat: result strides not dense");
            return VP_INVALID;
        }
    }
    if (out.size() == 0) {
        set_error(err, errlen, "reduceat: empty result");
        return VP_UNSUPPORTED;
    }
    std::string e;
    const numpy::UfuncLoop* loop = rt->loops->get_reduction(
        ufunc, l.dtype, {0, static_cast<std::intptr_t>(l.strides[ax]), 0}, e);
    if (loop == nullptr) {
        PyErr_Clear();
        set_error(err, errlen, e);
        return VP_UNSUPPORTED;
    }
    const ArrayPtr ins[] = {ref(in)};
    ArrayPtr res;
    rt->rt->issue(numpy::make_reduceat_kernel(loop, axis, std::move(ind)), ins, {},
                  std::span(&out, 1), site, std::span(&res, 1), l.size());
    *result = hand_out(std::move(res));
    return VP_OK;
}

int vp_ufunc_at(vp_runtime* rt, const vp_loop* loop, vp_array* a, vp_array* indices,
                vp_array* values, const char* site, char* err, size_t errlen) {
    rt->graveyard->drain();
    const Layout& al = arr(a).layout();
    const Layout& xl = arr(indices).layout();
    const int nin = loop->loop->nin();
    bool same =
        loop->loop->nout() == 1 && (nin == 1 || nin == 2) && (nin == 2) == (values != nullptr);
    for (const DType d : loop->dtypes) {
        same = same && d == al.dtype;
    }
    if (!same || al.shape.empty() || xl.shape.size() != 1 ||
        (xl.dtype != DType::Int32 && xl.dtype != DType::Int64)) {
        set_error(err, errlen, "ufunc.at: unsupported operands");
        return VP_UNSUPPORTED;
    }
    if (values != nullptr) {
        Shape full = al.shape;
        full[0] = xl.shape[0];
        Shape b;
        const Layout& vl = arr(values).layout();
        if (vl.dtype != al.dtype || !broadcast_shapes(vl.shape, full, b) || b != full) {
            set_error(err, errlen, "ufunc.at: values do not broadcast");
            return VP_UNSUPPORTED;
        }
    }
    std::int64_t work = xl.shape[0];
    for (std::size_t i = 1; i < al.shape.size(); ++i) {
        work *= al.shape[i];
    }
    ArrayPtr ins[2] = {ref(indices)};
    std::size_t nins = 1;
    if (values != nullptr) {
        ins[nins++] = ref(values);
    }
    const ArrayPtr io[] = {ref(a)};
    rt->rt->issue(numpy::make_at_kernel(loop->loop), std::span(ins, nins), io, {}, site, {}, work);
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
