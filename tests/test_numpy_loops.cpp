// Spike / regression test for the core assumption of phase 1: NumPy's inner
// loops, obtained through ufunc._resolve_dtypes_and_context() and
// ufunc._get_strided_loop(), can be called from a worker thread that does
// not hold the GIL, and produce results bit-identical to eager NumPy.
//
// Covered: element-wise ufunc with broadcasting and a 0-d scalar, reduction,
// linalg gufunc (det), fft gufunc, argmax via PyArray_ArrFuncs, and
// floating-point error flags raised on the worker thread.

#include "embed_python.h"

#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/dtype_api.h>

#include <algorithm>
#include <cfenv>
#include <cstdio>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace {

// Prefix of NumPy's `ufunc_call_info` (capsule "numpy_1.24_ufunc_call_info",
// documented in the docstring of ufunc._get_strided_loop). Only the leading
// fields are mirrored: the embedded PyArrayMethod_Context that follows them
// changes size between NumPy versions.
struct UfuncCallInfoPrefix {
    PyArrayMethod_StridedLoop* strided_loop;
    PyArrayMethod_Context* context;
    NpyAuxData* auxdata;
    npy_bool requires_pyapi;
    npy_bool no_floatingpoint_errors;
};

constexpr const char* kCallInfoCapsule = "numpy_1.24_ufunc_call_info";

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        ++g_failures;
    }
}

// Owns the capsule (and with it the loop's auxdata and context).
struct ResolvedLoop {
    PyObject* capsule = nullptr;
    UfuncCallInfoPrefix* info = nullptr;
};

// Caller thread, GIL held. `dtypes` is a tuple of dtype objects or None.
// `fixed_strides` (a tuple, or nullptr) is passed to _get_strided_loop like
// NumPy does when it knows the strides; it may select a specialized loop.
ResolvedLoop resolve_loop(PyObject* ufunc, PyObject* dtypes, bool reduction,
                          PyObject* fixed_strides = nullptr) {
    ResolvedLoop r;
    PyObject* meth = PyObject_GetAttrString(ufunc, "_resolve_dtypes_and_context");
    PyObject* args = PyTuple_Pack(1, dtypes);
    PyObject* kwargs = Py_BuildValue("{s:O}", "reduction", reduction ? Py_True : Py_False);
    PyObject* res = PyObject_Call(meth, args, kwargs);
    Py_DECREF(meth);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    if (res == nullptr) {
        PyErr_Print();
        return r;
    }
    r.capsule = PyTuple_GetItem(res, 1);
    Py_INCREF(r.capsule);
    Py_DECREF(res);

    PyObject* get = PyObject_GetAttrString(ufunc, "_get_strided_loop");
    PyObject* get_args = PyTuple_Pack(1, r.capsule);
    PyObject* get_kwargs = PyDict_New();
    if (fixed_strides != nullptr) {
        PyDict_SetItemString(get_kwargs, "fixed_strides", fixed_strides);
    }
    PyObject* filled = PyObject_Call(get, get_args, get_kwargs);
    Py_DECREF(get);
    Py_DECREF(get_args);
    Py_DECREF(get_kwargs);
    if (filled == nullptr) {
        PyErr_Print();
        Py_CLEAR(r.capsule);
        return r;
    }
    Py_DECREF(filled);
    r.info = static_cast<UfuncCallInfoPrefix*>(PyCapsule_GetPointer(r.capsule, kCallInfoCapsule));
    return r;
}

// Runs `fn` on a fresh thread while the caller has released the GIL.
// Returns whether the worker thread held the GIL (it must not).
bool run_on_worker_without_gil(const std::function<void()>& fn) {
    bool worker_had_gil = true;
    PyThreadState* saved = PyEval_SaveThread();
    std::thread worker([&] {
        worker_had_gil = PyGILState_Check() != 0;
        fn();
    });
    worker.join();
    PyEval_RestoreThread(saved);
    return worker_had_gil;
}

PyObject* np_attr(PyObject* np, const char* path) {
    // Resolves dotted attribute paths like "linalg._umath_linalg.det".
    PyObject* cur = np;
    Py_INCREF(cur);
    std::string p(path);
    size_t start = 0;
    while (start <= p.size()) {
        size_t dot = p.find('.', start);
        std::string name =
            p.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
        PyObject* next = PyObject_GetAttrString(cur, name.c_str());
        Py_DECREF(cur);
        if (next == nullptr) {
            PyErr_Print();
            return nullptr;
        }
        cur = next;
        if (dot == std::string::npos) {
            break;
        }
        start = dot + 1;
    }
    return cur;
}

bool array_equal(PyObject* np, PyObject* a, PyObject* b) {
    PyObject* r = PyObject_CallMethod(np, "array_equal", "OO", a, b);
    bool eq = r != nullptr && PyObject_IsTrue(r) == 1;
    Py_XDECREF(r);
    return eq;
}

PyObject* eval(const char* expr, PyObject* globals) {
    PyObject* r = PyRun_String(expr, Py_eval_input, globals, globals);
    if (r == nullptr) {
        PyErr_Print();
    }
    return r;
}

char* data_of(PyObject* arr) { return static_cast<char*>(PyArray_DATA((PyArrayObject*)arr)); }

void test_binary_broadcast(PyObject* np, PyObject* g) {
    // out[i, j] = a[i, j] + s, with s a true 0-d scalar (stride 0).
    PyObject* a = eval("rng.standard_normal((64, 33))", g);
    PyObject* s = eval("np.array(2.5)", g);
    PyDict_SetItemString(g, "_a", a);
    PyDict_SetItemString(g, "_s", s);
    PyObject* expected = eval("np.add(_a, _s)", g);
    PyObject* out = eval("np.empty_like(_a)", g);

    PyObject* add = np_attr(np, "add");
    PyObject* dtypes = eval("(np.dtype('f8'), np.dtype('f8'), None)", g);
    ResolvedLoop loop = resolve_loop(add, dtypes, false);
    check(loop.info != nullptr && !loop.info->requires_pyapi, "add f8: loop resolved, no pyapi");

    npy_intp n = PyArray_SIZE((PyArrayObject*)a);
    char* data[3] = {data_of(a), data_of(s), data_of(out)};
    npy_intp dims[1] = {n};
    npy_intp strides[3] = {sizeof(double), 0, sizeof(double)};
    int rc = -1;
    bool had_gil = run_on_worker_without_gil([&] {
        rc = loop.info->strided_loop(loop.info->context, data, dims, strides, loop.info->auxdata);
    });
    check(!had_gil, "add f8: worker ran without GIL");
    check(rc == 0 && array_equal(np, out, expected), "add f8 + 0-d scalar: bit-identical");

    Py_XDECREF(loop.capsule);
    Py_DECREF(dtypes);
    Py_DECREF(add);
    Py_DECREF(out);
    Py_DECREF(expected);
    Py_DECREF(a);
    Py_DECREF(s);
}

// Sums `a` with NumPy's add loop on a worker thread, the way a reduction
// calls it: args = {acc, in, acc}, acc stride 0, acc starting at 0 (add's
// identity). The input is passed in calls of at most `chunk` elements.
PyObject* worker_sum(PyObject* np, PyObject* g, PyObject* a, PyObject* fixed_strides,
                     npy_intp chunk) {
    PyObject* acc = eval("np.zeros((), dtype='f8')", g);
    PyObject* add = np_attr(np, "add");
    PyObject* dtypes = eval("(None, np.dtype('f8'), None)", g);
    ResolvedLoop loop = resolve_loop(add, dtypes, true, fixed_strides);
    Py_DECREF(dtypes);
    Py_DECREF(add);
    if (loop.info == nullptr) {
        Py_DECREF(acc);
        return nullptr;
    }
    const npy_intp n = PyArray_SIZE((PyArrayObject*)a);
    char* in = data_of(a);
    int rc = 0;
    run_on_worker_without_gil([&] {
        for (npy_intp start = 0; start < n && rc == 0; start += chunk) {
            char* data[3] = {data_of(acc), in + start * sizeof(double), data_of(acc)};
            npy_intp dims[1] = {std::min(chunk, n - start)};
            npy_intp strides[3] = {0, sizeof(double), 0};
            rc = loop.info->strided_loop(loop.info->context, data, dims, strides,
                                         loop.info->auxdata);
        }
    });
    Py_XDECREF(loop.capsule);
    if (rc != 0) {
        Py_DECREF(acc);
        return nullptr;
    }
    return acc;
}

void test_sum_reduction(PyObject* np, PyObject* g) {
    PyObject* a = eval("rng.standard_normal(100003)", g);
    PyDict_SetItemString(g, "_a", a);
    PyObject* expected = eval("np.sum(_a)", g);
    PyObject* fixed = eval("(0, 8, 0)", g);
    const npy_intp n = PyArray_SIZE((PyArrayObject*)a);

    // Variants of calling the loop; NumPy must match the first one. The others
    // are printed to find out how NumPy calls it where it does not.
    struct Variant {
        const char* name;
        PyObject* fixed_strides;
        npy_intp chunk;
    };
    const Variant variants[] = {
        {"one call, fixed strides", fixed, n},
        {"one call, generic strides", nullptr, n},
        {"chunks of 8192, fixed strides", fixed, 8192},
        {"chunks of 8192, generic strides", nullptr, 8192},
    };
    bool canonical = false;
    std::string matching;
    for (const Variant& v : variants) {
        PyObject* got = worker_sum(np, g, a, v.fixed_strides, v.chunk);
        const bool eq = got != nullptr && array_equal(np, got, expected);
        if (eq) {
            matching += std::string(matching.empty() ? "" : ", ") + v.name;
        }
        canonical = canonical || (eq && &v == &variants[0]);
        Py_XDECREF(got);
    }
    check(canonical, "sum f8: bit-identical to np.sum (one loop call, fixed strides)");
    if (!canonical) {
        PyObject* info =
            eval("f'numpy {np.__version__}, np.sum = {float(np.sum(_a)).hex()}, '"
                 "f'dispatch {getattr(np._core._multiarray_umath, \"__cpu_dispatch__\", \"?\")}'",
                 g);
        std::printf("     info: %s\n", info != nullptr ? PyUnicode_AsUTF8(info) : "?");
        std::printf("     variants matching np.sum: %s\n",
                    matching.empty() ? "none" : matching.c_str());
        Py_XDECREF(info);
    }

    Py_DECREF(fixed);
    Py_DECREF(expected);
    Py_DECREF(a);
}

void test_linalg_det(PyObject* np, PyObject* g) {
    // gufunc (m,m)->() over a stack of 10 matrices.
    PyObject* a = eval("rng.standard_normal((10, 5, 5))", g);
    PyDict_SetItemString(g, "_a", a);
    PyObject* expected = eval("np.linalg.det(_a)", g);
    PyObject* out = eval("np.empty(10)", g);

    PyObject* det = np_attr(np, "linalg._umath_linalg.det");
    PyObject* dtypes = eval("(np.dtype('f8'), None)", g);
    ResolvedLoop loop = resolve_loop(det, dtypes, false);
    check(loop.info != nullptr && !loop.info->requires_pyapi,
          "linalg det: loop resolved, no pyapi");

    char* data[2] = {data_of(a), data_of(out)};
    npy_intp dims[2] = {10, 5};                                 // outer count, m
    npy_intp strides[4] = {25 * sizeof(double), sizeof(double), // outer: in, out
                           5 * sizeof(double), sizeof(double)}; // core: in (m, m)
    int rc = -1;
    run_on_worker_without_gil([&] {
        rc = loop.info->strided_loop(loop.info->context, data, dims, strides, loop.info->auxdata);
    });
    check(rc == 0 && array_equal(np, out, expected), "linalg det: bit-identical");

    Py_XDECREF(loop.capsule);
    Py_DECREF(dtypes);
    Py_DECREF(det);
    Py_DECREF(out);
    Py_DECREF(expected);
    Py_DECREF(a);
}

void test_fft(PyObject* np, PyObject* g) {
    // gufunc (n),()->(m) with fct = 1 (norm="backward", forward transform).
    PyObject* a = eval("rng.standard_normal((4, 64)) + 1j * rng.standard_normal((4, 64))", g);
    PyDict_SetItemString(g, "_a", a);
    PyObject* expected = eval("np.fft.fft(_a)", g);
    PyObject* out = eval("np.empty_like(_a)", g);
    double fct = 1.0;

    PyObject* fft = np_attr(np, "fft._pocketfft_umath.fft");
    PyObject* dtypes = eval("(np.dtype('c16'), np.dtype('f8'), np.dtype('c16'))", g);
    ResolvedLoop loop = resolve_loop(fft, dtypes, false);
    check(loop.info != nullptr && !loop.info->requires_pyapi, "fft: loop resolved, no pyapi");

    const npy_intp c = 2 * sizeof(double);
    char* data[3] = {data_of(a), reinterpret_cast<char*>(&fct), data_of(out)};
    npy_intp dims[3] = {4, 64, 64};           // outer count, n, m
    npy_intp strides[5] = {64 * c, 0, 64 * c, // outer: in, fct, out
                           c, c};             // core: in (n), out (m)
    int rc = -1;
    run_on_worker_without_gil([&] {
        rc = loop.info->strided_loop(loop.info->context, data, dims, strides, loop.info->auxdata);
    });
    check(rc == 0 && array_equal(np, out, expected), "fft c16: bit-identical");

    Py_XDECREF(loop.capsule);
    Py_DECREF(dtypes);
    Py_DECREF(fft);
    Py_DECREF(out);
    Py_DECREF(expected);
    Py_DECREF(a);
}

void test_argmax(PyObject* g) {
    PyObject* a = eval("rng.standard_normal(1001)", g);
    PyDict_SetItemString(g, "_a", a);
    PyObject* expected_obj = eval("int(np.argmax(_a))", g);
    long expected = PyLong_AsLong(expected_obj);

    PyArray_Descr* descr = PyArray_DescrFromType(NPY_DOUBLE);
    PyArray_ArgFunc* argmax = PyDataType_GetArrFuncs(descr)->argmax;
    npy_intp idx = -1;
    npy_intp n = PyArray_SIZE((PyArrayObject*)a);
    char* p = data_of(a);
    int rc = -1;
    run_on_worker_without_gil([&] { rc = argmax(p, n, &idx, nullptr); });
    check(rc == 0 && idx == expected, "argmax f8 via ArrFuncs: matches np.argmax");

    Py_DECREF(descr);
    Py_DECREF(expected_obj);
    Py_DECREF(a);
}

void test_fp_error_flags(PyObject* np, PyObject* g) {
    PyObject* a = eval("np.array([1.0, -1.0, 0.0, 4.0] * 16)", g);
    PyObject* b = eval("np.array([0.0, 0.0, 0.0, 2.0] * 16)", g);
    PyObject* out = eval("np.empty(64)", g);

    PyObject* divide = np_attr(np, "divide");
    PyObject* dtypes = eval("(np.dtype('f8'), np.dtype('f8'), None)", g);
    ResolvedLoop loop = resolve_loop(divide, dtypes, false);
    check(loop.info != nullptr && !loop.info->no_floatingpoint_errors,
          "divide f8: loop may raise FP errors");

    char* data[3] = {data_of(a), data_of(b), data_of(out)};
    npy_intp dims[1] = {64};
    npy_intp strides[3] = {sizeof(double), sizeof(double), sizeof(double)};
    bool divbyzero = false;
    bool invalid = false;
    run_on_worker_without_gil([&] {
        std::feclearexcept(FE_ALL_EXCEPT);
        loop.info->strided_loop(loop.info->context, data, dims, strides, loop.info->auxdata);
        divbyzero = std::fetestexcept(FE_DIVBYZERO) != 0;
        invalid = std::fetestexcept(FE_INVALID) != 0;
    });
    check(divbyzero && invalid, "divide f8: divbyzero and invalid flags visible on worker");

    Py_XDECREF(loop.capsule);
    Py_DECREF(dtypes);
    Py_DECREF(divide);
    Py_DECREF(out);
    Py_DECREF(a);
    Py_DECREF(b);
}

int init_numpy() {
    import_array1(-1);
    return 0;
}

} // namespace

int main() {
    if (!init_embedded_python()) {
        return 1;
    }
    if (init_numpy() != 0) {
        PyErr_Print();
        return 1;
    }
    PyObject* np = PyImport_ImportModule("numpy");
    PyObject* linalg = PyImport_ImportModule("numpy.linalg._umath_linalg");
    PyObject* fft = PyImport_ImportModule("numpy.fft._pocketfft_umath");
    if (np == nullptr || linalg == nullptr || fft == nullptr) {
        PyErr_Print();
        return 1;
    }
    PyObject* g = PyDict_New();
    PyDict_SetItemString(g, "__builtins__", PyEval_GetBuiltins());
    PyDict_SetItemString(g, "np", np);
    PyObject* rng = eval("np.random.default_rng(42)", g);
    PyDict_SetItemString(g, "rng", rng);
    Py_DECREF(rng);

    test_binary_broadcast(np, g);
    test_sum_reduction(np, g);
    test_linalg_det(np, g);
    test_fft(np, g);
    test_argmax(g);
    test_fp_error_flags(np, g);

    Py_DECREF(g);
    Py_DECREF(fft);
    Py_DECREF(linalg);
    Py_DECREF(np);
    if (Py_FinalizeEx() < 0) {
        return 1;
    }
    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
