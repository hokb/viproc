/* CPython extension module `viproc._viproc`: binding of the C ABI in
 * include/viproc/viproc.h, plus the hot path of the Python bridge.
 *
 * `_Array` is the base class of `viproc.ndarray` (defined in _ndarray.py). It
 * owns one vp_array* and implements, in C, everything that runs once per
 * array op: `__array_ufunc__` for element-wise ufunc calls and the Python
 * operators (`+`, `*=`, `<`, ...). Dtype resolution is NumPy's own
 * (`ufunc.resolve_dtypes`), cached per (ufunc, input kinds). Anything the fast
 * path does not handle goes to the Python fallback registered by
 * _ndarray.py, which keeps NumPy semantics for all other cases. */

#include "viproc_python.h"

#include "viproc/viproc.h"

#include <assert.h>
#include <string.h>

/* Temporary elision (docs/design/memory-management.md, D5) needs to know
 * that an operand is referenced only by the interpreter's value stack.
 * CPython >= 3.14 answers that exactly. Before, a reference count of 1 is
 * necessary but not sufficient (C code may hold a borrowed reference to an
 * object owned elsewhere), so the C call chain is checked as NumPy does:
 * only frames of this module and of the Python runtime between the operator
 * slot and the bytecode interpreter. That needs the return addresses of the
 * C stack and the module each belongs to: backtrace() and dladdr() with
 * glibc, RtlCaptureStackBackTrace() and GetModuleHandleExW() on Windows. */
#if PY_VERSION_HEX >= 0x030E0000
#define VP_ELIDE_EXACT 1
#elif defined(_WIN32)
#define VP_ELIDE_BACKTRACE 1
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__GLIBC__)
#define VP_ELIDE_BACKTRACE 1
#include <dlfcn.h>
#include <execinfo.h>
#endif

#define ERRLEN 1024
#define MAXDIMS 64
#define MAXIN 3 /* element-wise ufuncs have at most 3 inputs */

static vp_runtime* g_rt = NULL;
static unsigned long g_main_thread = 0;
static PyObject* AsyncError = NULL;
static PyObject* AsyncIndexError = NULL; /* AsyncError and IndexError */
static PyObject* AsyncValueError = NULL; /* AsyncError and ValueError */
static long long g_small = 2048;         /* the runtime's sync_below: ops this small run directly */

/* Registered by _ndarray.py via _viproc.setup(). */
static PyTypeObject* g_array_cls = NULL; /* viproc.ndarray */
static PyObject* g_dtypes[7];            /* np.dtype per vp_dtype code */
static PyObject* g_matmul = NULL;        /* np.matmul */
static PyObject* g_fallback = NULL;      /* fallback(ufunc, method, inputs, kwargs) */
static PyObject* g_asarray = NULL;       /* asarray(ndarray) -> viproc.ndarray */
static PyObject* g_internal_dirs = NULL; /* tuple of path prefixes skipped for issue sites */
static PyObject* g_np_ndarray = NULL;    /* numpy.ndarray */
static PyObject* g_np_generic = NULL;    /* numpy.generic */
static PyObject* g_where_func = NULL;    /* numpy.where */
static PyObject* g_where_impl = NULL;    /* numpy.where._implementation */
static PyObject* g_np_integer = NULL;    /* numpy.integer */
static PyObject* g_ops = NULL;           /* dict: operator name -> ufunc */
static PyObject* g_getitem = NULL;       /* getitem(a, key): a[key] beyond the C fast path */
static PyObject* g_setitem = NULL;       /* setitem(a, key, v): a[key] = v beyond C */

/* Caches (main thread only). */

/* Per-op lookups use small open-addressing tables keyed by (pointer, int)
 * instead of dicts: no key objects to build, no Python hashing. Code objects
 * in particular are keyed by address (hashing one hashes its bytecode). Each
 * entry holds a strong reference to the object its key points to, so the
 * address cannot be reused while it is cached. */
typedef struct {
    const void* a; /* NULL: empty slot */
    long long b;
    long long v;
    const void* p;
    PyObject* keep;
} Entry;

typedef struct {
    Entry* e;
    size_t cap; /* power of two, or 0 */
    size_t n;
} Table;

static Table g_code_flags; /* (code, 0) -> v: frame is inside viproc/NumPy */
static Table g_sites;      /* (code, lasti) -> p: interned site (per runtime) */
static Table g_sigs;       /* (ufunc, nin|kinds) -> v: packed dtype codes or -1, p: vp_loop */

static size_t t_slot(const Table* t, const void* a, long long b) {
    unsigned long long h = (unsigned long long)(uintptr_t)a * 0x9E3779B97F4A7C15ull ^
                           (unsigned long long)b * 0xC2B2AE3D27D4EB4Full;
    return (size_t)(h ^ (h >> 31)) & (t->cap - 1);
}

static Entry* t_find(const Table* t, const void* a, long long b) {
    if (t->cap == 0) {
        return NULL;
    }
    for (size_t i = t_slot(t, a, b);; i = (i + 1) & (t->cap - 1)) {
        Entry* e = &t->e[i];
        if (e->a == a && e->b == b) {
            return e;
        }
        if (e->a == NULL) {
            return NULL;
        }
    }
}

/* Inserts a key that is not in the table. Returns 0, or -1 (MemoryError). */
static int t_put(Table* t, const void* a, long long b, long long v, const void* p, PyObject* keep) {
    if (2 * (t->n + 1) > t->cap) {
        const size_t cap = t->cap != 0 ? 2 * t->cap : 64;
        Entry* e = PyMem_Calloc(cap, sizeof(Entry));
        if (e == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        Table grown = {e, cap, 0};
        for (size_t i = 0; i < t->cap; ++i) {
            if (t->e[i].a != NULL) {
                size_t j = t_slot(&grown, t->e[i].a, t->e[i].b);
                while (e[j].a != NULL) {
                    j = (j + 1) & (cap - 1);
                }
                e[j] = t->e[i];
            }
        }
        grown.n = t->n;
        PyMem_Free(t->e);
        *t = grown;
    }
    size_t i = t_slot(t, a, b);
    while (t->e[i].a != NULL) {
        i = (i + 1) & (t->cap - 1);
    }
    t->e[i] = (Entry){a, b, v, p, Py_XNewRef(keep)};
    ++t->n;
    return 0;
}

static void t_clear(Table* t) {
    Entry* e = t->e;
    const size_t cap = t->cap;
    *t = (Table){NULL, 0, 0};
    for (size_t i = 0; i < cap; ++i) {
        Py_XDECREF(e[i].keep);
    }
    PyMem_Free(e);
}

static PyObject* s_call = NULL; /* "__call__" */
static PyObject* s_out = NULL;  /* "out" */

/* --- _Array: base class of viproc.ndarray ---------------------------------- */

typedef struct {
    PyObject_HEAD vp_array* a;
    int readonly; /* NumPy's WRITEABLE == False (e.g. views NumPy made read-only) */
    int scalar;   /* 0-d result NumPy returns as a scalar (reductions, a[i]) */
} ArrayObject;

static PyTypeObject ArrayType; /* forward */

static void Array_dealloc(ArrayObject* self) {
    if (self->a != NULL) {
        vp_array_release(self->a);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* New viproc.ndarray owning `a` (released on failure). */
static PyObject* wrap_array(vp_array* a) {
    PyTypeObject* cls = g_array_cls != NULL ? g_array_cls : &ArrayType;
    ArrayObject* o = (ArrayObject*)cls->tp_alloc(cls, 0);
    if (o == NULL) {
        vp_array_release(a);
        return NULL;
    }
    o->a = a;
    return (PyObject*)o;
}

static int is_array(PyObject* o) {
    return Py_TYPE(o) == g_array_cls || PyObject_TypeCheck(o, &ArrayType);
}

static vp_array* array_of(PyObject* o) {
    if (!is_array(o)) {
        PyErr_SetString(PyExc_TypeError, "expected a viproc array");
        return NULL;
    }
    return ((ArrayObject*)o)->a;
}

static PyObject* dims_tuple(int n, const int64_t* d) {
    PyObject* t = PyTuple_New(n);
    for (int i = 0; t != NULL && i < n; ++i) {
        PyTuple_SET_ITEM(t, i, PyLong_FromLongLong(d[i]));
    }
    return t;
}

static PyObject* Array_get_shape(ArrayObject* self, void* c) {
    (void)c;
    return dims_tuple(vp_array_ndim(self->a), vp_array_shape(self->a));
}

static PyObject* Array_get_strides(ArrayObject* self, void* c) {
    (void)c;
    return dims_tuple(vp_array_ndim(self->a), vp_array_strides(self->a));
}

static PyObject* Array_get_ndim(ArrayObject* self, void* c) {
    (void)c;
    return PyLong_FromLong(vp_array_ndim(self->a));
}

static PyObject* Array_get_dtype(ArrayObject* self, void* c) {
    (void)c;
    PyObject* d = g_dtypes[vp_array_dtype(self->a)];
    if (d == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "viproc._viproc.setup() was not called");
        return NULL;
    }
    return Py_NewRef(d);
}

static PyObject* Array_get_scalar(ArrayObject* self, void* c) {
    (void)c;
    return PyBool_FromLong(self->scalar);
}

static int Array_set_scalar(ArrayObject* self, PyObject* v, void* c) {
    (void)c;
    const int t = v != NULL ? PyObject_IsTrue(v) : 0;
    if (t < 0) {
        return -1;
    }
    self->scalar = t;
    return 0;
}

static PyObject* Array_get_readonly(ArrayObject* self, void* c) {
    (void)c;
    return PyBool_FromLong(self->readonly);
}

static int Array_set_readonly(ArrayObject* self, PyObject* v, void* c) {
    (void)c;
    const int t = v != NULL ? PyObject_IsTrue(v) : 0;
    if (t < 0) {
        return -1;
    }
    self->readonly = t;
    return 0;
}

static int c_contiguous(const vp_array* a);

static PyObject* Array_get_size(ArrayObject* self, void* c) {
    (void)c;
    const int nd = vp_array_ndim(self->a);
    const int64_t* shape = vp_array_shape(self->a);
    long long n = 1;
    for (int i = 0; i < nd; ++i) {
        n *= (long long)shape[i];
    }
    return PyLong_FromLongLong(n);
}

static PyObject* Array_get_c_contiguous(ArrayObject* self, void* c) {
    (void)c;
    return PyBool_FromLong(c_contiguous(self->a));
}

static PyGetSetDef Array_getset[] = {
    {"size", (getter)Array_get_size, NULL, "Number of elements (no sync).", NULL},
    {"_c_contiguous", (getter)Array_get_c_contiguous, NULL,
     "C-contiguous, as NumPy's flag (no sync).", NULL},
    {"_scalar", (getter)Array_get_scalar, (setter)Array_set_scalar,
     "0-d result NumPy would return as a scalar: in-place operators rebind, as on NumPy's "
     "scalars",
     NULL},
    {"_readonly", (getter)Array_get_readonly, (setter)Array_set_readonly,
     "True if writes raise, as for a NumPy array with WRITEABLE == False.", NULL},
    {"shape", (getter)Array_get_shape, NULL, "Shape (no sync).", NULL},
    {"strides", (getter)Array_get_strides, NULL, "Byte strides (no sync).", NULL},
    {"ndim", (getter)Array_get_ndim, NULL, "Number of dimensions (no sync).", NULL},
    {"dtype", (getter)Array_get_dtype, NULL, "Element type (no sync).", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

/* --- helpers -------------------------------------------------------------- */

static int check_runtime(void);
static int check_async_errors(void);
static int code_is_internal(PyCodeObject* code);

/* Arguments (viproc array, any object) of a METH_FASTCALL function; checks
 * the runtime too. 0 or -1 with an exception. */
static int array_and_arg(PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2 || !PyObject_TypeCheck(args[0], &ArrayType)) {
        PyErr_SetString(PyExc_TypeError, "expected (viproc array, object)");
        return -1;
    }
    return check_runtime();
}

static int check_runtime(void) {
    if (g_rt == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "viproc runtime is not initialized");
        return -1;
    }
    if (PyThread_get_thread_ident() != g_main_thread) {
        PyErr_SetString(PyExc_RuntimeError,
                        "viproc may only be used from the thread that initialized it");
        return -1;
    }
    return check_async_errors();
}

/* Raises the error of a failed asynchronous op. Kernels name the Python
 * exception type ("IndexError: ...", "ValueError: ..."): those are raised as
 * AsyncIndexError / AsyncValueError (subclasses of the type NumPy raises),
 * everything else as AsyncError. */
static void raise_async(const char* err) {
    if (strncmp(err, "IndexError: ", 12) == 0) {
        PyErr_SetString(AsyncIndexError, err + 12);
    } else if (strncmp(err, "ValueError: ", 12) == 0) {
        PyErr_SetString(AsyncValueError, err + 12);
    } else {
        PyErr_SetString(AsyncError, err);
    }
}

/* Asynchronous error reporting (docs/design/error-handling.md): raises the
 * error of the earliest failed op not reported yet. 0 if there is none. */
static int raise_pending_error(void) {
    char err[ERRLEN] = "";
    if (vp_take_error(g_rt, err, sizeof err) != VP_ERROR) {
        return 0;
    }
    raise_async(err);
    return -1;
}

/* 1 if the running Python code is the user's: no frame, or the innermost one
 * outside viproc and NumPy. An asynchronous error is raised only there, never
 * inside viproc's or NumPy's own Python code, whose `except ValueError:`
 * blocks would swallow it. */
static int in_user_code(void) {
    PyFrameObject* f = PyEval_GetFrame(); /* borrowed */
    if (f == NULL) {
        return 1;
    }
    PyCodeObject* code = PyFrame_GetCode(f); /* new ref */
    const int internal = code_is_internal(code);
    Py_DECREF(code);
    return !internal;
}

/* At every call into viproc: one atomic load while no op failed. */
static int check_async_errors(void) {
    return vp_error_pending(g_rt) && in_user_code() ? raise_pending_error() : 0;
}

/* Raises a failed op's error in the main thread as soon as it is recorded,
 * wherever the program is (as Python raises KeyboardInterrupt): run by the
 * interpreter's eval loop (Py_AddPendingCall) in the main thread, with the
 * GIL. Inside viproc's or NumPy's own code it is scheduled again, so it
 * surfaces in the user's code. Nothing is done if the error was raised
 * already (a sync point, wait_all, the next call into viproc). */
static int pending_raise(void* arg) {
    (void)arg;
    if (g_rt == NULL || PyThread_get_thread_ident() != g_main_thread || !vp_error_pending(g_rt)) {
        return 0; /* another thread uses viproc: the next call raises it */
    }
    if (!in_user_code()) {
        (void)Py_AddPendingCall(pending_raise, NULL); /* queue full: the next call */
        return 0;
    }
    return raise_pending_error();
}

/* vp_set_error_notify: on the failing thread, no GIL needed. */
static void error_recorded(void* arg) {
    (void)arg;
    (void)Py_AddPendingCall(pending_raise, NULL);
}

static int raise_status(int status, const char* err) {
    switch (status) {
    case VP_OK:
        return 0;
    case VP_PYERR:
        return -1;
    case VP_UNSUPPORTED:
        PyErr_SetString(PyExc_NotImplementedError, err);
        return -1;
    case VP_INVALID:
        PyErr_SetString(PyExc_ValueError, err);
        return -1;
    default:
        raise_async(err);
        return -1;
    }
}

/* Reads a sequence of ints into `out` (at most MAXDIMS). Returns count or -1. */
static int read_dims(PyObject* seq, int64_t* out) {
    PyObject* fast = PySequence_Fast(seq, "expected a sequence of ints");
    if (fast == NULL) {
        return -1;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n > MAXDIMS) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "too many dimensions");
        return -1;
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        out[i] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(fast, i));
        if (out[i] == -1 && PyErr_Occurred()) {
            Py_DECREF(fast);
            return -1;
        }
    }
    Py_DECREF(fast);
    return (int)n;
}

/* 1 if `code` belongs to viproc or NumPy (g_internal_dirs), cached per code
 * object. */
static int code_is_internal(PyCodeObject* code) {
    const Entry* flag = t_find(&g_code_flags, code, 0);
    if (flag != NULL) {
        return (int)flag->v;
    }
    PyObject* fn = PyObject_GetAttrString((PyObject*)code, "co_filename");
    int is_internal = 0;
    if (fn != NULL && g_internal_dirs != NULL) {
        PyObject* name = PyUnicode_InternFromString("startswith");
        PyObject* r = PyObject_CallMethodOneArg(fn, name, g_internal_dirs);
        Py_DECREF(name);
        is_internal = r == Py_True;
        Py_XDECREF(r);
    }
    Py_XDECREF(fn);
    PyErr_Clear();
    if (t_put(&g_code_flags, code, 0, is_internal, NULL, (PyObject*)code) < 0) {
        PyErr_Clear(); /* not cached: computed again next time */
    }
    return is_internal;
}

/* Issue site: "file:line in function" of the first frame outside viproc and
 * NumPy, interned in the runtime (vp_intern) and cached per (code object,
 * bytecode offset). The offset, unlike the line number, is free to read: the
 * line is computed only on a cache miss. Never NULL. */
static const char* issue_site(void) {
    PyFrameObject* f = PyEval_GetFrame(); /* borrowed */
    Py_XINCREF(f);
    while (f != NULL) {
        PyCodeObject* code = PyFrame_GetCode(f); /* new ref */
        const int is_internal = code_is_internal(code);
        if (!is_internal) {
            const int lasti = PyFrame_GetLasti(f);
            const Entry* hit = t_find(&g_sites, code, lasti);
            const char* site = hit != NULL ? (const char*)hit->p : NULL;
            if (site == NULL) {
                PyObject* name = PyObject_GetAttrString((PyObject*)code, "co_qualname");
                PyObject* fn = PyObject_GetAttrString((PyObject*)code, "co_filename");
                PyObject* text = PyBytes_FromFormat("%s:%d in %s", fn ? PyUnicode_AsUTF8(fn) : "?",
                                                    PyFrame_GetLineNumber(f),
                                                    name ? PyUnicode_AsUTF8(name) : "?");
                Py_XDECREF(name);
                Py_XDECREF(fn);
                if (text != NULL) {
                    site = vp_intern(g_rt, PyBytes_AS_STRING(text));
                    Py_DECREF(text);
                    if (t_put(&g_sites, code, lasti, 0, site, (PyObject*)code) < 0) {
                        PyErr_Clear();
                    }
                }
                PyErr_Clear();
            }
            Py_DECREF(code);
            Py_DECREF(f);
            return site != NULL ? site : vp_intern(g_rt, "<unknown>");
        }
        Py_DECREF(code);
        PyFrameObject* back = PyFrame_GetBack(f); /* new ref */
        Py_DECREF(f);
        f = back;
    }
    return vp_intern(g_rt, "<unknown>");
}

/* --- temporary elision (D5) ---------------------------------------------- */

static unsigned long long g_elided = 0; /* ops that reused a temporary's buffer */

#if defined(VP_ELIDE_BACKTRACE)
/* Outputs smaller than this are not worth a backtrace (~1-2 us). NumPy uses
 * the same bound for its own elision. */
#define VP_ELIDE_MIN_BYTES (256 * 1024)

enum { FRAME_OTHER = 0, FRAME_OURS, FRAME_PYTHON };

/* Code address of a function (ISO C has no function -> object pointer cast). */
typedef void (*vp_fn)(void);
static void* code_address(vp_fn f) {
    void* p;
    memcpy(&p, &f, sizeof p);
    return p;
}

/* Load address of the shared object (DLL) containing `addr`, or NULL. */
static void* module_base(void* addr) {
#if defined(_WIN32)
    HMODULE m = NULL;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCWSTR)addr, &m)) {
        return (void*)m; /* an HMODULE is the module's load address */
    }
    return NULL;
#else
    Dl_info info;
    return dladdr(addr, &info) ? info.dli_fbase : NULL;
#endif
}

/* Return addresses of the calling thread's stack, innermost first. */
static int capture_stack(void** frames, int max) {
#if defined(_WIN32)
    return (int)RtlCaptureStackBackTrace(0, (DWORD)max, frames, NULL);
#else
    return backtrace(frames, max);
#endif
}

/* Shared object of a return address: this module, the Python runtime or
 * something else. Cached per address (direct-mapped). */
static int frame_class(void* addr) {
    static struct {
        void* addr;
        int cls;
    } cache[256];
    static void* ours = NULL;
    static void* python = NULL;
    if (ours == NULL) {
        ours = module_base(code_address((vp_fn)frame_class));
        python = module_base(code_address((vp_fn)PyNumber_Add));
    }
    const size_t slot = ((uintptr_t)addr >> 4) & 255;
    if (cache[slot].addr == addr) {
        return cache[slot].cls;
    }
    void* base = module_base(addr);
    int cls = FRAME_OTHER;
    if (base != NULL) {
        cls = base == ours ? FRAME_OURS : base == python ? FRAME_PYTHON : FRAME_OTHER;
    }
    cache[slot].addr = addr;
    cache[slot].cls = cls;
    return cls;
}

/* The return addresses after this module's frames, innermost first: the
 * runtime's operator entry (PyNumber_Add and the helpers it calls, unless
 * inlined or tail-called), the interpreter's call site, and whoever entered
 * the interpreter. Empty if the first one is not in the Python runtime. */
#define CHAIN_FRAMES 8
typedef struct {
    void* f[CHAIN_FRAMES];
    int n;
} call_chain;

static call_chain operator_call_chain(void) {
    void* frames[32];
    const int n = capture_stack(frames, 32);
    int i = 1;
    while (i < n && frame_class(frames[i]) == FRAME_OURS) {
        ++i;
    }
    call_chain c;
    c.n = 0;
    if (i < n && frame_class(frames[i]) == FRAME_PYTHON) {
        for (; i < n && c.n < CHAIN_FRAMES; ++i) {
            c.f[c.n++] = frames[i];
        }
    }
    return c;
}

/* 1 if `c` starts with the frames of `prefix`. */
static int chain_starts_with(const call_chain* c, const call_chain* prefix) {
    return c->n >= prefix->n && memcmp(c->f, prefix->f, (size_t)prefix->n * sizeof(void*)) == 0;
}

/* The call chains of all operators that may elide, as the bytecode
 * interpreter produces them (BINARY_OP, UNARY_NEGATIVE, ...), learned once by
 * running each operator on a probe object, entered two ways: module code
 * (PyEval_EvalCode) and a function called from C. The frames both runs share
 * end with the interpreter's call site; below it they differ. That common
 * prefix is the chain: an operator elides only if its stack starts with it,
 * so any other caller of PyNumber_* (C extensions, builtins such as sum())
 * differs where the interpreter's call site would be. How many frames the
 * runtime's operator entry contributes depends on how CPython was compiled
 * (inlining, tail calls: 0 to 2 frames seen), hence a prefix, not a fixed
 * count. A third run applies the operators from C (operator.add, ...): a
 * chain that one of them matches does not reach the interpreter's call site,
 * and is dropped. Recorded addresses
 * also survive the BOLT/PGO layout of distribution builds, where symbol
 * ranges do not. */
#define MAX_CHAINS 64
static call_chain g_chains[MAX_CHAINS];
static int g_chains_n = -1;     /* -1: not calibrated yet */
static call_chain* g_recording; /* where probe_record writes, or NULL */
static int g_recorded;

static PyObject* probe_record(PyObject* self) {
    if (g_recording != NULL && g_recorded < MAX_CHAINS) {
        g_recording[g_recorded++] = operator_call_chain();
    }
    return Py_NewRef(self);
}
static PyObject* probe_binary(PyObject* a, PyObject* b) {
    (void)b;
    return probe_record(a);
}
static PyNumberMethods probe_as_number = {
    .nb_add = probe_binary,
    .nb_subtract = probe_binary,
    .nb_multiply = probe_binary,
    .nb_remainder = probe_binary,
    .nb_negative = probe_record,
    .nb_positive = probe_record,
    .nb_invert = probe_record,
    .nb_lshift = probe_binary,
    .nb_rshift = probe_binary,
    .nb_and = probe_binary,
    .nb_xor = probe_binary,
    .nb_or = probe_binary,
    .nb_floor_divide = probe_binary,
    .nb_true_divide = probe_binary,
    .nb_matrix_multiply = probe_binary,
};
static PyTypeObject ProbeType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "viproc._viproc._ElisionProbe",
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_as_number = &probe_as_number,
    .tp_new = PyType_GenericNew,
};

/* The probe statements: every binary operator with operands of the same type,
 * with a scalar on the right and with a scalar on the left (reflected), then
 * the unary ones. Each records one chain. The three binary forms take
 * different paths through the runtime's dispatch (binary_op1: slotw unset,
 * slotv then slotw, slotw only), and an optimized build (PGO) may give each
 * path its own call site (seen with GitHub's actions/setup-python CPython). */
static const char* const probe_binops[] = {"+",  "-",  "*", "/", "//", "%",
                                           "<<", ">>", "&", "|", "^",  "@"};
static const char* const probe_opfuncs[] = {"add",      "sub", "mul",    "truediv",
                                            "floordiv", "mod", "lshift", "rshift",
                                            "and_",     "or_", "xor",    "matmul"};
#define N_PROBE_BINOPS 12
#define N_PROBES (3 * N_PROBE_BINOPS + 3)

static int append(char* buf, size_t size, size_t* len, const char* text) {
    const size_t n = strlen(text);
    if (*len + n + 1 > size) {
        return -1;
    }
    memcpy(buf + *len, text, n + 1);
    *len += n;
    return 0;
}

/* Appends the probe statements as interpreter code (`p + p`, ...) or, with
 * `as_calls`, as calls from C (`o.add(p, p)`, ...), one per line, in the order
 * of N_PROBES. */
static int probe_source(char* buf, size_t size, size_t* len, const char* indent, int as_calls) {
    static const char* const forms[3][2] = {
        {"p %s p", "o.%s(p, p)"}, {"p %s 1.0", "o.%s(p, 1.0)"}, {"1.0 %s p", "o.%s(1.0, p)"}};
    static const char* const unary[3][2] = {
        {"-p", "o.neg(p)"}, {"+p", "o.pos(p)"}, {"~p", "o.invert(p)"}};
    char line[64];
    for (int f = 0; f < 3; ++f) {
        for (int k = 0; k < N_PROBE_BINOPS; ++k) {
            snprintf(line, sizeof line, forms[f][as_calls],
                     as_calls ? probe_opfuncs[k] : probe_binops[k]);
            if (append(buf, size, len, indent) || append(buf, size, len, line) ||
                append(buf, size, len, "\n")) {
                return -1;
            }
        }
    }
    for (int k = 0; k < 3; ++k) {
        if (append(buf, size, len, indent) || append(buf, size, len, unary[k][as_calls]) ||
            append(buf, size, len, "\n")) {
            return -1;
        }
    }
    return 0;
}

static void calibrate_operator_chains(void) {
    g_chains_n = 0;
    call_chain run1[MAX_CHAINS], run2[MAX_CHAINS], run3[MAX_CHAINS];
    int n1 = 0, n2 = 0, n3 = 0;
    /* module code, then the same in a function; the operator module's calls */
    char code[4096], calls[2048];
    size_t code_len = 0, calls_len = 0;
    if (probe_source(code, sizeof code, &code_len, "", 0) ||
        append(code, sizeof code, &code_len, "def f(p):\n") ||
        probe_source(code, sizeof code, &code_len, "    ", 0) ||
        append(calls, sizeof calls, &calls_len, "import operator as o\n") ||
        probe_source(calls, sizeof calls, &calls_len, "", 1)) {
        return; /* never elide */
    }
    PyObject* globals = PyDict_New();
    PyObject* probe =
        PyType_Ready(&ProbeType) == 0 ? PyObject_CallNoArgs((PyObject*)&ProbeType) : NULL;
    if (globals != NULL && probe != NULL && PyDict_SetItemString(globals, "p", probe) == 0 &&
        PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins()) == 0) {
        g_recording = run1; /* entered through module code (PyEval_EvalCode) */
        g_recorded = 0;
        PyObject* r = PyRun_String(code, Py_file_input, globals, globals);
        n1 = g_recorded;
        Py_XDECREF(r);
        PyObject* f = PyDict_GetItemString(globals, "f"); /* borrowed */
        if (f != NULL) {
            g_recording = run2; /* entered through a call from C */
            g_recorded = 0;
            r = PyObject_CallOneArg(f, probe);
            n2 = g_recorded;
            Py_XDECREF(r);
        }
        g_recording = run3; /* called from C: the operator module */
        g_recorded = 0;
        r = PyRun_String(calls, Py_file_input, globals, globals);
        n3 = g_recorded;
        Py_XDECREF(r);
        g_recording = NULL;
    }
    PyErr_Clear();
    Py_XDECREF(probe);
    Py_XDECREF(globals);
    if (n1 != N_PROBES || n2 != N_PROBES || n3 != N_PROBES) {
        return; /* not the expected probes: never elide */
    }
    for (int k = 0; k < n1; ++k) {
        call_chain c;
        c.n = 0;
        while (c.n < run1[k].n && c.n < run2[k].n && run1[k].f[c.n] == run2[k].f[c.n]) {
            c.f[c.n] = run1[k].f[c.n];
            ++c.n;
        }
        if (c.n == 0 || c.n == run1[k].n || c.n == run2[k].n) {
            continue; /* no common frame, or no divergence seen: never elide on it */
        }
        int reaches_interpreter = 1;
        for (int j = 0; j < n3; ++j) {
            if (chain_starts_with(&run3[j], &c)) {
                reaches_interpreter = 0; /* a C caller would match it */
            }
        }
        if (reaches_interpreter) {
            g_chains[g_chains_n++] = c;
        }
    }
}

/* 1 if the operator was applied by the bytecode interpreter itself (not by C
 * code holding a possibly borrowed reference). */
static int called_from_interpreter(void) {
    if (g_chains_n < 0) {
        calibrate_operator_chains();
    }
    const call_chain c = operator_call_chain();
    for (int i = 0; i < g_chains_n; ++i) {
        if (chain_starts_with(&c, &g_chains[i])) {
            return 1;
        }
    }
    return 0;
}

static PyObject* m_elision_chains(PyObject* self, PyObject* noargs) {
    (void)self;
    (void)noargs;
    if (g_chains_n < 0) {
        calibrate_operator_chains();
    }
    PyObject* r = PyList_New(g_chains_n);
    for (int i = 0; r != NULL && i < g_chains_n; ++i) {
        PyList_SET_ITEM(r, i, PyLong_FromLong(g_chains[i].n));
    }
    return r;
}
#endif

/* 1 if `obj` (an operand of the operator being applied) is referenced only
 * by the interpreter's value stack, so its buffer may take the result.
 * `nbytes` is the size of the buffer that would be reused. */
static int is_temporary(PyObject* obj, size_t nbytes) {
#if defined(VP_ELIDE_EXACT)
    (void)nbytes;
    return PyUnstable_Object_IsUniqueReferencedTemporary(obj);
#elif defined(VP_ELIDE_BACKTRACE)
    return Py_REFCNT(obj) == 1 && nbytes >= VP_ELIDE_MIN_BYTES && called_from_interpreter();
#else
    (void)obj;
    (void)nbytes;
    return 0; /* no safe test on this platform / Python version */
#endif
}

/* C-contiguous (extent-1 dimensions may have any stride). */
static int c_contiguous(const vp_array* a) {
    const int nd = vp_array_ndim(a);
    const int64_t* shape = vp_array_shape(a);
    const int64_t* strides = vp_array_strides(a);
    static const int64_t itemsizes[] = {1, 4, 8, 4, 8, 8, 16}; /* vp_dtype order */
    int64_t step = itemsizes[vp_array_dtype(a)];
    for (int i = nd; i-- > 0;) {
        if (shape[i] == 0) {
            return 1;
        }
        if (shape[i] != 1) {
            if (strides[i] != step) {
                return 0;
            }
            step *= shape[i];
        }
    }
    return 1;
}

/* An input of an operator whose buffer can take the result: same dtype and
 * shape as the output, reusable per the runtime, and a temporary. The other
 * array inputs must be C-contiguous too (any shape): then NumPy would
 * allocate its result in C order, the temporary's layout, and its iterator
 * calls the loop the same way with either output (strided or mixed-order
 * operands make the iteration depend on the output's layout). */
static PyObject* elision_candidate(PyObject* const* inputs, vp_array* const* ins, Py_ssize_t n,
                                   int out_code, int ndim, const int64_t* shape) {
#if defined(VP_ELIDE_BACKTRACE)
    /* Below the bound nothing is elided (is_temporary): skip the checks. */
    static const int64_t itemsizes[] = {1, 4, 8, 4, 8, 8, 16}; /* vp_dtype order */
    int64_t nbytes = itemsizes[out_code];
    for (int i = 0; i < ndim; ++i) {
        nbytes *= shape[i];
    }
    if (nbytes < VP_ELIDE_MIN_BYTES) {
        return NULL;
    }
#endif
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (is_array(inputs[i]) && !c_contiguous(ins[i])) {
            return NULL;
        }
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (!is_array(inputs[i]) || ((ArrayObject*)inputs[i])->readonly ||
            (int)vp_array_dtype(ins[i]) != out_code || vp_array_ndim(ins[i]) != ndim ||
            memcmp(vp_array_shape(ins[i]), shape, (size_t)ndim * sizeof(int64_t)) != 0) {
            continue;
        }
        const size_t nbytes = vp_array_reusable_bytes(ins[i]);
        if (nbytes != 0 && is_temporary(inputs[i], nbytes)) {
            return inputs[i];
        }
    }
    return NULL;
}

/* --- scalar shortcut ------------------------------------------------------ */

/* An op on scalars only (ready 0-d viproc arrays, Python and NumPy scalars)
 * has nothing to run asynchronously: NumPy computes it right here, on NumPy
 * scalars, as the NumPy program would (its 0-d values are NumPy scalars:
 * reductions and scalar indexing return them, and ufuncs on 0-d arrays do
 * too). Its result is NumPy's: a NumPy scalar. Any n-d operand (even of one
 * element) keeps the array path, so NumPy's output shapes stay as they
 * are. Fills `out` (capacity `out_len` entries, at least `n`) with new
 * references and returns 1, or 0 if an operand is not a ready scalar. Call it
 * through SCALARS_OF, which passes the length of the `out` array. */
static int scalars_of(PyObject* const* inputs, Py_ssize_t n, PyObject** out, Py_ssize_t out_len) {
    if (n > MAXIN) {
        return 0;
    }
    assert(n >= 0 && n <= out_len);
    (void)out_len;
    if (g_rt == NULL) {
        return 0;
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* x = inputs[i];
        if (is_array(x)) {
            vp_array* a = ((ArrayObject*)x)->a;
            if (vp_array_ndim(a) != 0 || !vp_array_ready(a)) {
                return 0;
            }
        } else if (!(PyFloat_CheckExact(x) || PyLong_CheckExact(x) || PyBool_Check(x) ||
                     PyComplex_CheckExact(x) ||
                     PyObject_TypeCheck(x, (PyTypeObject*)g_np_generic))) {
            return 0;
        }
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* x = inputs[i];
        out[i] = is_array(x) ? vp_array_scalar(((ArrayObject*)x)->a) : Py_NewRef(x);
        if (out[i] == NULL) {
            for (Py_ssize_t j = 0; j < i; ++j) {
                Py_DECREF(out[j]);
            }
            return -1;
        }
    }
    return 1;
}

/* `out` must be an array (not a pointer): its length is checked against n. */
#define SCALARS_OF(inputs, n, out)                                                                 \
    scalars_of((inputs), (n), (out), (Py_ssize_t)(sizeof(out) / sizeof((out)[0])))

static void release_all(PyObject** objs, Py_ssize_t n) {
    for (Py_ssize_t i = 0; i < n; ++i) {
        Py_DECREF(objs[i]);
    }
}

/* --- the ufunc fast path -------------------------------------------------- */

/* Input kinds in the dtype-resolution cache key. Arrays and NumPy scalars use
 * their vp_dtype code (0..6): NumPy treats both as typed ("strong"). Python
 * scalars are weak (NEP 50) and resolved by type, not value. */
enum { K_PYBOOL = 10, K_PYINT = 11, K_PYFLOAT = 12, K_PYCOMPLEX = 13 };

static int dtype_code(PyObject* dtype) {
    for (int c = 0; c < 7; ++c) {
        if (dtype == g_dtypes[c]) {
            return c;
        }
    }
    for (int c = 0; c < 7; ++c) {
        int eq = PyObject_RichCompareBool(dtype, g_dtypes[c], Py_EQ);
        if (eq < 0) {
            PyErr_Clear();
            return -1;
        }
        if (eq) {
            return c;
        }
    }
    return -1;
}

/* Kind of an input, or -1 if the fast path cannot take it. */
static int input_kind(PyObject* x) {
    if (is_array(x)) {
        return (int)vp_array_dtype(((ArrayObject*)x)->a);
    }
    if (PyBool_Check(x)) {
        return K_PYBOOL;
    }
    if (PyLong_CheckExact(x)) {
        return K_PYINT;
    }
    if (PyFloat_CheckExact(x)) {
        return K_PYFLOAT;
    }
    if (PyComplex_CheckExact(x)) {
        return K_PYCOMPLEX;
    }
    if (PyObject_TypeCheck(x, (PyTypeObject*)g_np_generic) ||
        PyObject_TypeCheck(x, (PyTypeObject*)g_np_ndarray)) {
        PyObject* dt = PyObject_GetAttrString(x, "dtype");
        if (dt == NULL) {
            PyErr_Clear();
            return -1;
        }
        int c = dtype_code(dt);
        Py_DECREF(dt);
        return c;
    }
    return -1;
}

static PyObject* kind_spec(int kind) {
    switch (kind) {
    case K_PYBOOL:
        return (PyObject*)&PyBool_Type;
    case K_PYINT:
        return (PyObject*)&PyLong_Type;
    case K_PYFLOAT:
        return (PyObject*)&PyFloat_Type;
    case K_PYCOMPLEX:
        return (PyObject*)&PyComplex_Type;
    default:
        return g_dtypes[kind];
    }
}

/* Resolves (ufunc, kinds) on a g_sigs miss: *packed gets the dtype codes
 * (inputs, then the output; 8 bits each) or -1 if the fast path does not
 * handle the call, *loop NumPy's loop for them. Returns 0, or -1 on error. */
static int resolve_slow(PyObject* ufunc, int nin, const int* kinds, long long* packed,
                        const vp_loop** loop) {
    *packed = -1;
    *loop = NULL;
    PyObject* a_nin = PyObject_GetAttrString(ufunc, "nin");
    PyObject* a_nout = PyObject_GetAttrString(ufunc, "nout");
    PyObject* a_sig = PyObject_GetAttrString(ufunc, "signature");
    const int simple = a_nin != NULL && a_nout != NULL && a_sig == Py_None &&
                       PyLong_AsLong(a_nin) == nin && PyLong_AsLong(a_nout) == 1;
    Py_XDECREF(a_nin);
    Py_XDECREF(a_nout);
    Py_XDECREF(a_sig);
    PyErr_Clear();
    if (!simple) {
        return 0; /* gufunc, several outputs, or wrong argument count: fallback */
    }
    PyObject* spec = PyTuple_New(nin + 1);
    if (spec == NULL) {
        return -1;
    }
    for (int i = 0; i < nin; ++i) {
        PyTuple_SET_ITEM(spec, i, Py_NewRef(kind_spec(kinds[i])));
    }
    PyTuple_SET_ITEM(spec, nin, Py_NewRef(Py_None));
    PyObject* name = PyUnicode_InternFromString("resolve_dtypes");
    PyObject* res = name != NULL ? PyObject_CallMethodOneArg(ufunc, name, spec) : NULL;
    Py_XDECREF(name);
    Py_DECREF(spec);
    if (res == NULL) {
        PyErr_Clear(); /* no loop / promotion error: NumPy raises it in the fallback */
        return 0;
    }
    long long p = 0;
    int ok = 1;
    for (int i = 0; i <= nin && ok; ++i) {
        int c = dtype_code(PyTuple_GET_ITEM(res, i));
        /* Typed inputs of another dtype are cast as NumPy casts them
         * (vp_ufunc_issue); every dtype must be a viproc dtype. */
        ok = c >= 0;
        p |= (long long)c << (8 * i);
    }
    Py_DECREF(res);
    if (!ok) {
        return 0;
    }
    vp_dtype dts[MAXIN + 1];
    for (int i = 0; i <= nin; ++i) {
        dts[i] = (vp_dtype)((p >> (8 * i)) & 0xff);
    }
    char err[ERRLEN];
    if (vp_ufunc_loop(g_rt, ufunc, nin, dts, loop, err, sizeof err) != VP_OK) {
        *loop = NULL; /* no exact NumPy loop (or it needs the Python API) */
        return 0;
    }
    *packed = p;
    return 0;
}

/* NumPy broadcasting of `shape` with (nd, d). Returns 0, or -1 if they do
 * not broadcast. */
static int broadcast_into(int64_t* shape, int* ndim, int nd, const int64_t* d) {
    if (nd > *ndim) {
        memmove(shape + (nd - *ndim), shape, (size_t)*ndim * sizeof(int64_t));
        for (int i = 0; i < nd - *ndim; ++i) {
            shape[i] = 1;
        }
        *ndim = nd;
    }
    for (int i = 0; i < nd; ++i) {
        int64_t* s = &shape[*ndim - nd + i];
        if (*s == 1) {
            *s = d[i];
        } else if (d[i] != 1 && d[i] != *s) {
            return -1;
        }
    }
    return 0;
}

/* np.matmul(a, b) of two viproc arrays (vp_matmul). Returns 1 and sets
 * *result (new reference), 0 if NumPy has to run it (fallback), -1 on a
 * Python error. A 0-d result (vector @ vector) is NumPy's scalar if it is
 * there already, else a pending 0-d array marked as a scalar. */
static int fast_matmul(PyObject* ufunc, PyObject* a, PyObject* b, PyObject** result) {
    vp_array* r = NULL;
    char err[ERRLEN] = "";
    int st = vp_matmul(g_rt, ufunc, ((ArrayObject*)a)->a, ((ArrayObject*)b)->a, &r, issue_site(),
                       err, sizeof err);
    if (st == VP_UNSUPPORTED) {
        return 0;
    }
    if (raise_status(st, err) < 0) {
        return -1;
    }
    if (vp_array_ndim(r) == 0 && vp_array_ready(r) &&
        vp_array_wait(g_rt, r, err, sizeof err) == VP_OK) {
        *result = vp_array_scalar(r);
        vp_array_release(r);
        return *result != NULL ? 1 : -1;
    }
    *result = wrap_array(r);
    if (*result == NULL) {
        return -1;
    }
    if (vp_array_ndim(r) == 0) {
        ((ArrayObject*)*result)->scalar = 1;
    }
    return 1;
}

/* Issues ufunc(*inputs, out=out) asynchronously. Returns 1 and sets *result
 * (new reference), 0 if the fast path does not handle the call, -1 on a
 * Python error. `out` is NULL or a viproc array. */
static int fast_ufunc(PyObject* ufunc, PyObject* const* inputs, Py_ssize_t n, PyObject* out,
                      int elide, PyObject** result) {
    if (g_rt == NULL || g_array_cls == NULL || n > MAXIN ||
        PyThread_get_thread_ident() != g_main_thread) {
        return 0;
    }
    if (check_async_errors() < 0) {
        return -1;
    }
    if (ufunc == g_matmul) {
        return n == 2 && out == NULL && is_array(inputs[0]) && is_array(inputs[1])
                   ? fast_matmul(ufunc, inputs[0], inputs[1], result)
                   : 0;
    }
    int kinds[MAXIN];
    long long key = (long long)n << 24;
    for (Py_ssize_t i = 0; i < n; ++i) {
        if ((kinds[i] = input_kind(inputs[i])) < 0) {
            return 0;
        }
        key |= (long long)kinds[i] << (8 * i);
    }
    long long packed;
    const vp_loop* loop = NULL;
    const Entry* hit = t_find(&g_sigs, ufunc, key);
    if (hit != NULL) {
        packed = hit->v;
        loop = (const vp_loop*)hit->p;
    } else {
        int r = resolve_slow(ufunc, (int)n, kinds, &packed, &loop);
        if (r < 0 || t_put(&g_sigs, ufunc, key, packed, loop, ufunc) < 0) {
            return -1;
        }
    }
    if (packed < 0) {
        return 0;
    }
    int codes[MAXIN + 1];
    for (Py_ssize_t i = 0; i <= n; ++i) {
        codes[i] = (int)((packed >> (8 * i)) & 0xff);
    }

    /* Inputs as runtime arrays; scalars and NumPy arrays are converted. */
    vp_array* ins[MAXIN];
    vp_array* temps[MAXIN];
    PyObject* keep[MAXIN];
    int ntemps = 0, nkeep = 0, status = 0;
    int64_t shape[MAXDIMS];
    int ndim = 0;
    for (Py_ssize_t i = 0; i < n && status == 0; ++i) {
        PyObject* x = inputs[i];
        if (is_array(x)) {
            ins[i] = ((ArrayObject*)x)->a;
        } else if (PyObject_TypeCheck(x, (PyTypeObject*)g_np_ndarray)) {
            PyObject* w = PyObject_CallOneArg(g_asarray, x); /* copy into the runtime */
            if (w == NULL) {
                status = -1;
                break;
            }
            keep[nkeep++] = w;
            ins[i] = ((ArrayObject*)w)->a;
        } else {
            vp_array* t = NULL;
            if (vp_array_from_scalar(g_rt, x, (vp_dtype)codes[i], &t) != VP_OK) {
                status = -1; /* e.g. OverflowError, as in NumPy */
                break;
            }
            temps[ntemps++] = t;
            ins[i] = t;
        }
        if (broadcast_into(shape, &ndim, vp_array_ndim(ins[i]), vp_array_shape(ins[i])) < 0) {
            status = 1; /* not broadcastable: the fallback raises NumPy's error */
        }
    }
    if (status == 0 && out != NULL) {
        vp_array* o = ((ArrayObject*)out)->a;
        int same = vp_array_ndim(o) == ndim && (int)vp_array_dtype(o) == codes[n] &&
                   memcmp(vp_array_shape(o), shape, (size_t)ndim * sizeof(int64_t)) == 0;
        if (!same || ((ArrayObject*)out)->readonly) {
            status = 1; /* casting/broadcasting into out=, or read-only: NumPy decides */
        }
    }
    if (status == 0 && elide && out == NULL) {
        /* Operators only (as NumPy): write into a temporary operand. */
        out = elision_candidate(inputs, ins, n, codes[n], ndim, shape);
        if (out != NULL) {
            ++g_elided;
        }
    }
    int handled = 0;
    if (status == 0) {
        char err[ERRLEN] = "";
        vp_array* res = NULL;
        int st = vp_ufunc_issue(g_rt, loop, ins, ndim, shape,
                                out != NULL ? ((ArrayObject*)out)->a : NULL, &res, issue_site(),
                                err, sizeof err);
        if (st == VP_OK) {
            *result = out != NULL ? Py_NewRef(out) : wrap_array(res);
            if (out == NULL && *result != NULL && ndim == 0) {
                ((ArrayObject*)*result)->scalar = 1; /* NumPy returns a scalar */
            }
            handled = *result != NULL ? 1 : -1;
        } else if (st != VP_UNSUPPORTED) {
            raise_status(st, err);
            handled = -1;
        }
    }
    for (int i = 0; i < ntemps; ++i) {
        vp_array_release(temps[i]); /* the issued task keeps the data alive */
    }
    for (int i = 0; i < nkeep; ++i) {
        Py_DECREF(keep[i]);
    }
    return status < 0 ? -1 : handled;
}

/* _Array.__array_ufunc__(self, ufunc, method, *inputs, **kwargs) */
static PyObject* Array_array_ufunc(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                                   PyObject* kwnames) {
    (void)self;
    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "__array_ufunc__ needs ufunc and method");
        return NULL;
    }
    const Py_ssize_t nkw = kwnames != NULL ? PyTuple_GET_SIZE(kwnames) : 0;
    PyObject* const* inputs = args + 2;
    const Py_ssize_t n = nargs - 2;
    if (PyUnicode_Check(args[1]) && PyUnicode_Compare(args[1], s_call) == 0 &&
        (nkw == 0 || (nkw == 1 && PyUnicode_Compare(PyTuple_GET_ITEM(kwnames, 0), s_out) == 0))) {
        PyObject* out = NULL;
        if (nkw == 1) {
            PyObject* o = args[nargs];
            if (PyTuple_Check(o) && PyTuple_GET_SIZE(o) == 1 && is_array(PyTuple_GET_ITEM(o, 0))) {
                out = PyTuple_GET_ITEM(o, 0);
            } else {
                goto fallback;
            }
        }
        if (out == NULL) {
            PyObject* sc[MAXIN];
            const int k = SCALARS_OF(inputs, n, sc);
            if (k < 0) {
                return NULL;
            }
            if (k > 0) {
                PyObject* r = PyObject_Vectorcall(args[0], sc, (size_t)n, NULL);
                release_all(sc, n);
                return r;
            }
        }
        PyObject* result = NULL;
        int r = fast_ufunc(args[0], inputs, n, out, 0, &result);
        if (r != 0) {
            return r > 0 ? result : NULL;
        }
    }
fallback:;
    PyObject* in = PyTuple_New(n);
    PyObject* kw = PyDict_New();
    if (in == NULL || kw == NULL) {
        Py_XDECREF(in);
        Py_XDECREF(kw);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyTuple_SET_ITEM(in, i, Py_NewRef(inputs[i]));
    }
    for (Py_ssize_t i = 0; i < nkw; ++i) {
        PyDict_SetItem(kw, PyTuple_GET_ITEM(kwnames, i), args[nargs + i]);
    }
    PyObject* r = PyObject_CallFunctionObjArgs(g_fallback, args[0], args[1], in, kw, NULL);
    Py_DECREF(in);
    Py_DECREF(kw);
    return r;
}

/* --- operators ------------------------------------------------------------ */

/* Operator ufuncs by name, cached by the address of the name literal (one
 * per operator): a dict lookup by string built a str object per op. Reset
 * by setup(); entries borrow from g_ops. */
#define MAX_OPS 64
static struct {
    const char* name;
    PyObject* ufunc;
} g_op_cache[MAX_OPS];
static int g_op_cache_n = 0;

static PyObject* op_ufunc(const char* name) {
    for (int i = 0; i < g_op_cache_n; ++i) {
        if (g_op_cache[i].name == name) {
            return g_op_cache[i].ufunc;
        }
    }
    PyObject* u = g_ops != NULL ? PyDict_GetItemString(g_ops, name) : NULL;
    if (u == NULL) {
        PyErr_Format(PyExc_RuntimeError, "viproc: ufunc for %s not registered", name);
        return NULL;
    }
    if (g_op_cache_n < MAX_OPS) {
        g_op_cache[g_op_cache_n].name = name;
        g_op_cache[g_op_cache_n++].ufunc = u;
    }
    return u; /* borrowed */
}

/* True if `other` opts out of NumPy's ufunc protocol (__array_ufunc__ = None),
 * like NDArrayOperatorsMixin checks. */
static int defers(PyObject* other) {
    if (is_array(other) || PyLong_CheckExact(other) || PyFloat_CheckExact(other) ||
        PyComplex_CheckExact(other) || PyBool_Check(other)) {
        return 0;
    }
    PyObject* au = PyObject_GetAttrString((PyObject*)Py_TYPE(other), "__array_ufunc__");
    if (au == NULL) {
        PyErr_Clear();
        return 0;
    }
    int r = au == Py_None;
    Py_DECREF(au);
    return r;
}

static PyObject* binop(const char* name, binaryfunc scalar_op, PyObject* a, PyObject* b) {
    if (defers(is_array(a) ? b : a)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    PyObject* sc[2];
    PyObject* in2[2] = {a, b};
    const int k = scalar_op != NULL ? SCALARS_OF(in2, 2, sc) : 0;
    if (k != 0) {
        PyObject* r = k > 0 ? scalar_op(sc[0], sc[1]) : NULL;
        if (k > 0) {
            release_all(sc, 2);
        }
        return r;
    }
    PyObject* ufunc = op_ufunc(name);
    if (ufunc == NULL) {
        return NULL;
    }
    PyObject* args[2] = {a, b};
    PyObject* result = NULL;
    int r = fast_ufunc(ufunc, args, 2, NULL, 1, &result);
    if (r != 0) {
        return r > 0 ? result : NULL;
    }
    return PyObject_CallFunctionObjArgs(ufunc, a, b, NULL); /* NumPy dispatch -> fallback */
}

static PyObject* inplace_op(const char* name, binaryfunc scalar_op, PyObject* a, PyObject* b) {
    if (defers(b)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    if (((ArrayObject*)a)->scalar) {
        /* A NumPy scalar in the NumPy program: `s += 1` rebinds s. */
        return binop(name, scalar_op, a, b);
    }
    PyObject* ufunc = op_ufunc(name);
    if (ufunc == NULL) {
        return NULL;
    }
    PyObject* args[2] = {a, b};
    PyObject* result = NULL;
    int r = fast_ufunc(ufunc, args, 2, a, 0, &result);
    if (r != 0) {
        return r > 0 ? result : NULL;
    }
    PyObject* call_args = PyTuple_Pack(2, a, b);
    PyObject* kw = Py_BuildValue("{s:(O)}", "out", a);
    PyObject* res = call_args && kw ? PyObject_Call(ufunc, call_args, kw) : NULL;
    Py_XDECREF(call_args);
    Py_XDECREF(kw);
    return res;
}

static PyObject* unop(const char* name, unaryfunc scalar_op, PyObject* a) {
    PyObject* sc[1];
    const int k = SCALARS_OF(&a, 1, sc);
    if (k != 0) {
        PyObject* r = k > 0 ? scalar_op(sc[0]) : NULL;
        Py_XDECREF(k > 0 ? sc[0] : NULL);
        return r;
    }
    PyObject* ufunc = op_ufunc(name);
    if (ufunc == NULL) {
        return NULL;
    }
    PyObject* result = NULL;
    int r = fast_ufunc(ufunc, &a, 1, NULL, 1, &result);
    if (r != 0) {
        return r > 0 ? result : NULL;
    }
    return PyObject_CallOneArg(ufunc, a);
}

#define BINOP(fn, name, scalar_op)                                                                 \
    static PyObject* fn(PyObject* a, PyObject* b) { return binop(name, scalar_op, a, b); }
#define INPLACE(fn, name, scalar_op)                                                               \
    static PyObject* fn(PyObject* a, PyObject* b) { return inplace_op(name, scalar_op, a, b); }
#define UNOP(fn, name, scalar_op)                                                                  \
    static PyObject* fn(PyObject* a) { return unop(name, scalar_op, a); }

BINOP(nb_add, "add", PyNumber_Add)
BINOP(nb_sub, "subtract", PyNumber_Subtract)
BINOP(nb_mul, "multiply", PyNumber_Multiply)
BINOP(nb_truediv, "true_divide", PyNumber_TrueDivide)
BINOP(nb_floordiv, "floor_divide", PyNumber_FloorDivide)
BINOP(nb_rem, "remainder", PyNumber_Remainder)
BINOP(nb_divmod, "divmod", PyNumber_Divmod)
BINOP(nb_lshift, "left_shift", PyNumber_Lshift)
BINOP(nb_rshift, "right_shift", PyNumber_Rshift)
BINOP(nb_and, "bitwise_and", PyNumber_And)
BINOP(nb_or, "bitwise_or", PyNumber_Or)
BINOP(nb_xor, "bitwise_xor", PyNumber_Xor)
BINOP(nb_matmul, "matmul", NULL) /* matmul of 0-d operands raises: NumPy's error */
INPLACE(nb_iadd, "add", PyNumber_Add)
INPLACE(nb_isub, "subtract", PyNumber_Subtract)
INPLACE(nb_imul, "multiply", PyNumber_Multiply)
INPLACE(nb_itruediv, "true_divide", PyNumber_TrueDivide)
INPLACE(nb_ifloordiv, "floor_divide", PyNumber_FloorDivide)
INPLACE(nb_irem, "remainder", PyNumber_Remainder)
INPLACE(nb_ilshift, "left_shift", PyNumber_Lshift)
INPLACE(nb_irshift, "right_shift", PyNumber_Rshift)
INPLACE(nb_iand, "bitwise_and", PyNumber_And)
INPLACE(nb_ior, "bitwise_or", PyNumber_Or)
INPLACE(nb_ixor, "bitwise_xor", PyNumber_Xor)
INPLACE(nb_imatmul, "matmul", NULL)
UNOP(nb_neg, "negative", PyNumber_Negative)
UNOP(nb_pos, "positive", PyNumber_Positive)
UNOP(nb_abs, "absolute", PyNumber_Absolute)
UNOP(nb_invert, "invert", PyNumber_Invert)

static PyObject* scalar_pow(PyObject* a, PyObject* b) { return PyNumber_Power(a, b, Py_None); }

static PyObject* nb_pow(PyObject* a, PyObject* b, PyObject* mod) {
    if (mod != Py_None) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return binop("power", scalar_pow, a, b);
}

static PyObject* nb_ipow(PyObject* a, PyObject* b, PyObject* mod) {
    if (mod != Py_None) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return inplace_op("power", scalar_pow, a, b);
}

#define CMP(fn, op)                                                                                \
    static PyObject* fn(PyObject* a, PyObject* b) { return PyObject_RichCompare(a, b, op); }
CMP(scalar_lt, Py_LT)
CMP(scalar_le, Py_LE)
CMP(scalar_eq, Py_EQ)
CMP(scalar_ne, Py_NE)
CMP(scalar_gt, Py_GT)
CMP(scalar_ge, Py_GE)

static PyObject* Array_richcompare(PyObject* a, PyObject* b, int op) {
    static const char* names[] = {"less",      "less_equal", "equal",
                                  "not_equal", "greater",    "greater_equal"};
    static const binaryfunc cmp[] = {scalar_lt, scalar_le, scalar_eq,
                                     scalar_ne, scalar_gt, scalar_ge};
    return binop(names[op], cmp[op], a, b);
}

static PyNumberMethods Array_as_number = {
    .nb_add = nb_add,
    .nb_subtract = nb_sub,
    .nb_multiply = nb_mul,
    .nb_remainder = nb_rem,
    .nb_divmod = nb_divmod,
    .nb_power = nb_pow,
    .nb_negative = nb_neg,
    .nb_positive = nb_pos,
    .nb_absolute = nb_abs,
    .nb_invert = nb_invert,
    .nb_lshift = nb_lshift,
    .nb_rshift = nb_rshift,
    .nb_and = nb_and,
    .nb_xor = nb_xor,
    .nb_or = nb_or,
    .nb_inplace_add = nb_iadd,
    .nb_inplace_subtract = nb_isub,
    .nb_inplace_multiply = nb_imul,
    .nb_inplace_remainder = nb_irem,
    .nb_inplace_power = nb_ipow,
    .nb_inplace_lshift = nb_ilshift,
    .nb_inplace_rshift = nb_irshift,
    .nb_inplace_and = nb_iand,
    .nb_inplace_xor = nb_ixor,
    .nb_inplace_or = nb_ior,
    .nb_floor_divide = nb_floordiv,
    .nb_true_divide = nb_truediv,
    .nb_inplace_floor_divide = nb_ifloordiv,
    .nb_inplace_true_divide = nb_itruediv,
    .nb_matrix_multiply = nb_matmul,
    .nb_inplace_matrix_multiply = nb_imatmul,
};

/* Reductions and other methods with a C entry (see "reductions in C"). */
// clang-format off
#define RED_METHODS(X) \
    X(sum) X(prod) X(max) X(min) X(any) X(all) X(mean) X(var) X(std) X(argmax) X(argmin) \
    X(copy) X(astype)
// clang-format on
#define X(name)                                                                                    \
    static PyObject* Array_##name(PyObject* self, PyObject* const* args, Py_ssize_t nargs,         \
                                  PyObject* kwnames);
RED_METHODS(X)
#undef X
static PyObject* Array_array_function(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
static PyObject* Array_reshape(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                               PyObject* kwnames);
static PyObject* Array_swapaxes(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                                PyObject* kwnames);

static PyMethodDef Array_methods[] = {
    {"__array_ufunc__", (PyCFunction)(void (*)(void))Array_array_ufunc,
     METH_FASTCALL | METH_KEYWORDS, "NEP 13: element-wise ufuncs run asynchronously."},
    {"__array_function__", (PyCFunction)(void (*)(void))Array_array_function, METH_FASTCALL,
     "NEP 18: reductions in C, everything else in _ndarray.py."},
#define X(name)                                                                                    \
    {#name, (PyCFunction)(void (*)(void))Array_##name, METH_FASTCALL | METH_KEYWORDS,              \
     "As NumPy's ndarray." #name "()."},
    RED_METHODS(X)
#undef X
    // clang-format off
    {"reshape", (PyCFunction)(void (*)(void))Array_reshape, METH_FASTCALL | METH_KEYWORDS,
     "As NumPy's ndarray.reshape(): a view in C, copies in _ndarray.py."},
    {"swapaxes", (PyCFunction)(void (*)(void))Array_swapaxes, METH_FASTCALL | METH_KEYWORDS,
     "As NumPy's ndarray.swapaxes()."},
    // clang-format on
    {NULL, NULL, 0, NULL},
};

static PyObject* m_item(PyObject* self, PyObject* const* args, Py_ssize_t nargs);
static PyObject* m_basic_view(PyObject* self, PyObject* const* args, Py_ssize_t nargs);

/* a[key]. In C for what runs once per op, without a Python frame: an
 * integer key of a ready array (NumPy's scalar) and basic keys (a view).
 * Everything else (advanced indexing, a pending a[i, j], errors) goes to the
 * Python hook, which keeps NumPy's semantics and messages. */
static PyObject* Array_subscript(PyObject* self, PyObject* key) {
    PyObject* args[2] = {self, key};
    PyObject* r = m_item(NULL, args, 2);
    if (r != Py_None) {
        return r; /* NumPy's scalar, or NULL */
    }
    Py_DECREF(r);
    r = m_basic_view(NULL, args, 2);
    if (r == NULL) {
        return NULL;
    }
    if (r != Py_None) {
        if (vp_array_ndim(((ArrayObject*)r)->a) > 0) {
            return r;
        }
        Py_ssize_t n = PyTuple_Check(key) ? PyTuple_GET_SIZE(key) : 1;
        for (Py_ssize_t i = 0; i < n; ++i) {
            if ((PyTuple_Check(key) ? PyTuple_GET_ITEM(key, i) : key) == Py_Ellipsis) {
                return r; /* a 0-d view, as NumPy's a[..., i] */
            }
        }
        /* NumPy returns a scalar: the hook decides (pending: a 0-d copy) */
    }
    Py_DECREF(r);
    if (g_getitem == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "viproc: setup() was not called");
        return NULL;
    }
    return PyObject_CallFunctionObjArgs(g_getitem, self, key, NULL);
}

static int64_t array_size(vp_array* a);

/* Viproc arrays in an index key (the key or the items of a key tuple):
 * *found counts them, *pending those not ready. */
static void key_arrays(PyObject* key, int* found, int* pending) {
    PyObject* const* items = PyTuple_Check(key) ? &PyTuple_GET_ITEM(key, 0) : &key;
    const Py_ssize_t n = PyTuple_Check(key) ? PyTuple_GET_SIZE(key) : 1;
    *found = *pending = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (is_array(items[i])) {
            ++*found;
            *pending += !vp_array_ready(((ArrayObject*)items[i])->a);
        }
    }
}

/* Ends a borrow (vp_array_borrow): a view NumPy kept a reference to becomes a
 * read-only export. */
static int end_borrow(vp_array* a, PyObject* view) {
    int r = 0;
    if (Py_REFCNT(view) > 1 && vp_array_pin(a, view) != VP_OK) {
        r = -1;
    }
    Py_DECREF(view);
    return r;
}

/* The key with its (ready) viproc arrays replaced by borrowed NumPy views;
 * the views are appended to `borrowed` (pairs of array object and view). */
static PyObject* borrow_key(PyObject* key, PyObject* borrowed) {
    if (is_array(key)) {
        PyObject* v = vp_array_borrow(((ArrayObject*)key)->a);
        if (v == NULL || PyList_Append(borrowed, key) < 0 || PyList_Append(borrowed, v) < 0) {
            Py_XDECREF(v);
            return NULL;
        }
        return v; /* the list holds one reference, the caller this one */
    }
    if (!PyTuple_Check(key)) {
        return Py_NewRef(key);
    }
    const Py_ssize_t n = PyTuple_GET_SIZE(key);
    PyObject* t = PyTuple_New(n);
    for (Py_ssize_t i = 0; t != NULL && i < n; ++i) {
        PyObject* k = borrow_key(PyTuple_GET_ITEM(key, i), borrowed);
        if (k == NULL) {
            Py_CLEAR(t);
            break;
        }
        PyTuple_SET_ITEM(t, i, k);
    }
    return t;
}

/* a[key] = value by NumPy itself: its mapping slot (array_assign_subscript)
 * called on a borrowed, writable view of a's buffer, viproc arrays in the key
 * and the value as borrowed views too. Only when NumPy may write the buffer
 * right now (vp_array_host_writable) and every viproc array involved is
 * ready: then no task touches the memory during the call, and the view never
 * reaches user code (design doc D7). Values, casts, errors: NumPy's. */
static int numpy_setitem(PyObject* self, PyObject* key, PyObject* value, int nkey) {
    vp_array* x = ((ArrayObject*)self)->a;
    PyObject* borrowed = PyList_New(0);
    PyObject* view = borrowed != NULL ? vp_array_borrow(x) : NULL;
    PyObject* k = view == NULL ? NULL : nkey ? borrow_key(key, borrowed) : Py_NewRef(key);
    PyObject* v = NULL;
    if (k != NULL) {
        v = is_array(value) ? vp_array_borrow(((ArrayObject*)value)->a) : Py_NewRef(value);
    }
    int r = -1;
    if (v != NULL) {
        r = ((PyTypeObject*)g_np_ndarray)->tp_as_mapping->mp_ass_subscript(view, k, v);
    }
    if (v != NULL && is_array(value) && end_borrow(((ArrayObject*)value)->a, v) < 0) {
        r = -1;
    } else if (v != NULL && !is_array(value)) {
        Py_DECREF(v);
    }
    Py_XDECREF(k);
    for (Py_ssize_t i = 0; borrowed != NULL && i + 1 < PyList_GET_SIZE(borrowed); i += 2) {
        PyObject* bv = PyList_GET_ITEM(borrowed, i + 1); /* the list's reference only */
        if (Py_REFCNT(bv) > 1 &&
            vp_array_pin(((ArrayObject*)PyList_GET_ITEM(borrowed, i))->a, bv) != VP_OK) {
            r = -1;
        }
    }
    Py_XDECREF(borrowed);
    if (view != NULL && end_borrow(x, view) < 0) {
        r = -1;
    }
    return r;
}

/* True for the values NumPy's a[i] = value converts with PyArray_Pack
 * without a sequence: Python and NumPy scalars. */
static int is_scalar_value(PyObject* v) {
    return PyFloat_Check(v) || PyLong_Check(v) || PyComplex_Check(v) ||
           PyObject_TypeCheck(v, (PyTypeObject*)g_np_generic);
}

static int call_setitem_hook(PyObject* self, PyObject* key, PyObject* value) {
    if (g_setitem == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "viproc: setup() was not called");
        return -1;
    }
    PyObject* r = PyObject_CallFunctionObjArgs(g_setitem, self, key, value, NULL);
    Py_XDECREF(r);
    return r == NULL ? -1 : 0;
}

/* a[key] = value. Small work on ready data (every array involved ready, the
 * written part at most sync_below elements, nothing reading a's buffer): NumPy
 * assigns, synchronously, as in a NumPy program. Otherwise basic keys are
 * issued as an assignment: the value is converted now (scalars as NumPy's
 * a[i] = value converts them), the target is a view, the task copies.
 * Everything else (advanced keys, sequences as values, errors of the async
 * path) goes to the Python hook, which keeps NumPy's semantics. */
static int Array_ass_subscript(PyObject* self, PyObject* key, PyObject* value) {
    if (value == NULL) {
        PyErr_SetString(PyExc_ValueError, "cannot delete array elements");
        return -1;
    }
    if (check_runtime() < 0) {
        return -1;
    }
    if (((ArrayObject*)self)->readonly) {
        PyErr_SetString(PyExc_ValueError, "assignment destination is read-only");
        return -1;
    }
    vp_array* x = ((ArrayObject*)self)->a;
    int nkey, kpending;
    key_arrays(key, &nkey, &kpending);
    const int val_array = is_array(value);
    const int all_ready = !kpending && (!val_array || vp_array_ready(((ArrayObject*)value)->a));
    const int writable = all_ready && vp_array_host_writable(x);
    if (writable && array_size(x) <= g_small) {
        return numpy_setitem(self, key, value, nkey);
    }
    PyObject* target = NULL;
    if (nkey == 0) {
        PyObject* args[2] = {self, key};
        target = m_basic_view(NULL, args, 2);
        if (target == NULL) {
            PyErr_Clear(); /* the hook raises NumPy's error */
        } else if (target == Py_None) {
            Py_CLEAR(target);
        }
    }
    if (target == NULL) { /* advanced key: scatter / masked assignment tasks */
        return call_setitem_hook(self, key, value);
    }
    vp_array* t = ((ArrayObject*)target)->a;
    if (writable && array_size(t) <= g_small) {
        Py_DECREF(target);
        return numpy_setitem(self, key, value, nkey);
    }
    PyObject* src = NULL;
    if (val_array) {
        src = Py_NewRef(value); /* another dtype: vp_assign casts */
    } else if (is_scalar_value(value)) {
        vp_array* s = NULL;
        if (vp_array_from_item(g_rt, value, vp_array_dtype(x), &s) != VP_OK) {
            Py_DECREF(target);
            return -1; /* NumPy's conversion error, as a[i] = value raises it */
        }
        src = wrap_array(s);
    }
    if (src == NULL) {
        Py_DECREF(target);
        return PyErr_Occurred() ? -1 : call_setitem_hook(self, key, value);
    }
    char err[ERRLEN] = "";
    const int status = vp_assign(g_rt, t, ((ArrayObject*)src)->a, issue_site(), err, sizeof err);
    Py_DECREF(src);
    Py_DECREF(target);
    if (status == VP_OK) {
        return 0;
    }
    if (status == VP_INVALID || status == VP_UNSUPPORTED) {
        return call_setitem_hook(self, key, value); /* NumPy's error or cast */
    }
    return raise_status(status, err);
}

static PyMappingMethods Array_as_mapping = {
    .mp_subscript = Array_subscript,
    .mp_ass_subscript = Array_ass_subscript,
};

static PyTypeObject ArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "viproc._viproc._Array",
    .tp_basicsize = sizeof(ArrayObject),
    .tp_dealloc = (destructor)Array_dealloc,
    .tp_as_number = &Array_as_number,
    .tp_as_mapping = &Array_as_mapping,
    .tp_richcompare = Array_richcompare,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Base class of viproc.ndarray: owns one runtime array.",
    .tp_methods = Array_methods,
    .tp_getset = Array_getset,
    /* no tp_new: instances are only created by this module */
};

/* --- module functions ----------------------------------------------------- */

static PyObject* m_setup(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *cls, *dtypes, *fallback, *asarray, *dirs, *ops, *ndarray, *generic, *getitem,
        *setitem;
    if (!PyArg_ParseTuple(args, "O!O!OOO!O!OOOO", &PyType_Type, &cls, &PyTuple_Type, &dtypes,
                          &fallback, &asarray, &PyTuple_Type, &dirs, &PyDict_Type, &ops, &ndarray,
                          &generic, &getitem, &setitem)) {
        return NULL;
    }
    if (!PyType_IsSubtype((PyTypeObject*)cls, &ArrayType) || PyTuple_GET_SIZE(dtypes) != 7) {
        PyErr_SetString(PyExc_TypeError, "setup: bad array class or dtype table");
        return NULL;
    }
    Py_XSETREF(g_array_cls, (PyTypeObject*)Py_NewRef(cls));
    for (int c = 0; c < 7; ++c) {
        Py_XSETREF(g_dtypes[c], Py_NewRef(PyTuple_GET_ITEM(dtypes, c)));
    }
    Py_XSETREF(g_fallback, Py_NewRef(fallback));
    Py_XSETREF(g_asarray, Py_NewRef(asarray));
    Py_XSETREF(g_internal_dirs, Py_NewRef(dirs));
    g_op_cache_n = 0; /* borrowed from the old g_ops */
    Py_XSETREF(g_ops, Py_NewRef(ops));
    Py_XSETREF(g_matmul, Py_XNewRef(PyDict_GetItemString(ops, "matmul")));
    Py_XSETREF(g_np_ndarray, Py_NewRef(ndarray));
    Py_XSETREF(g_np_generic, Py_NewRef(generic));
    Py_XSETREF(g_getitem, Py_NewRef(getitem));
    Py_XSETREF(g_setitem, Py_NewRef(setitem));
    PyObject* np = PyImport_ImportModule("numpy");
    Py_XSETREF(g_np_integer, np != NULL ? PyObject_GetAttrString(np, "integer") : NULL);
    Py_XSETREF(g_where_func, np != NULL ? PyObject_GetAttrString(np, "where") : NULL);
    Py_XSETREF(g_where_impl, g_where_func != NULL
                                 ? PyObject_GetAttrString(g_where_func, "_implementation")
                                 : NULL);
    Py_XDECREF(np);
    if (g_np_integer == NULL || g_where_func == NULL || g_where_impl == NULL) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* m_init(PyObject* self, PyObject* args) {
    (void)self;
    Py_ssize_t workers = 0, max_active = 0;
    long long sync_below = -1; /* -1: runtime default */
    int adaptive = 1;
    if (!PyArg_ParseTuple(args, "|nnLp", &workers, &max_active, &sync_below, &adaptive)) {
        return NULL;
    }
    if (g_rt != NULL) {
        PyErr_SetString(PyExc_RuntimeError, "viproc runtime already initialized");
        return NULL;
    }
    char err[ERRLEN] = "";
    g_rt = vp_runtime_create((size_t)workers, (size_t)max_active, err, sizeof err);
    if (g_rt == NULL) {
        PyErr_SetString(PyExc_RuntimeError, err);
        return NULL;
    }
    g_main_thread = PyThread_get_thread_ident();
    vp_set_error_notify(g_rt, error_recorded, NULL);
    g_small = sync_below >= 0 ? sync_below : 2048;
    if (sync_below >= 0 || !adaptive) {
        vp_set_offload_policy(g_rt, sync_below >= 0 ? sync_below : 2048, adaptive);
    }
    Py_RETURN_NONE;
}

static PyObject* m_offload_counts(PyObject* self, PyObject* noargs) {
    (void)self;
    (void)noargs;
    if (check_runtime() < 0) {
        return NULL;
    }
    uint64_t inl = 0, off = 0;
    vp_offload_counts(g_rt, &inl, &off);
    return Py_BuildValue("{s:K,s:K,s:K}", "inline", (unsigned long long)inl, "offloaded",
                         (unsigned long long)off, "direct",
                         (unsigned long long)vp_direct_count(g_rt));
}

static PyObject* m_pool_stats(PyObject* self, PyObject* noargs) {
    (void)self;
    (void)noargs;
    vp_pool_stats s;
    vp_pool_get_stats(&s);
    return Py_BuildValue("{s:K,s:K,s:K,s:K,s:K}", "hits", (unsigned long long)s.hits, "misses",
                         (unsigned long long)s.misses, "cached", (unsigned long long)s.cached,
                         "dropped", (unsigned long long)s.dropped, "limit",
                         (unsigned long long)s.limit);
}

static PyObject* m_pool_trim(PyObject* self, PyObject* noargs) {
    (void)self;
    (void)noargs;
    vp_pool_trim();
    Py_RETURN_NONE;
}

static PyObject* m_pool_set_limit(PyObject* self, PyObject* arg) {
    (void)self;
    unsigned long long v = PyLong_AsUnsignedLongLong(arg);
    if (v == (unsigned long long)-1 && PyErr_Occurred()) {
        return NULL;
    }
    vp_pool_set_limit(v);
    Py_RETURN_NONE;
}

static PyObject* m_elision_mode(PyObject* self, PyObject* noargs) {
    (void)self;
    (void)noargs;
#if defined(VP_ELIDE_EXACT)
    return PyUnicode_FromString("exact");
#elif defined(VP_ELIDE_BACKTRACE)
    return PyUnicode_FromString("backtrace");
#else
    return PyUnicode_FromString("off");
#endif
}

static PyObject* m_elided(PyObject* self, PyObject* noargs) {
    (void)self;
    (void)noargs;
    return PyLong_FromUnsignedLongLong(g_elided);
}

static PyObject* m_shutdown(PyObject* self, PyObject* noargs) {
    (void)self;
    (void)noargs;
    if (g_rt != NULL) {
        /* An error nothing reported yet is raised here, after the runtime is
         * gone (at exit, Python prints it: "Exception ignored in atexit"). */
        Py_BEGIN_ALLOW_THREADS;
        vp_wait_all(g_rt);
        Py_END_ALLOW_THREADS;
        char err[ERRLEN] = "";
        const int failed = vp_take_error(g_rt, err, sizeof err) == VP_ERROR;
        vp_runtime* rt = g_rt;
        g_rt = NULL;
        t_clear(&g_sites); /* interned in the runtime being destroyed */
        t_clear(&g_sigs);  /* holds its loop handles */
        vp_runtime_destroy(rt);
        if (failed) {
            raise_async(err);
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

static PyObject* m_initialized(PyObject* self, PyObject* noargs) {
    (void)self;
    (void)noargs;
    return PyBool_FromLong(g_rt != NULL);
}

static PyObject* m_workers(PyObject* self, PyObject* noargs) {
    (void)self;
    (void)noargs;
    if (check_runtime() < 0) {
        return NULL;
    }
    return PyLong_FromSize_t(vp_worker_count(g_rt));
}

static PyObject* m_active_tasks(PyObject* self, PyObject* noargs) {
    (void)self;
    (void)noargs;
    if (check_runtime() < 0) {
        return NULL;
    }
    return PyLong_FromSize_t(vp_active_tasks(g_rt));
}

static PyObject* m_wait_all(PyObject* self, PyObject* noargs) {
    (void)self;
    (void)noargs;
    if (check_runtime() < 0) {
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS;
    vp_wait_all(g_rt);
    Py_END_ALLOW_THREADS;
    if (raise_pending_error() < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* m_from_ndarray(PyObject* self, PyObject* arr) {
    (void)self;
    if (check_runtime() < 0) {
        return NULL;
    }
    char err[ERRLEN] = "";
    vp_array* a = NULL;
    if (raise_status(vp_array_from_ndarray(g_rt, arr, &a, err, sizeof err), err) < 0) {
        return NULL;
    }
    return wrap_array(a);
}

/* --- views: NumPy's metadata rules, computed here ------------------------- */

/* A view of `a` with this layout (offset relative to a's first element),
 * read-only if `a` is; new reference or NULL. */
static PyObject* make_view(PyObject* a, vp_dtype dtype, int nd, const int64_t* shape,
                           const int64_t* strides, int64_t offset, int readonly) {
    PyObject* v = wrap_array(vp_array_view(g_rt, array_of(a), dtype, nd, shape, strides, offset));
    if (v != NULL) {
        ((ArrayObject*)v)->readonly = ((ArrayObject*)a)->readonly || readonly;
    }
    return v;
}

/* Basic indexing (ints, slices, None, one Ellipsis) as NumPy's
 * get_view_from_index computes it: slices through CPython's
 * PySlice_Unpack/PySlice_AdjustIndices, as NumPy's PySlice_GetIndicesEx.
 * Returns the view, or None for any other key (and for errors, which the
 * caller leaves to NumPy's messages). An all-int key gives a 0-d view. */
static PyObject* m_basic_view(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    if (array_and_arg(args, nargs) < 0) {
        return NULL;
    }
    PyObject *a = args[0], *key = args[1];
    vp_array* x = array_of(a);
    const int nd = vp_array_ndim(x);
    const int64_t* shape = vp_array_shape(x);
    const int64_t* strides = vp_array_strides(x);
    PyObject* const* items = PyTuple_Check(key) ? &PyTuple_GET_ITEM(key, 0) : &key;
    const Py_ssize_t n = PyTuple_Check(key) ? PyTuple_GET_SIZE(key) : 1;
    /* Count the dimensions the key consumes. */
    int consumed = 0, ellipses = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* k = items[i];
        if (k == Py_Ellipsis) {
            ++ellipses;
        } else if (k != Py_None) {
            if (!(PySlice_Check(k) || (PyLong_Check(k) && !PyBool_Check(k)) ||
                  (g_np_integer != NULL && PyObject_TypeCheck(k, (PyTypeObject*)g_np_integer)))) {
                Py_RETURN_NONE; /* arrays, bools, lists, ...: not basic */
            }
            ++consumed;
        }
    }
    if (ellipses > 1 || consumed > nd) {
        Py_RETURN_NONE;
    }
    int64_t nshape[MAXDIMS], nstrides[MAXDIMS];
    int newnd = 0, od = 0;
    int64_t offset = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* k = items[i];
        if (k == Py_None) {
            if (newnd >= MAXDIMS) {
                Py_RETURN_NONE;
            }
            nshape[newnd] = 1;
            nstrides[newnd++] = 0;
        } else if (k == Py_Ellipsis) {
            for (int j = 0; j < nd - consumed; ++j, ++od) {
                if (newnd >= MAXDIMS) {
                    Py_RETURN_NONE;
                }
                nshape[newnd] = shape[od];
                nstrides[newnd++] = strides[od];
            }
        } else if (PySlice_Check(k)) {
            Py_ssize_t start, stop, step;
            if (PySlice_Unpack(k, &start, &stop, &step) < 0) {
                PyErr_Clear();
                Py_RETURN_NONE;
            }
            Py_ssize_t len = PySlice_AdjustIndices((Py_ssize_t)shape[od], &start, &stop, step);
            if (len <= 0) { /* as NumPy: points to the start */
                len = 0;
                step = 1;
                start = 0;
            }
            if (newnd >= MAXDIMS) {
                Py_RETURN_NONE;
            }
            offset += strides[od] * start;
            nshape[newnd] = len;
            nstrides[newnd++] = strides[od] * step;
            ++od;
        } else {
            const Py_ssize_t v = PyNumber_AsSsize_t(k, NULL);
            if (v == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                Py_RETURN_NONE;
            }
            if (v < -shape[od] || v >= shape[od]) {
                Py_RETURN_NONE; /* NumPy raises its IndexError */
            }
            offset += strides[od] * (v < 0 ? v + shape[od] : v);
            ++od;
        }
    }
    for (; od < nd; ++od) { /* implicit trailing Ellipsis */
        if (newnd >= MAXDIMS) {
            Py_RETURN_NONE;
        }
        nshape[newnd] = shape[od];
        nstrides[newnd++] = strides[od];
    }
    return make_view(a, vp_array_dtype(x), newnd, nshape, nstrides, offset, 0);
}

/* a.transpose(axes) (axes: None or a permutation, negative axes allowed) as
 * NumPy's PyArray_Transpose; None for an invalid permutation (NumPy raises). */
static PyObject* m_transpose(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    if (array_and_arg(args, nargs) < 0) {
        return NULL;
    }
    PyObject *a = args[0], *axes = args[1];
    vp_array* x = array_of(a);
    const int nd = vp_array_ndim(x);
    const int64_t* shape = vp_array_shape(x);
    const int64_t* strides = vp_array_strides(x);
    int64_t perm[MAXDIMS];
    if (axes == Py_None) {
        for (int i = 0; i < nd; ++i) {
            perm[i] = nd - 1 - i;
        }
    } else {
        const int n = read_dims(axes, perm);
        if (n < 0) {
            PyErr_Clear();
            Py_RETURN_NONE;
        }
        if (n != nd) {
            Py_RETURN_NONE;
        }
        int seen[MAXDIMS] = {0};
        for (int i = 0; i < nd; ++i) {
            if (perm[i] < 0) {
                perm[i] += nd;
            }
            if (perm[i] < 0 || perm[i] >= nd || seen[perm[i]]++) {
                Py_RETURN_NONE;
            }
        }
    }
    int64_t nshape[MAXDIMS], nstrides[MAXDIMS];
    for (int i = 0; i < nd; ++i) {
        nshape[i] = shape[perm[i]];
        nstrides[i] = strides[perm[i]];
    }
    return make_view(a, vp_array_dtype(x), nd, nshape, nstrides, 0, 0);
}

/* reshape(a, shape, order): NumPy's a.reshape(shape, order=order) ('C',
 * 'F' or 'A') as a view, None if NumPy would copy. One -1 is resolved and
 * size mismatches raise as in NumPy (_fix_unknown_dimension). */
static PyObject* reshape_view(PyObject* a, int64_t* dims, int nd, int order) {
    vp_array* x = array_of(a);
    int64_t size = 1;
    for (int i = 0; i < vp_array_ndim(x); ++i) {
        size *= vp_array_shape(x)[i];
    }
    int unknown = -1;
    int64_t known = 1;
    for (int i = 0; i < nd; ++i) {
        if (dims[i] < 0) {
            if (unknown >= 0) {
                PyErr_SetString(PyExc_ValueError, "can only specify one unknown dimension");
                return NULL;
            }
            unknown = i;
        } else {
            known *= dims[i];
        }
    }
    if (unknown >= 0) {
        if (known == 0 || size % known != 0) {
            known = -1; /* mismatch */
        } else {
            dims[unknown] = size / known;
            known = size;
        }
    }
    if (known != size) { /* NumPy's message (convert_shape_to_string) */
        char text[MAXDIMS * 24 + 4];
        int i = 0, len = 0;
        while (i < nd && dims[i] < 0) {
            ++i; /* leading unknown dimensions are not printed */
        }
        if (i == nd) {
            snprintf(text, sizeof text, "()");
        } else {
            len = snprintf(text, sizeof text, "(%lld", (long long)dims[i++]);
            for (; i < nd; ++i) {
                len += dims[i] < 0 ? snprintf(text + len, sizeof text - (size_t)len, ",newaxis")
                                   : snprintf(text + len, sizeof text - (size_t)len, ",%lld",
                                              (long long)dims[i]);
            }
            snprintf(text + len, sizeof text - (size_t)len, nd == 1 ? ",)" : ")");
        }
        PyErr_Format(PyExc_ValueError, "cannot reshape array of size %lld into shape %s",
                     (long long)size, text);
        return NULL;
    }
    if (order == 'A') { /* F if the array is F- and not C-contiguous */
        const int64_t* shape = vp_array_shape(x);
        const int64_t* strides = vp_array_strides(x);
        const int n = vp_array_ndim(x);
        static const int64_t itemsizes[] = {1, 4, 8, 4, 8, 8, 16}; /* vp_dtype order */
        int64_t step = itemsizes[vp_array_dtype(x)];
        int f = 1;
        for (int i = 0; i < n && f; ++i) {
            if (shape[i] == 0) {
                break;
            }
            if (shape[i] != 1) {
                f = strides[i] == step;
                step *= shape[i];
            }
        }
        order = f && !c_contiguous(x) ? 'F' : 'C';
    }
    if (order != 'C' && order != 'F') {
        Py_RETURN_NONE;
    }
    vp_array* r = vp_array_reshape(g_rt, x, nd, dims, (char)order);
    if (r == NULL) {
        Py_RETURN_NONE;
    }
    PyObject* v = wrap_array(r);
    if (v != NULL) {
        ((ArrayObject*)v)->readonly = ((ArrayObject*)a)->readonly;
    }
    return v;
}

static PyObject* m_reshape(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *a, *shape_o;
    int order;
    if (!PyArg_ParseTuple(args, "O!OC", &ArrayType, &a, &shape_o, &order) || check_runtime() < 0) {
        return NULL;
    }
    int64_t dims[MAXDIMS];
    const int nd = read_dims(shape_o, dims);
    return nd < 0 ? NULL : reshape_view(a, dims, nd, order);
}

/* Exact Python ints `items` as dims; 0 for anything else (left to NumPy's
 * argument handling in Python). */
static int exact_dims(PyObject* const* items, Py_ssize_t n, int64_t* dims) {
    if (n > MAXDIMS) {
        return 0;
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (!PyLong_CheckExact(items[i])) {
            return 0;
        }
        dims[i] = PyLong_AsLongLong(items[i]);
        if (dims[i] == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return 0;
        }
    }
    return 1;
}

/* a.reshape(shape) / a.reshape(*shape) in C order where NumPy makes a view;
 * everything else (order=, copy=, copies, other argument types) in Python. */
static PyObject* Array_reshape(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                               PyObject* kwnames) {
    if (Py_TYPE(self) == g_array_cls && kwnames == NULL && nargs >= 1 && check_runtime() == 0) {
        int64_t dims[MAXDIMS];
        PyObject* const* items = args;
        Py_ssize_t n = nargs;
        if (nargs == 1 && PyTuple_CheckExact(args[0])) {
            items = &PyTuple_GET_ITEM(args[0], 0);
            n = PyTuple_GET_SIZE(args[0]);
        }
        if (exact_dims(items, n, dims)) {
            PyObject* r = reshape_view(self, dims, (int)n, 'C');
            if (r != Py_None) {
                return r; /* the view, or NumPy's size error */
            }
            Py_DECREF(r);
        }
    } else if (PyErr_Occurred()) {
        return NULL;
    }
    static PyObject* name = NULL;
    if (name == NULL && (name = PyUnicode_InternFromString("_reshape_py")) == NULL) {
        return NULL;
    }
    PyObject* stack[16];
    const Py_ssize_t total = nargs + (kwnames != NULL ? PyTuple_GET_SIZE(kwnames) : 0);
    if (total >= 16) {
        PyErr_SetString(PyExc_TypeError, "reshape: too many arguments");
        return NULL;
    }
    stack[0] = self;
    memcpy(stack + 1, args, (size_t)total * sizeof(PyObject*));
    return PyObject_VectorcallMethod(name, stack, (size_t)(nargs + 1), kwnames);
}

/* a.swapaxes(i, j) for exact ints in range; everything else in Python
 * (NumPy's AxisError messages). */
static PyObject* Array_swapaxes(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                                PyObject* kwnames) {
    int64_t ij[2];
    if (Py_TYPE(self) == g_array_cls && kwnames == NULL && nargs == 2 && exact_dims(args, 2, ij) &&
        check_runtime() == 0) {
        vp_array* x = array_of(self);
        const int nd = vp_array_ndim(x);
        int64_t i = ij[0] < 0 ? ij[0] + nd : ij[0], j = ij[1] < 0 ? ij[1] + nd : ij[1];
        if (i >= 0 && i < nd && j >= 0 && j < nd) {
            int64_t shape[MAXDIMS], strides[MAXDIMS];
            memcpy(shape, vp_array_shape(x), (size_t)nd * sizeof(int64_t));
            memcpy(strides, vp_array_strides(x), (size_t)nd * sizeof(int64_t));
            int64_t t = shape[i];
            shape[i] = shape[j];
            shape[j] = t;
            t = strides[i];
            strides[i] = strides[j];
            strides[j] = t;
            return make_view(self, vp_array_dtype(x), nd, shape, strides, 0, 0);
        }
    } else if (PyErr_Occurred()) {
        return NULL;
    }
    static PyObject* name = NULL;
    if (name == NULL && (name = PyUnicode_InternFromString("_swapaxes_py")) == NULL) {
        return NULL;
    }
    PyObject* stack[8];
    const Py_ssize_t total = nargs + (kwnames != NULL ? PyTuple_GET_SIZE(kwnames) : 0);
    if (total >= 8) {
        PyErr_SetString(PyExc_TypeError, "swapaxes: too many arguments");
        return NULL;
    }
    stack[0] = self;
    memcpy(stack + 1, args, (size_t)total * sizeof(PyObject*));
    return PyObject_VectorcallMethod(name, stack, (size_t)(nargs + 1), kwnames);
}

/* broadcast_to(a, shape): NumPy's read-only broadcast view (stride 0 for
 * broadcast dimensions); None if the shapes do not broadcast. */
static PyObject* m_broadcast_to(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *a, *shape_o;
    if (!PyArg_ParseTuple(args, "O!O", &ArrayType, &a, &shape_o) || check_runtime() < 0) {
        return NULL;
    }
    vp_array* x = array_of(a);
    int64_t dims[MAXDIMS], st[MAXDIMS];
    const int nd = read_dims(shape_o, dims);
    const int ond = vp_array_ndim(x);
    if (nd < 0) {
        PyErr_Clear();
        Py_RETURN_NONE;
    }
    if (nd < ond) {
        Py_RETURN_NONE;
    }
    for (int i = 0; i < nd; ++i) {
        const int j = i - (nd - ond);
        if (dims[i] < 0) {
            Py_RETURN_NONE;
        }
        if (j < 0) {
            st[i] = 0;
        } else if (vp_array_shape(x)[j] == dims[i]) {
            st[i] = dims[i] == 1 ? 0 : vp_array_strides(x)[j]; /* nditer: 0 for extent 1 */
        } else if (vp_array_shape(x)[j] == 1) {
            st[i] = 0;
        } else {
            Py_RETURN_NONE;
        }
    }
    return make_view(a, vp_array_dtype(x), nd, dims, st, 0, 1);
}

static PyObject* m_view(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *base, *shape_o, *strides_o;
    long long offset;
    int code = -1; /* dtype of the view; default: the base's */
    if (!PyArg_ParseTuple(args, "O!OOL|i", &ArrayType, &base, &shape_o, &strides_o, &offset,
                          &code) ||
        check_runtime() < 0) {
        return NULL;
    }
    if (code < 0) {
        code = (int)vp_array_dtype(array_of(base));
    }
    int64_t shape[MAXDIMS], strides[MAXDIMS];
    int nd = read_dims(shape_o, shape);
    if (nd < 0 || read_dims(strides_o, strides) != nd) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "shape and strides differ in length");
        }
        return NULL;
    }
    PyObject* v =
        wrap_array(vp_array_view(g_rt, array_of(base), (vp_dtype)code, nd, shape, strides, offset));
    if (v != NULL) {
        ((ArrayObject*)v)->readonly = ((ArrayObject*)base)->readonly; /* as NumPy views */
    }
    return v;
}

static PyObject* m_scalar(PyObject* self, PyObject* a) {
    (void)self;
    if (!PyObject_TypeCheck(a, &ArrayType) || check_runtime() < 0) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "scalar: a viproc array expected");
        }
        return NULL;
    }
    vp_array* x = array_of(a);
    if (!vp_array_ready(x)) {
        Py_RETURN_NONE;
    }
    char err[ERRLEN] = "";
    if (raise_status(vp_array_wait(g_rt, x, err, sizeof err), err) < 0) {
        return NULL; /* the producer failed */
    }
    return vp_array_scalar(x);
}

/* item(a, key): for an integer key (an int, or a tuple of one int per
 * dimension) the element as NumPy's scalar if `a` is ready; None if `a` is
 * pending or the key is something else. IndexError as NumPy's. */
static PyObject* m_item(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    (void)self;
    if (array_and_arg(args, nargs) < 0) {
        return NULL;
    }
    PyObject *a = args[0], *key = args[1];
    vp_array* x = array_of(a);
    const int nd = vp_array_ndim(x);
    const Py_ssize_t nk = PyTuple_Check(key) ? PyTuple_GET_SIZE(key) : 1;
    if (nk != nd || nd == 0 || !vp_array_ready(x)) {
        Py_RETURN_NONE;
    }
    const int64_t* shape = vp_array_shape(x);
    const int64_t* strides = vp_array_strides(x);
    int64_t offset = 0;
    for (int i = 0; i < nd; ++i) {
        PyObject* k = PyTuple_Check(key) ? PyTuple_GET_ITEM(key, i) : key;
        if (PyBool_Check(k) ||
            !(PyLong_Check(k) ||
              (g_np_integer != NULL && PyObject_TypeCheck(k, (PyTypeObject*)g_np_integer)))) {
            Py_RETURN_NONE; /* arrays, slices, masks, np.bool_: the general path */
        }
        const Py_ssize_t v = PyNumber_AsSsize_t(k, PyExc_IndexError);
        if (v == -1 && PyErr_Occurred()) {
            return NULL;
        }
        if (v < -shape[i] || v >= shape[i]) {
            PyErr_Format(PyExc_IndexError, "index %zd is out of bounds for axis %d with size %lld",
                         v, i, (long long)shape[i]);
            return NULL;
        }
        offset += (v < 0 ? v + shape[i] : v) * strides[i];
    }
    int64_t zero = 0;
    vp_array* e = vp_array_view(g_rt, x, vp_array_dtype(x), 0, &zero, &zero, offset);
    PyObject* r = vp_array_scalar(e);
    vp_array_release(e);
    return r;
}

static PyObject* m_ready(PyObject* self, PyObject* o) {
    (void)self;
    vp_array* a = array_of(o);
    if (a == NULL || check_runtime() < 0) {
        return NULL;
    }
    return PyBool_FromLong(vp_array_ready(a));
}

static PyObject* m_same_storage(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *x, *y;
    if (!PyArg_ParseTuple(args, "O!O!", &ArrayType, &x, &ArrayType, &y)) {
        return NULL;
    }
    return PyBool_FromLong(vp_array_same_storage(array_of(x), array_of(y)));
}

static int wait_array(vp_array* a) {
    char err[ERRLEN] = "";
    int status;
    Py_BEGIN_ALLOW_THREADS;
    status = vp_array_wait(g_rt, a, err, sizeof err);
    Py_END_ALLOW_THREADS;
    return raise_status(status, err);
}

static PyObject* m_wait(PyObject* self, PyObject* o) {
    (void)self;
    vp_array* a = array_of(o);
    if (a == NULL || check_runtime() < 0 || wait_array(a) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* export(a, writable=False): NumPy view of a's data, no copy (D7). A writable
 * export first makes the buffer exclusive (vp_array_prepare_write); it is for
 * NumPy code the bridge runs synchronously, never handed to user code. */
static PyObject* m_export(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* o;
    int writable = 0;
    if (!PyArg_ParseTuple(args, "O!|p", &ArrayType, &o, &writable) || check_runtime() < 0) {
        return NULL;
    }
    vp_array* a = array_of(o);
    writable = writable && !((ArrayObject*)o)->readonly;
    if (writable) {
        char err[ERRLEN] = "";
        int status;
        Py_BEGIN_ALLOW_THREADS;
        status = vp_array_prepare_write(g_rt, a, err, sizeof err);
        Py_END_ALLOW_THREADS;
        if (raise_status(status, err) < 0) {
            return NULL;
        }
    } else if (wait_array(a) < 0) {
        return NULL;
    }
    return vp_array_export(a, writable);
}

static PyObject* m_assign(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *dst, *src;
    if (!PyArg_ParseTuple(args, "O!O!", &ArrayType, &dst, &ArrayType, &src) ||
        check_runtime() < 0) {
        return NULL;
    }
    if (((ArrayObject*)dst)->readonly) {
        PyErr_SetString(PyExc_ValueError, "assignment destination is read-only");
        return NULL;
    }
    char err[ERRLEN] = "";
    int status = vp_assign(g_rt, array_of(dst), array_of(src), issue_site(), err, sizeof err);
    if (raise_status(status, err) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* reduce(ufunc, a, axes, keepdims, dtype, initial): async reduction with
 * NumPy's reduction loop (vp_reduce). `axes`: a sequence of bools, one per
 * dimension of `a`; `dtype`: dtype= or None; `initial`: start value or None
 * (start at the first elements). Returns the (pending) result, or None if
 * viproc cannot run it (the caller falls back to NumPy). */
static PyObject* m_reduce(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *ufunc, *a, *axes, *dtype, *identity;
    int keepdims;
    if (!PyArg_ParseTuple(args, "OO!OpOO", &ufunc, &ArrayType, &a, &axes, &keepdims, &dtype,
                          &identity) ||
        check_runtime() < 0) {
        return NULL;
    }
    vp_array* in = array_of(a);
    const int nd = vp_array_ndim(in);
    PyObject* fast = PySequence_Fast(axes, "axes must be a sequence");
    if (fast == NULL) {
        return NULL;
    }
    if (PySequence_Fast_GET_SIZE(fast) != nd || nd > MAXDIMS) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "reduce: one axis flag per dimension expected");
        return NULL;
    }
    unsigned char flags[MAXDIMS];
    for (int i = 0; i < nd; ++i) {
        const int t = PyObject_IsTrue(PySequence_Fast_GET_ITEM(fast, i));
        if (t < 0) {
            Py_DECREF(fast);
            return NULL;
        }
        flags[i] = (unsigned char)t;
    }
    Py_DECREF(fast);
    vp_array* res = NULL;
    char err[ERRLEN] = "";
    const int status = vp_reduce(g_rt, ufunc, in, flags, keepdims, dtype, identity, &res,
                                 issue_site(), err, sizeof err);
    if (status == VP_UNSUPPORTED) {
        Py_RETURN_NONE;
    }
    if (raise_status(status, err) < 0) {
        return NULL;
    }
    return wrap_array(res);
}

/* --- reductions in C --------------------------------------------------------
 * a.sum(), np.sum(a), ... (also mean, var, std, argmax, argmin) without a
 * Python frame:
 *  - Small ready data (at most g_small elements, the direct path's bound):
 *    NumPy itself computes it, with the caller's arguments, on a borrowed
 *    read-only view of the buffer (vp_array_borrow): NumPy's result, errors
 *    and warnings by construction, at NumPy's cost. There is nothing to run
 *    asynchronously, as for the direct path. An ndarray result is fresh
 *    memory and adopted.
 *  - Otherwise sum/prod/max/min/any/all with axis and keepdims only:
 *    vp_reduce, asynchronously.
 *  - Everything else: the Python handlers (_ndarray.py). */

enum { RK_SUM = 0, RK_MINMAX = 1, RK_ANYALL = 2, RK_OTHER = 3 };
enum { RA_AXIS = 0, RA_DTYPE, RA_OUT, RA_KEEPDIMS, RA_INITIAL, RA_WHERE, RA_N };

typedef struct {
    PyObject* func;     /* np.sum, ... */
    PyObject* impl;     /* func._implementation: NumPy's code without the dispatch */
    PyObject* method;   /* ndarray method name (interned), or NULL */
    PyObject* ufunc;    /* np.add, ... (NULL for RK_OTHER) */
    PyObject* identity; /* ufunc.identity (Py_None for minimum/maximum) */
    PyObject* fallback; /* fallback(args, kwargs): the Python implementation */
    int kind;
} RedSpec;

#define MAX_RED 32
static RedSpec g_red[MAX_RED];
static int g_nred = 0;
static PyObject* g_array_function = NULL; /* array_function(self, func, types, args, kwargs) */
static PyObject* g_novalue = NULL;        /* np._NoValue */
static int g_red_async = 0;               /* NumPy >= 2.3: vp_reduce's blocks are NumPy's */

/* Positional parameters per kind, in NumPy's order (any/all: `where` is
 * keyword-only). */
static const int red_order[3][RA_N] = {
    {RA_AXIS, RA_DTYPE, RA_OUT, RA_KEEPDIMS, RA_INITIAL, RA_WHERE},
    {RA_AXIS, RA_OUT, RA_KEEPDIMS, RA_INITIAL, RA_WHERE, -1},
    {RA_AXIS, RA_OUT, RA_KEEPDIMS, -1, -1, -1},
};
static const int red_npos[3] = {6, 5, 3};
static const char* const red_names[RA_N] = {"axis", "dtype", "out", "keepdims", "initial", "where"};
static PyObject* s_red_names[RA_N];

static const RedSpec* red_spec_of_method(const char* name) {
    for (int i = 0; i < g_nred; ++i) {
        if (g_red[i].method != NULL && strcmp(PyUnicode_AsUTF8(g_red[i].method), name) == 0) {
            return &g_red[i];
        }
    }
    return NULL;
}

/* The arguments of an RK_SUM/RK_MINMAX/RK_ANYALL call by slot (NULL:
 * absent). 0 if they do not parse as NumPy's signature (NumPy reports it). */
static int red_parse(int kind, PyObject* const* pos, Py_ssize_t npos, PyObject* const* kwn,
                     PyObject* const* kwv, Py_ssize_t nkw, PyObject** v) {
    for (int i = 0; i < RA_N; ++i) {
        v[i] = NULL;
    }
    if (npos > red_npos[kind]) {
        return 0;
    }
    for (Py_ssize_t i = 0; i < npos; ++i) {
        v[red_order[kind][i]] = pos[i];
    }
    for (Py_ssize_t i = 0; i < nkw; ++i) {
        int slot = -1;
        for (int k = 0; k < RA_N && slot < 0; ++k) {
            if (kwn[i] == s_red_names[k] || PyUnicode_Compare(kwn[i], s_red_names[k]) == 0) {
                slot = k;
            }
        }
        const int allowed = slot >= 0 && !(kind == RK_MINMAX && slot == RA_DTYPE) &&
                            !(kind == RK_ANYALL && (slot == RA_DTYPE || slot == RA_INITIAL));
        if (!allowed || v[slot] != NULL) {
            return 0;
        }
        v[slot] = kwv[i];
    }
    return 1;
}

static int red_absent(PyObject* v) { return v == NULL || v == g_novalue; }

/* Axis flags of `axis` (None, an int or a tuple of ints) for `nd`
 * dimensions; 0 for anything NumPy has to judge (out of range, repeated). */
static int red_axes(PyObject* axis, int nd, unsigned char* flags) {
    if (red_absent(axis) || axis == Py_None) {
        memset(flags, 1, (size_t)nd);
        return 1;
    }
    memset(flags, 0, (size_t)nd);
    const Py_ssize_t n = PyTuple_Check(axis) ? PyTuple_GET_SIZE(axis) : 1;
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* x = PyTuple_Check(axis) ? PyTuple_GET_ITEM(axis, i) : axis;
        if (!PyLong_CheckExact(x)) {
            return 0;
        }
        const long k = PyLong_AsLong(x);
        if (k == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return 0;
        }
        if (k < -nd || k >= nd || flags[k < 0 ? k + nd : k]) {
            return 0;
        }
        flags[k < 0 ? k + nd : k] = 1;
    }
    return 1;
}

static int64_t array_size(vp_array* a) {
    const int nd = vp_array_ndim(a);
    const int64_t* shape = vp_array_shape(a);
    int64_t n = 1;
    for (int i = 0; i < nd; ++i) {
        n *= shape[i];
    }
    return n;
}

/* 1 if NumPy computes the call right away on `self`'s data (small, ready,
 * no other viproc array among the arguments). */
static int red_delegates(PyObject* self, PyObject* const* v, Py_ssize_t n) {
    if (Py_TYPE(self) != g_array_cls || !vp_array_ready(((ArrayObject*)self)->a) ||
        array_size(((ArrayObject*)self)->a) > g_small) {
        return 0;
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (is_array(v[i])) {
            return 0; /* out=, where= of viproc arrays: the Python handlers */
        }
    }
    return 1;
}

/* NumPy's read-only view of a ready array's data, borrowed for one call. */
static PyObject* red_borrow(vp_array* a) {
    char err[ERRLEN] = "";
    if (raise_status(vp_array_wait(g_rt, a, err, sizeof err), err) < 0) {
        return NULL; /* the producer failed */
    }
    return vp_array_borrow(a);
}

/* Ends the borrow: a view NumPy kept a reference to becomes an export. Then
 * maps the result: an ndarray (fresh memory) is adopted. */
static PyObject* red_return(PyObject* self, PyObject* view, PyObject* r) {
    vp_array* a = ((ArrayObject*)self)->a;
    if (r == view) { /* e.g. astype(copy=False) of the same dtype: NumPy's `a` itself */
        Py_DECREF(r);
        Py_DECREF(view);
        return Py_NewRef(self);
    }
    if (Py_REFCNT(view) > 1 && vp_array_pin(a, view) != VP_OK) {
        Py_DECREF(view);
        Py_XDECREF(r);
        return NULL;
    }
    Py_DECREF(view);
    if (r != NULL && Py_TYPE(r) == (PyTypeObject*)g_np_ndarray) {
        vp_array* x = NULL;
        char err[ERRLEN] = "";
        if (vp_array_from_ndarray(g_rt, r, &x, err, sizeof err) == VP_OK) {
            Py_DECREF(r);
            return wrap_array(x);
        }
        /* a dtype viproc does not support: NumPy's array, as the fallback */
    }
    return r;
}

/* 1 if the call gives an array (an axis or keepdims): vp_reduce, run
 * directly for small ready data, makes it cheaper than NumPy's call on a
 * borrowed view plus adopting NumPy's result. */
static int red_array_result(PyObject* const* v) {
    return !(red_absent(v[RA_AXIS]) || v[RA_AXIS] == Py_None) || v[RA_KEEPDIMS] == Py_True;
}

/* sum/prod/max/min/any/all asynchronously (vp_reduce). 1 with *result, 0 if
 * the Python handlers decide, -1 on error. */
static int red_async(const RedSpec* sp, PyObject* self, PyObject* const* v, PyObject** result) {
    if (!g_red_async || sp->kind == RK_OTHER || Py_TYPE(self) != g_array_cls) {
        return 0;
    }
    if (!(red_absent(v[RA_OUT]) || v[RA_OUT] == Py_None) ||
        !(red_absent(v[RA_DTYPE]) || v[RA_DTYPE] == Py_None) || !red_absent(v[RA_INITIAL]) ||
        !(red_absent(v[RA_WHERE]) || v[RA_WHERE] == Py_True)) {
        return 0;
    }
    int keepdims = 0;
    if (!red_absent(v[RA_KEEPDIMS])) {
        if (v[RA_KEEPDIMS] != Py_True && v[RA_KEEPDIMS] != Py_False) {
            return 0;
        }
        keepdims = v[RA_KEEPDIMS] == Py_True;
    }
    vp_array* a = ((ArrayObject*)self)->a;
    if (sp->kind == RK_ANYALL && vp_array_dtype(a) != VP_BOOL) {
        return 0; /* `a != 0` first: the Python handler */
    }
    unsigned char flags[MAXDIMS];
    if (!red_axes(v[RA_AXIS], vp_array_ndim(a), flags)) {
        return 0;
    }
    vp_array* r = NULL;
    char err[ERRLEN] = "";
    const int st = vp_reduce(g_rt, sp->ufunc, a, flags, keepdims, Py_None, sp->identity, &r,
                             issue_site(), err, sizeof err);
    if (st == VP_UNSUPPORTED) {
        return 0;
    }
    if (raise_status(st, err) < 0) {
        return -1;
    }
    if (vp_array_ndim(r) == 0 && vp_array_ready(r) &&
        vp_array_wait(g_rt, r, err, sizeof err) == VP_OK) {
        *result = vp_array_scalar(r); /* NumPy's scalar: the data is there */
        vp_array_release(r);
        return *result != NULL ? 1 : -1;
    }
    *result = wrap_array(r);
    if (*result == NULL) {
        return -1;
    }
    if (vp_array_ndim(r) == 0) {
        ((ArrayObject*)*result)->scalar = 1;
    }
    return 1;
}

static int red_ready(void) {
    if (g_array_function == NULL || g_array_cls == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "viproc: setup_reductions() was not called");
        return -1;
    }
    return check_runtime();
}

/* The Python handlers: fallback(func, args, kwargs). */
static PyObject* red_fallback(const RedSpec* sp, PyObject* self, PyObject* const* pos,
                              Py_ssize_t npos, PyObject* kwnames, PyObject* const* kwv) {
    PyObject* args = PyTuple_New(npos + 1);
    PyObject* kw = PyDict_New();
    if (args == NULL || kw == NULL) {
        Py_XDECREF(args);
        Py_XDECREF(kw);
        return NULL;
    }
    PyTuple_SET_ITEM(args, 0, Py_NewRef(self));
    for (Py_ssize_t i = 0; i < npos; ++i) {
        PyTuple_SET_ITEM(args, i + 1, Py_NewRef(pos[i]));
    }
    const Py_ssize_t nkw = kwnames != NULL ? PyTuple_GET_SIZE(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; ++i) {
        if (PyDict_SetItem(kw, PyTuple_GET_ITEM(kwnames, i), kwv[i]) < 0) {
            Py_DECREF(args);
            Py_DECREF(kw);
            return NULL;
        }
    }
    PyObject* r = PyObject_CallFunctionObjArgs(sp->fallback, args, kw, NULL);
    Py_DECREF(args);
    Py_DECREF(kw);
    return r;
}

/* a.<method>(*args, **kwargs) */
static PyObject* red_method(const char* name, int* cache, PyObject* self, PyObject* const* args,
                            Py_ssize_t nargs, PyObject* kwnames) {
    if (red_ready() < 0) {
        return NULL;
    }
    if (*cache < 0 || *cache >= g_nred) {
        const RedSpec* found = red_spec_of_method(name);
        if (found == NULL) {
            PyErr_Format(PyExc_RuntimeError, "viproc: %s not registered", name);
            return NULL;
        }
        *cache = (int)(found - g_red);
    }
    const RedSpec* sp = &g_red[*cache];
    const Py_ssize_t nkw = kwnames != NULL ? PyTuple_GET_SIZE(kwnames) : 0;
    PyObject* v[RA_N];
    PyObject* const* kwn = nkw > 0 ? &PyTuple_GET_ITEM(kwnames, 0) : NULL;
    const int parsed =
        sp->kind != RK_OTHER && red_parse(sp->kind, args, nargs, kwn, args + nargs, nkw, v);
    if (parsed && red_array_result(v)) {
        PyObject* result = NULL;
        const int h = red_async(sp, self, v, &result);
        if (h != 0) {
            return h > 0 ? result : NULL;
        }
    }
    if (nargs + nkw <= RA_N + 4 && red_delegates(self, args, nargs + nkw)) {
        vp_array* a = ((ArrayObject*)self)->a;
        PyObject* view = red_borrow(a);
        if (view == NULL) {
            return NULL;
        }
        PyObject* stack[RA_N + 5];
        stack[0] = view;
        memcpy(stack + 1, args, (size_t)(nargs + nkw) * sizeof(PyObject*));
        PyObject* r = PyObject_VectorcallMethod(sp->method, stack, (size_t)(nargs + 1), kwnames);
        return red_return(self, view, r);
    }
    if (parsed && !red_array_result(v)) {
        PyObject* result = NULL;
        const int h = red_async(sp, self, v, &result);
        if (h != 0) {
            return h > 0 ? result : NULL;
        }
    }
    return red_fallback(sp, self, args, nargs, kwnames, args + nargs);
}

#define X(name)                                                                                    \
    static PyObject* Array_##name(PyObject* self, PyObject* const* args, Py_ssize_t nargs,         \
                                  PyObject* kwnames) {                                             \
        static int cache = -1; /* index in g_red (fixed after setup_reductions) */                 \
        return red_method(#name, &cache, self, args, nargs, kwnames);                              \
    }
RED_METHODS(X)
#undef X

/* Number of elements of a where() operand for the delegate's size check:
 * viproc arrays (ready only), ndarrays and scalars; -1 for anything else
 * (lists, pending arrays): the Python handler decides. *nd/shape get its
 * shape (at most MAXDIMS dimensions). */
static int where_operand(PyObject* x, int* nd, int64_t* shape) {
    if (is_array(x)) {
        vp_array* a = ((ArrayObject*)x)->a;
        if (!vp_array_ready(a) || vp_array_ndim(a) > MAXDIMS) {
            return -1;
        }
        *nd = vp_array_ndim(a);
        memcpy(shape, vp_array_shape(a), sizeof(int64_t) * (size_t)*nd);
        return 0;
    }
    if (PyFloat_Check(x) || PyLong_Check(x) || PyComplex_Check(x) ||
        PyObject_TypeCheck(x, (PyTypeObject*)g_np_generic)) {
        *nd = 0;
        return 0;
    }
    if (Py_TYPE(x) == (PyTypeObject*)g_np_ndarray) {
        PyObject* t = PyObject_GetAttrString(x, "shape");
        if (t == NULL || !PyTuple_Check(t) || PyTuple_GET_SIZE(t) > MAXDIMS) {
            Py_XDECREF(t);
            PyErr_Clear();
            return -1;
        }
        *nd = (int)PyTuple_GET_SIZE(t);
        for (int i = 0; i < *nd; ++i) {
            shape[i] = PyLong_AsLongLong(PyTuple_GET_ITEM(t, i));
        }
        Py_DECREF(t);
        return 0;
    }
    return -1;
}

/* Kind of a dtype or Python scalar for NEP 50's weak promotion: bool 0,
 * integer 1, floating 2, complex 3. */
static int dtype_kind(vp_dtype d) {
    return d == VP_BOOL ? 0 : d <= VP_INT64 ? 1 : d <= VP_FLOAT64 ? 2 : 3;
}
static int scalar_kind(PyObject* v) {
    return PyBool_Check(v)           ? 0
           : PyLong_CheckExact(v)    ? 1
           : PyFloat_CheckExact(v)   ? 2
           : PyComplex_CheckExact(v) ? 3
                                     : -1;
}

/* np.where(cond, x, y) as a select (vp_select: run directly for small ready
 * data, asynchronously otherwise) when NumPy's result dtype needs no cast:
 * a bool viproc condition, x and y viproc arrays of one dtype, or one of
 * them a Python scalar that NEP 50 promotes weakly to the other's dtype
 * (converted now, as NumPy converts it). NULL without an error otherwise. */
static PyObject* where_select(PyObject* fargs, PyObject* fkw) {
    if (PyTuple_GET_SIZE(fargs) != 3 || PyDict_GET_SIZE(fkw) != 0) {
        return NULL;
    }
    PyObject *c = PyTuple_GET_ITEM(fargs, 0), *x = PyTuple_GET_ITEM(fargs, 1),
             *y = PyTuple_GET_ITEM(fargs, 2);
    if (!is_array(c) || vp_array_dtype(((ArrayObject*)c)->a) != VP_BOOL) {
        return NULL;
    }
    const int xa = is_array(x), ya = is_array(y);
    vp_dtype d;
    if (xa && ya) {
        d = vp_array_dtype(((ArrayObject*)x)->a);
        if (vp_array_dtype(((ArrayObject*)y)->a) != d) {
            return NULL;
        }
    } else if (xa || ya) {
        d = vp_array_dtype(((ArrayObject*)(xa ? x : y))->a);
        const int k = scalar_kind(xa ? y : x);
        if (k < 0 || k > dtype_kind(d)) {
            return NULL;
        }
    } else {
        return NULL;
    }
    vp_array* conv = NULL;
    if (!(xa && ya) && vp_array_from_item(g_rt, xa ? y : x, d, &conv) != VP_OK) {
        PyErr_Clear(); /* NumPy raises its own error */
        return NULL;
    }
    vp_array* r = NULL;
    char err[ERRLEN] = "";
    const int status =
        vp_select(g_rt, ((ArrayObject*)c)->a, xa ? ((ArrayObject*)x)->a : conv,
                  ya ? ((ArrayObject*)y)->a : conv, &r, issue_site(), err, sizeof err);
    if (conv != NULL) {
        vp_array_release(conv);
    }
    if (status == VP_INVALID || status == VP_UNSUPPORTED) {
        return NULL; /* broadcasting errors: NumPy's, from the handlers */
    }
    if (raise_status(status, err) < 0) {
        return NULL;
    }
    return wrap_array(r);
}

/* np.where(cond, x, y) of small ready data: NumPy's own where on borrowed
 * views of the viproc operands (design doc D7), its fresh result adopted.
 * Values, dtype, broadcasting and errors are NumPy's. NULL without an error
 * when the call is not small and ready: the Python handler (a select task). */
static PyObject* where_delegate(PyObject* fargs, PyObject* fkw) {
    if (PyTuple_GET_SIZE(fargs) != 3 || PyDict_GET_SIZE(fkw) != 0) {
        return NULL;
    }
    int out_nd = 0;
    int64_t out_shape[MAXDIMS];
    for (int i = 0; i < 3; ++i) {
        int nd;
        int64_t shape[MAXDIMS];
        if (where_operand(PyTuple_GET_ITEM(fargs, i), &nd, shape) < 0) {
            return NULL;
        }
        for (int k = 0; k < nd; ++k) { /* broadcast, right-aligned; NumPy checks */
            const int o = MAXDIMS - nd + k;
            const int64_t cur = o >= MAXDIMS - out_nd ? out_shape[o] : 1;
            out_shape[o] = cur == 1 ? shape[k] : cur;
        }
        if (nd > out_nd) {
            out_nd = nd;
        }
    }
    int64_t size = 1;
    for (int o = MAXDIMS - out_nd; o < MAXDIMS; ++o) {
        size *= out_shape[o];
    }
    if (size > g_small) {
        return NULL;
    }
    PyObject* t = PyTuple_New(3);
    PyObject* views[3] = {NULL, NULL, NULL};
    for (int i = 0; t != NULL && i < 3; ++i) {
        PyObject* x = PyTuple_GET_ITEM(fargs, i);
        if (is_array(x) && (views[i] = red_borrow(((ArrayObject*)x)->a)) == NULL) {
            Py_CLEAR(t);
            break;
        }
        PyTuple_SET_ITEM(t, i, Py_NewRef(views[i] != NULL ? views[i] : x));
    }
    PyObject* r = t != NULL ? PyObject_Call(g_where_impl, t, NULL) : NULL;
    Py_XDECREF(t);
    for (int i = 0; i < 3; ++i) {
        if (views[i] != NULL &&
            end_borrow(((ArrayObject*)PyTuple_GET_ITEM(fargs, i))->a, views[i]) < 0) {
            Py_CLEAR(r);
        }
    }
    if (r != NULL && Py_TYPE(r) == (PyTypeObject*)g_np_ndarray) {
        vp_array* a = NULL;
        char err[ERRLEN] = "";
        if (vp_array_from_ndarray(g_rt, r, &a, err, sizeof err) == VP_OK) {
            Py_DECREF(r);
            return wrap_array(a);
        }
        /* a dtype viproc does not support: NumPy's array, as the fallback */
    }
    if (r == NULL && !PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "viproc: where failed");
    }
    return r;
}

/* _Array.__array_function__(self, func, types, args, kwargs): NumPy's
 * reductions as the methods above, np.where of small ready data by NumPy;
 * every other function in _ndarray.py. */
static PyObject* Array_array_function(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 4 || !PyTuple_Check(args[2]) || !PyDict_Check(args[3])) {
        PyErr_SetString(PyExc_TypeError, "__array_function__(func, types, args, kwargs)");
        return NULL;
    }
    if (red_ready() < 0) {
        return NULL;
    }
    PyObject *func = args[0], *fargs = args[2], *fkw = args[3];
    if (func == g_where_func) {
        PyObject* r = where_select(fargs, fkw);
        if (r == NULL && !PyErr_Occurred()) {
            r = where_delegate(fargs, fkw);
        }
        if (r != NULL || PyErr_Occurred()) {
            return r;
        }
    }
    const RedSpec* sp = NULL;
    for (int i = 0; i < g_nred && sp == NULL; ++i) {
        sp = g_red[i].func == func && func != Py_None ? &g_red[i] : NULL;
    }
    const Py_ssize_t npos = PyTuple_GET_SIZE(fargs);
    const Py_ssize_t nkw = PyDict_GET_SIZE(fkw);
    if (sp == NULL || npos < 1 || Py_TYPE(PyTuple_GET_ITEM(fargs, 0)) != g_array_cls ||
        nkw > RA_N + 4) {
        PyObject* call[5] = {self, func, args[1], fargs, fkw};
        return PyObject_Vectorcall(g_array_function, call, 5, NULL);
    }
    PyObject* target = PyTuple_GET_ITEM(fargs, 0);
    PyObject* kwn[RA_N + 4];
    PyObject* kwv[RA_N + 4];
    Py_ssize_t it = 0, k = 0;
    PyObject *key, *value;
    while (PyDict_Next(fkw, &it, &key, &value)) {
        kwn[k] = key;
        kwv[k++] = value;
    }
    PyObject* const* pos = &PyTuple_GET_ITEM(fargs, 1);
    PyObject* v[RA_N];
    const int parsed = sp->kind != RK_OTHER && red_parse(sp->kind, pos, npos - 1, kwn, kwv, nkw, v);
    if (parsed && red_array_result(v)) {
        PyObject* result = NULL;
        const int h = red_async(sp, target, v, &result);
        if (h != 0) {
            return h > 0 ? result : NULL;
        }
    }
    int delegate =
        sp->impl != NULL && red_delegates(target, pos, npos - 1) && red_delegates(target, kwv, nkw);
    if (delegate) {
        vp_array* a = ((ArrayObject*)target)->a;
        PyObject* view = red_borrow(a);
        if (view == NULL) {
            return NULL;
        }
        PyObject* t = PyTuple_New(npos);
        if (t == NULL) {
            Py_DECREF(view);
            return NULL;
        }
        PyTuple_SET_ITEM(t, 0, Py_NewRef(view));
        for (Py_ssize_t i = 1; i < npos; ++i) {
            PyTuple_SET_ITEM(t, i, Py_NewRef(PyTuple_GET_ITEM(fargs, i)));
        }
        PyObject* r = PyObject_Call(sp->impl, t, nkw > 0 ? fkw : NULL);
        Py_DECREF(t);
        return red_return(target, view, r);
    }
    if (parsed && !red_array_result(v)) {
        PyObject* result = NULL;
        const int h = red_async(sp, target, v, &result);
        if (h != 0) {
            return h > 0 ? result : NULL;
        }
    }
    PyObject* call[2] = {fargs, fkw};
    return PyObject_Vectorcall(sp->fallback, call, 2, NULL);
}

/* setup_reductions(entries, array_function, novalue, async_ok): entries =
 * ((func or None, method name or None, kind, ufunc or None, fallback), ...),
 * fallback(args, kwargs) running the call in Python. */
static PyObject* m_setup_reductions(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *entries, *array_function, *novalue;
    int async_ok;
    if (!PyArg_ParseTuple(args, "O!OOp", &PyTuple_Type, &entries, &array_function, &novalue,
                          &async_ok)) {
        return NULL;
    }
    if (PyTuple_GET_SIZE(entries) > MAX_RED) {
        PyErr_SetString(PyExc_ValueError, "setup_reductions: too many entries");
        return NULL;
    }
    for (int i = 0; i < RA_N; ++i) {
        if (s_red_names[i] == NULL &&
            (s_red_names[i] = PyUnicode_InternFromString(red_names[i])) == NULL) {
            return NULL;
        }
    }
    for (int i = 0; i < g_nred; ++i) {
        Py_XDECREF(g_red[i].func);
        Py_XDECREF(g_red[i].impl);
        Py_XDECREF(g_red[i].method);
        Py_XDECREF(g_red[i].ufunc);
        Py_XDECREF(g_red[i].identity);
        Py_XDECREF(g_red[i].fallback);
    }
    g_nred = 0;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(entries); ++i) {
        PyObject *func, *method, *ufunc, *fallback;
        int kind;
        if (!PyArg_ParseTuple(PyTuple_GET_ITEM(entries, i), "OOiOO", &func, &method, &kind, &ufunc,
                              &fallback)) {
            return NULL;
        }
        RedSpec sp = {Py_NewRef(func), NULL, NULL, NULL, NULL, Py_NewRef(fallback), kind};
        if (func != Py_None) {
            sp.impl = PyObject_GetAttrString(func, "_implementation");
            PyErr_Clear(); /* no implementation: no delegate for the function form */
        }
        if (method != Py_None) {
            Py_INCREF(method);
            PyUnicode_InternInPlace(&method);
            sp.method = method;
        }
        if (ufunc != Py_None) {
            sp.ufunc = Py_NewRef(ufunc);
            sp.identity = PyObject_GetAttrString(ufunc, "identity");
            if (sp.identity == NULL) {
                return NULL;
            }
        }
        g_red[g_nred++] = sp;
    }
    Py_XSETREF(g_array_function, Py_NewRef(array_function));
    Py_XSETREF(g_novalue, Py_NewRef(novalue));
    g_red_async = async_ok;
    Py_RETURN_NONE;
}

static PyObject* m_astype(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* a;
    int code;
    int order;
    if (!PyArg_ParseTuple(args, "O!iC", &ArrayType, &a, &code, &order) || check_runtime() < 0) {
        return NULL;
    }
    vp_array* res = NULL;
    char err[ERRLEN] = "";
    const int status = vp_astype(g_rt, array_of(a), (vp_dtype)code, (char)order, &res, issue_site(),
                                 err, sizeof err);
    if (status == VP_UNSUPPORTED) {
        Py_RETURN_NONE;
    }
    if (raise_status(status, err) < 0) {
        return NULL;
    }
    return wrap_array(res);
}

static PyObject* m_arg_reduce(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *a, *shape_o;
    int argmax;
    if (!PyArg_ParseTuple(args, "pO!O", &argmax, &ArrayType, &a, &shape_o) || check_runtime() < 0) {
        return NULL;
    }
    int64_t shape[MAXDIMS];
    const int nd = read_dims(shape_o, shape);
    if (nd < 0) {
        return NULL;
    }
    vp_array* res = NULL;
    char err[ERRLEN] = "";
    const int status =
        vp_arg_reduce(g_rt, argmax, array_of(a), nd, shape, &res, issue_site(), err, sizeof err);
    if (status == VP_UNSUPPORTED) {
        Py_RETURN_NONE;
    }
    if (raise_status(status, err) < 0) {
        return NULL;
    }
    return wrap_array(res);
}

/* Reads (index arrays, their axes, result axes) for gather/scatter. */
static int read_index_args(PyObject* idx_o, PyObject* axes_o, PyObject* res_o, vp_array** idx,
                           int* axes, int* nidx, int* res, int* nres) {
    PyObject* fi = PySequence_Fast(idx_o, "index arrays: a sequence expected");
    if (fi == NULL) {
        return -1;
    }
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fi);
    int64_t tmp[MAXDIMS];
    if (n > MAXDIMS || read_dims(axes_o, tmp) != n) {
        Py_DECREF(fi);
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "one axis per index array expected");
        }
        return -1;
    }
    for (Py_ssize_t j = 0; j < n; ++j) {
        PyObject* x = PySequence_Fast_GET_ITEM(fi, j);
        if (!PyObject_TypeCheck(x, &ArrayType)) {
            Py_DECREF(fi);
            PyErr_SetString(PyExc_TypeError, "index arrays must be viproc arrays");
            return -1;
        }
        idx[j] = array_of(x);
        axes[j] = (int)tmp[j];
    }
    Py_DECREF(fi);
    *nidx = (int)n;
    const int nr = read_dims(res_o, tmp);
    if (nr < 0) {
        return -1;
    }
    for (int k = 0; k < nr; ++k) {
        res[k] = (int)tmp[k];
    }
    *nres = nr;
    return 0;
}

static PyObject* m_gather(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *a, *idx_o, *axes_o, *res_o, *strides_o;
    if (!PyArg_ParseTuple(args, "O!OOOO", &ArrayType, &a, &idx_o, &axes_o, &res_o, &strides_o) ||
        check_runtime() < 0) {
        return NULL;
    }
    vp_array* idx[MAXDIMS];
    int axes[MAXDIMS], res[MAXDIMS], nidx, nres;
    if (read_index_args(idx_o, axes_o, res_o, idx, axes, &nidx, res, &nres) < 0) {
        return NULL;
    }
    int64_t strides[MAXDIMS];
    const int ns = strides_o == Py_None ? nres : read_dims(strides_o, strides);
    if (ns != nres) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "one stride per result axis expected");
        }
        return NULL;
    }
    vp_array* r = NULL;
    char err[ERRLEN] = "";
    const int status =
        vp_gather(g_rt, array_of(a), nidx, idx, axes, nres, res,
                  strides_o == Py_None ? NULL : strides, &r, issue_site(), err, sizeof err);
    if (raise_status(status, err) < 0) {
        return NULL;
    }
    return wrap_array(r);
}

static PyObject* m_scatter(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *a, *idx_o, *axes_o, *res_o, *values;
    if (!PyArg_ParseTuple(args, "O!OOOO!", &ArrayType, &a, &idx_o, &axes_o, &res_o, &ArrayType,
                          &values) ||
        check_runtime() < 0) {
        return NULL;
    }
    vp_array* idx[MAXDIMS];
    int axes[MAXDIMS], res[MAXDIMS], nidx, nres;
    if (read_index_args(idx_o, axes_o, res_o, idx, axes, &nidx, res, &nres) < 0) {
        return NULL;
    }
    char err[ERRLEN] = "";
    const int status = vp_scatter(g_rt, array_of(a), nidx, idx, axes, nres, res, array_of(values),
                                  issue_site(), err, sizeof err);
    if (raise_status(status, err) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* matmul(a, b): np.matmul of two viproc arrays, or None if NumPy has to run
 * it. */
static PyObject* m_matmul(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "O!O!", &ArrayType, &a, &ArrayType, &b) || check_runtime() < 0) {
        return NULL;
    }
    if (g_matmul == NULL) {
        Py_RETURN_NONE;
    }
    PyObject* r = NULL;
    const int st = fast_matmul(g_matmul, a, b, &r);
    if (st == 0) {
        Py_RETURN_NONE;
    }
    return st > 0 ? r : NULL;
}

static PyObject* m_select(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *c, *x, *y;
    if (!PyArg_ParseTuple(args, "O!O!O!", &ArrayType, &c, &ArrayType, &x, &ArrayType, &y) ||
        check_runtime() < 0) {
        return NULL;
    }
    vp_array* r = NULL;
    char err[ERRLEN] = "";
    const int status =
        vp_select(g_rt, array_of(c), array_of(x), array_of(y), &r, issue_site(), err, sizeof err);
    if (raise_status(status, err) < 0) {
        return NULL;
    }
    PyObject* w = wrap_array(r);
    if (w != NULL && vp_array_ndim(r) == 0) {
        ((ArrayObject*)w)->scalar = 1;
    }
    return w;
}

static PyObject* m_mask_assign(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *a, *mask, *value;
    if (!PyArg_ParseTuple(args, "O!O!O!", &ArrayType, &a, &ArrayType, &mask, &ArrayType, &value) ||
        check_runtime() < 0) {
        return NULL;
    }
    char err[ERRLEN] = "";
    const int status = vp_mask_assign(g_rt, array_of(a), array_of(mask), array_of(value),
                                      issue_site(), err, sizeof err);
    if (raise_status(status, err) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* m_issue_site(PyObject* self, PyObject* noargs) {
    (void)self;
    (void)noargs;
    if (check_runtime() < 0) {
        return NULL;
    }
    return PyUnicode_FromString(issue_site());
}

/* ---- Module-level ufuncs and np.where without NumPy's dispatch -------- */

/* viproc.<ufunc> (module mode: `import viproc as np`, python -m viproc): a
 * proxy of an element-wise NumPy ufunc whose call takes the C hot path
 * (fast_ufunc) right away when an argument is a viproc array, instead of
 * NumPy's override dispatch (PyUFunc_CheckOverride: argument tuple, kwargs
 * dict, __array_ufunc__ call; ~2,000 instructions). Every other call and
 * every attribute (reduce, outer, at, nin, types, ...) goes to the ufunc
 * itself; __class__ is numpy.ufunc, so isinstance(viproc.add, np.ufunc)
 * holds. */
typedef struct {
    PyObject_HEAD PyObject* ufunc;
    vectorcallfunc vectorcall;
} UfuncProxy;

static PyTypeObject UfuncProxyType; /* forward */

static PyObject* UfuncProxy_vectorcall(PyObject* self, PyObject* const* args, size_t nargsf,
                                       PyObject* kwnames) {
    PyObject* uf = ((UfuncProxy*)self)->ufunc;
    const Py_ssize_t n = PyVectorcall_NARGS(nargsf);
    if (kwnames == NULL && n >= 1 && n <= MAXIN && g_rt != NULL) {
        int any = 0;
        for (Py_ssize_t i = 0; i < n; ++i) {
            any |= is_array(args[i]);
        }
        if (any) {
            PyObject* sc[MAXIN];
            const int k = SCALARS_OF(args, n, sc);
            if (k < 0) {
                return NULL;
            }
            if (k > 0) {
                PyObject* r = PyObject_Vectorcall(uf, sc, (size_t)n, NULL);
                release_all(sc, n);
                return r;
            }
            PyObject* result = NULL;
            const int r = fast_ufunc(uf, args, n, NULL, 0, &result);
            if (r != 0) {
                return r > 0 ? result : NULL;
            }
        }
    }
    return PyObject_Vectorcall(uf, args, nargsf, kwnames);
}

static PyObject* UfuncProxy_new(PyTypeObject* type, PyObject* args, PyObject* kw) {
    PyObject* uf;
    if (kw != NULL || !PyArg_ParseTuple(args, "O", &uf)) {
        PyErr_SetString(PyExc_TypeError, "ufunc_proxy(ufunc)");
        return NULL;
    }
    UfuncProxy* p = (UfuncProxy*)type->tp_alloc(type, 0);
    if (p != NULL) {
        p->ufunc = Py_NewRef(uf);
        p->vectorcall = UfuncProxy_vectorcall;
    }
    return (PyObject*)p;
}

static void UfuncProxy_dealloc(PyObject* self) {
    Py_XDECREF(((UfuncProxy*)self)->ufunc);
    Py_TYPE(self)->tp_free(self);
}

static PyObject* UfuncProxy_getattro(PyObject* self, PyObject* name) {
    PyObject* uf = ((UfuncProxy*)self)->ufunc;
    if (PyUnicode_Check(name) && PyUnicode_CompareWithASCIIString(name, "__class__") == 0) {
        return Py_NewRef((PyObject*)Py_TYPE(uf));
    }
    if (PyUnicode_Check(name) && PyUnicode_CompareWithASCIIString(name, "_ufunc") == 0) {
        return Py_NewRef(uf);
    }
    return PyObject_GetAttr(uf, name);
}

static PyObject* unproxy(PyObject* o) {
    return Py_TYPE(o) == &UfuncProxyType ? ((UfuncProxy*)o)->ufunc : o;
}

/* Equal to (and hashed as) the wrapped ufunc. */
static PyObject* UfuncProxy_richcompare(PyObject* a, PyObject* b, int op) {
    return PyObject_RichCompare(unproxy(a), unproxy(b), op);
}

static Py_hash_t UfuncProxy_hash(PyObject* self) {
    return PyObject_Hash(((UfuncProxy*)self)->ufunc);
}

static PyObject* UfuncProxy_repr(PyObject* self) {
    return PyObject_Repr(((UfuncProxy*)self)->ufunc);
}

static PyTypeObject UfuncProxyType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "viproc._viproc.ufunc_proxy",
    .tp_basicsize = sizeof(UfuncProxy),
    .tp_dealloc = UfuncProxy_dealloc,
    .tp_vectorcall_offset = offsetof(UfuncProxy, vectorcall),
    .tp_call = PyVectorcall_Call,
    .tp_repr = UfuncProxy_repr,
    .tp_getattro = UfuncProxy_getattro,
    .tp_richcompare = UfuncProxy_richcompare,
    .tp_hash = UfuncProxy_hash,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_new = UfuncProxy_new,
};

/* viproc.where (module mode): np.where(cond, x, y) without NumPy's
 * __array_function__ dispatch when a viproc array is among the arguments;
 * everything else is numpy.where itself. */
static PyObject* m_where(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                         PyObject* kwnames) {
    (void)self;
    if (kwnames == NULL && nargs == 3 && g_rt != NULL && g_where_func != NULL &&
        (is_array(args[0]) || is_array(args[1]) || is_array(args[2]))) {
        PyObject* t = PyTuple_New(3);
        PyObject* kw = t != NULL ? PyDict_New() : NULL;
        PyObject* r = NULL;
        if (kw != NULL) {
            for (int i = 0; i < 3; ++i) {
                PyTuple_SET_ITEM(t, i, Py_NewRef(args[i]));
            }
            r = where_select(t, kw);
            if (r == NULL && !PyErr_Occurred()) {
                r = where_delegate(t, kw);
            }
        }
        Py_XDECREF(t);
        Py_XDECREF(kw);
        if (r != NULL || PyErr_Occurred()) {
            return r;
        }
    }
    if (g_where_func == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "viproc: setup() was not called");
        return NULL;
    }
    return PyObject_Vectorcall(g_where_func, args, (size_t)nargs, kwnames);
}

static PyMethodDef methods[] = {
    {"_issue_site", m_issue_site, METH_NOARGS, "The issue site an op issued here would get."},
    {"setup", m_setup, METH_VARARGS,
     "setup(cls, dtypes, fallback, asarray, internal_dirs, ops, ndarray, generic)"},
    {"init", m_init, METH_VARARGS,
     "init(workers=0, max_active_tasks=0, sync_below=-1, adaptive=True)"},
    {"shutdown", m_shutdown, METH_NOARGS, "Wait for all tasks and destroy the runtime."},
    {"initialized", m_initialized, METH_NOARGS, NULL},
    {"offload_counts", m_offload_counts, METH_NOARGS,
     "{'inline': n, 'offloaded': m}: ops run inline / handed to workers"},
    {"pool_stats", m_pool_stats, METH_NOARGS, "Element memory pool counters."},
    {"elided", m_elided, METH_NOARGS, "Ops that wrote into a temporary operand's buffer."},
    {"elision_mode", m_elision_mode, METH_NOARGS,
     "How temporaries are recognized: 'exact', 'backtrace' or 'off'."},
#if defined(VP_ELIDE_BACKTRACE)
    {"elision_chains", m_elision_chains, METH_NOARGS,
     "Frames per calibrated operator call chain (backtrace mode; one per operator)."},
#endif
    {"pool_trim", m_pool_trim, METH_NOARGS, "Return all cached element memory."},
    {"pool_set_limit", m_pool_set_limit, METH_O, "Bound for cached element memory (bytes)."},
    {"workers", m_workers, METH_NOARGS, NULL},
    {"active_tasks", m_active_tasks, METH_NOARGS, NULL},
    {"wait_all", m_wait_all, METH_NOARGS, NULL},
    {"from_ndarray", m_from_ndarray, METH_O, "Wrap a NumPy array (no copy)."},
    {"view", m_view, METH_VARARGS, "view(a, shape, strides, offset)"},
    {"ready", m_ready, METH_O, NULL},
    {"same_storage", m_same_storage, METH_VARARGS, NULL},
    {"wait", m_wait, METH_O, "Sync point: wait for a, raise AsyncError on failure."},
    {"export", m_export, METH_VARARGS,
     "export(a, writable=False): sync point, NumPy view of a without a copy."},
    {"assign", m_assign, METH_VARARGS, "assign(dst, src): dst[...] = src"},
    {"scalar", m_scalar, METH_O,
     "scalar(a): NumPy's scalar of a's first element if a is ready (no wait), else None"},
    {"item", (PyCFunction)(void (*)(void))m_item, METH_FASTCALL,
     "item(a, key): NumPy's scalar a[key] for an integer key if a is ready, else None"},
    {"basic_view", (PyCFunction)(void (*)(void))m_basic_view, METH_FASTCALL,
     "basic_view(a, key): a[key] for ints, slices, None, Ellipsis as a view, else None"},
    {"transpose", (PyCFunction)(void (*)(void))m_transpose, METH_FASTCALL,
     "transpose(a, axes): view, None if axes invalid"},
    {"reshape", m_reshape, METH_VARARGS,
     "reshape(a, shape, order): view as NumPy's reshape, None if NumPy would copy"},
    {"broadcast_to", m_broadcast_to, METH_VARARGS,
     "broadcast_to(a, shape): read-only broadcast view, None if not broadcastable"},
    {"gather", m_gather, METH_VARARGS,
     "gather(a, idx, idx_axes, result_axes, result_strides): a[idx...] (vp_gather)"},
    {"scatter", m_scatter, METH_VARARGS,
     "scatter(a, idx, idx_axes, result_axes, values): a[idx...] = values (vp_scatter)"},
    {"where", (PyCFunction)(void (*)(void))m_where, METH_FASTCALL | METH_KEYWORDS,
     "where(condition, [x, y]): numpy.where without NumPy's dispatch for viproc arrays"},
    {"select", m_select, METH_VARARGS, "select(cond, x, y): np.where(cond, x, y) (vp_select)"},
    {"mask_assign", m_mask_assign, METH_VARARGS, "mask_assign(a, mask, value): a[mask] = value"},
    {"matmul", m_matmul, METH_VARARGS,
     "matmul(a, b): np.matmul of two viproc arrays asynchronously, or None"},
    {"arg_reduce", m_arg_reduce, METH_VARARGS,
     "arg_reduce(argmax, a, out_shape): argmin/argmax along a's last axis (C-contiguous)"},
    {"astype", m_astype, METH_VARARGS,
     "astype(a, code, order): async cast (order 'K', 'C' or 'F'), or None if unsupported"},
    {"setup_reductions", m_setup_reductions, METH_VARARGS,
     "setup_reductions(entries, array_function, novalue, async_ok)"},
    {"reduce", m_reduce, METH_VARARGS,
     "reduce(ufunc, a, axes, keepdims, dtype, initial): async reduction, or None if unsupported"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_viproc",
    "viproc runtime binding.",
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit__viproc(void) {
    if (PyType_Ready(&ArrayType) < 0 || PyType_Ready(&UfuncProxyType) < 0) {
        return NULL;
    }
    PyObject* m = PyModule_Create(&module);
    if (m == NULL) {
        return NULL;
    }
    s_call = PyUnicode_InternFromString("__call__");
    s_out = PyUnicode_InternFromString("out");
    AsyncError = PyErr_NewExceptionWithDoc(
        "viproc._viproc.AsyncError",
        "An array op that ran asynchronously failed. Raised at the next call into "
        "viproc after the failure and at every sync point touching its results; "
        "the message names the op and where it was issued.",
        PyExc_RuntimeError, NULL);
    PyObject* bases = AsyncError != NULL ? PyTuple_Pack(2, AsyncError, PyExc_IndexError) : NULL;
    AsyncIndexError =
        bases != NULL
            ? PyErr_NewExceptionWithDoc("viproc._viproc.AsyncIndexError",
                                        "An index out of bounds in an asynchronous indexing op.",
                                        bases, NULL)
            : NULL;
    Py_XDECREF(bases);
    bases = AsyncError != NULL ? PyTuple_Pack(2, AsyncError, PyExc_ValueError) : NULL;
    AsyncValueError =
        bases != NULL
            ? PyErr_NewExceptionWithDoc("viproc._viproc.AsyncValueError",
                                        "An invalid element value in an asynchronous op (as "
                                        "NumPy's ValueError: integer power with a negative "
                                        "exponent).",
                                        bases, NULL)
            : NULL;
    Py_XDECREF(bases);
    if (s_call == NULL || s_out == NULL || AsyncError == NULL || AsyncIndexError == NULL ||
        AsyncValueError == NULL || PyModule_AddObjectRef(m, "AsyncError", AsyncError) < 0 ||
        PyModule_AddObjectRef(m, "AsyncIndexError", AsyncIndexError) < 0 ||
        PyModule_AddObjectRef(m, "AsyncValueError", AsyncValueError) < 0 ||
        PyModule_AddObjectRef(m, "_Array", (PyObject*)&ArrayType) < 0 ||
        PyModule_AddObjectRef(m, "ufunc_proxy", (PyObject*)&UfuncProxyType) < 0 ||
        PyModule_AddStringConstant(m, "version", vp_version()) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
