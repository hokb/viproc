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

/* Registered by _ndarray.py via _viproc.setup(). */
static PyTypeObject* g_array_cls = NULL; /* viproc.ndarray */
static PyObject* g_dtypes[7];            /* np.dtype per vp_dtype code */
static PyObject* g_fallback = NULL;      /* fallback(ufunc, method, inputs, kwargs) */
static PyObject* g_asarray = NULL;       /* asarray(ndarray) -> viproc.ndarray */
static PyObject* g_internal_dirs = NULL; /* tuple of path prefixes skipped for issue sites */
static PyObject* g_np_ndarray = NULL;    /* numpy.ndarray */
static PyObject* g_np_generic = NULL;    /* numpy.generic */
static PyObject* g_ops = NULL;           /* dict: operator name -> ufunc */

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

static int is_array(PyObject* o) { return PyObject_TypeCheck(o, &ArrayType); }

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

static PyGetSetDef Array_getset[] = {
    {"_readonly", (getter)Array_get_readonly, (setter)Array_set_readonly,
     "True if writes raise, as for a NumPy array with WRITEABLE == False.", NULL},
    {"shape", (getter)Array_get_shape, NULL, "Shape (no sync).", NULL},
    {"strides", (getter)Array_get_strides, NULL, "Byte strides (no sync).", NULL},
    {"ndim", (getter)Array_get_ndim, NULL, "Number of dimensions (no sync).", NULL},
    {"dtype", (getter)Array_get_dtype, NULL, "Element type (no sync).", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

/* --- helpers -------------------------------------------------------------- */

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
    return 0;
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
        /* Kernels report index errors as "IndexError: ...": raise them as an
         * IndexError too, as NumPy would have. */
        PyErr_SetString(strncmp(err, "IndexError: ", 12) == 0 ? AsyncIndexError : AsyncError, err);
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

/* Issue site: "file:line in function" of the first frame outside viproc and
 * NumPy, interned in the runtime (vp_intern) and cached per (code object,
 * bytecode offset). The offset, unlike the line number, is free to read: the
 * line is computed only on a cache miss. Never NULL. */
static const char* issue_site(void) {
    PyFrameObject* f = PyEval_GetFrame(); /* borrowed */
    Py_XINCREF(f);
    while (f != NULL) {
        PyCodeObject* code = PyFrame_GetCode(f); /* new ref */
        const Entry* flag = t_find(&g_code_flags, code, 0);
        int is_internal;
        if (flag != NULL) {
            is_internal = (int)flag->v;
        } else {
            PyObject* fn = PyObject_GetAttrString((PyObject*)code, "co_filename");
            is_internal = 0;
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
        }
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
#define MAX_CHAINS 32
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

static void calibrate_operator_chains(void) {
    g_chains_n = 0;
    call_chain run1[MAX_CHAINS], run2[MAX_CHAINS], run3[MAX_CHAINS];
    int n1 = 0, n2 = 0, n3 = 0;
    PyObject* globals = PyDict_New();
    PyObject* probe =
        PyType_Ready(&ProbeType) == 0 ? PyObject_CallNoArgs((PyObject*)&ProbeType) : NULL;
    if (globals != NULL && probe != NULL && PyDict_SetItemString(globals, "p", probe) == 0 &&
        PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins()) == 0) {
#define VP_PROBE_OPS                                                                               \
    "p + p; p - p; p * p; p / p; p // p; p % p\n"                                                  \
    "p << p; p >> p; p & p; p | p; p ^ p; p @ p\n"                                                 \
    "-p; +p; ~p\n"
        g_recording = run1; /* entered through module code (PyEval_EvalCode) */
        g_recorded = 0;
        PyObject* r = PyRun_String(VP_PROBE_OPS "def f(p):\n"
                                                "    p + p; p - p; p * p; p / p; p // p; p % p\n"
                                                "    p << p; p >> p; p & p; p | p; p ^ p; p @ p\n"
                                                "    -p; +p; ~p\n",
                                   Py_file_input, globals, globals);
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
        r = PyRun_String("import operator as o\n"
                         "o.add(p, p); o.sub(p, p); o.mul(p, p); o.truediv(p, p)\n"
                         "o.floordiv(p, p); o.mod(p, p); o.lshift(p, p); o.rshift(p, p)\n"
                         "o.and_(p, p); o.or_(p, p); o.xor(p, p); o.matmul(p, p)\n"
                         "o.neg(p); o.pos(p); o.invert(p)\n",
                         Py_file_input, globals, globals);
        n3 = g_recorded;
        Py_XDECREF(r);
        g_recording = NULL;
#undef VP_PROBE_OPS
    }
    PyErr_Clear();
    Py_XDECREF(probe);
    Py_XDECREF(globals);
    if (n1 != n2 || n3 != n1) {
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

/* Issues ufunc(*inputs, out=out) asynchronously. Returns 1 and sets *result
 * (new reference), 0 if the fast path does not handle the call, -1 on a
 * Python error. `out` is NULL or a viproc array. */
static int fast_ufunc(PyObject* ufunc, PyObject* const* inputs, Py_ssize_t n, PyObject* out,
                      int elide, PyObject** result) {
    if (g_rt == NULL || g_array_cls == NULL || n > MAXIN ||
        PyThread_get_thread_ident() != g_main_thread) {
        return 0;
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

static PyObject* op_ufunc(const char* name) {
    PyObject* u = g_ops != NULL ? PyDict_GetItemString(g_ops, name) : NULL;
    if (u == NULL) {
        PyErr_Format(PyExc_RuntimeError, "viproc: ufunc for %s not registered", name);
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

static PyObject* binop(const char* name, PyObject* a, PyObject* b) {
    if (defers(is_array(a) ? b : a)) {
        Py_RETURN_NOTIMPLEMENTED;
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

static PyObject* inplace_op(const char* name, PyObject* a, PyObject* b) {
    if (defers(b)) {
        Py_RETURN_NOTIMPLEMENTED;
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

static PyObject* unop(const char* name, PyObject* a) {
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

#define BINOP(fn, name)                                                                            \
    static PyObject* fn(PyObject* a, PyObject* b) { return binop(name, a, b); }
#define INPLACE(fn, name)                                                                          \
    static PyObject* fn(PyObject* a, PyObject* b) { return inplace_op(name, a, b); }
#define UNOP(fn, name)                                                                             \
    static PyObject* fn(PyObject* a) { return unop(name, a); }

BINOP(nb_add, "add")
BINOP(nb_sub, "subtract")
BINOP(nb_mul, "multiply")
BINOP(nb_truediv, "true_divide")
BINOP(nb_floordiv, "floor_divide")
BINOP(nb_rem, "remainder")
BINOP(nb_divmod, "divmod")
BINOP(nb_lshift, "left_shift")
BINOP(nb_rshift, "right_shift")
BINOP(nb_and, "bitwise_and")
BINOP(nb_or, "bitwise_or")
BINOP(nb_xor, "bitwise_xor")
BINOP(nb_matmul, "matmul")
INPLACE(nb_iadd, "add")
INPLACE(nb_isub, "subtract")
INPLACE(nb_imul, "multiply")
INPLACE(nb_itruediv, "true_divide")
INPLACE(nb_ifloordiv, "floor_divide")
INPLACE(nb_irem, "remainder")
INPLACE(nb_ilshift, "left_shift")
INPLACE(nb_irshift, "right_shift")
INPLACE(nb_iand, "bitwise_and")
INPLACE(nb_ior, "bitwise_or")
INPLACE(nb_ixor, "bitwise_xor")
INPLACE(nb_imatmul, "matmul")
UNOP(nb_neg, "negative")
UNOP(nb_pos, "positive")
UNOP(nb_abs, "absolute")
UNOP(nb_invert, "invert")

static PyObject* nb_pow(PyObject* a, PyObject* b, PyObject* mod) {
    if (mod != Py_None) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return binop("power", a, b);
}

static PyObject* nb_ipow(PyObject* a, PyObject* b, PyObject* mod) {
    if (mod != Py_None) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return inplace_op("power", a, b);
}

static PyObject* Array_richcompare(PyObject* a, PyObject* b, int op) {
    static const char* names[] = {"less",      "less_equal", "equal",
                                  "not_equal", "greater",    "greater_equal"};
    return binop(names[op], a, b);
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

static PyMethodDef Array_methods[] = {
    {"__array_ufunc__", (PyCFunction)(void (*)(void))Array_array_ufunc,
     METH_FASTCALL | METH_KEYWORDS, "NEP 13: element-wise ufuncs run asynchronously."},
    {NULL, NULL, 0, NULL},
};

static PyTypeObject ArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "viproc._viproc._Array",
    .tp_basicsize = sizeof(ArrayObject),
    .tp_dealloc = (destructor)Array_dealloc,
    .tp_as_number = &Array_as_number,
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
    PyObject *cls, *dtypes, *fallback, *asarray, *dirs, *ops, *ndarray, *generic;
    if (!PyArg_ParseTuple(args, "O!O!OOO!O!OO", &PyType_Type, &cls, &PyTuple_Type, &dtypes,
                          &fallback, &asarray, &PyTuple_Type, &dirs, &PyDict_Type, &ops, &ndarray,
                          &generic)) {
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
    Py_XSETREF(g_ops, Py_NewRef(ops));
    Py_XSETREF(g_np_ndarray, Py_NewRef(ndarray));
    Py_XSETREF(g_np_generic, Py_NewRef(generic));
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
    return Py_BuildValue("{s:K,s:K}", "inline", (unsigned long long)inl, "offloaded",
                         (unsigned long long)off);
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
        vp_runtime* rt = g_rt;
        g_rt = NULL;
        t_clear(&g_sites); /* interned in the runtime being destroyed */
        t_clear(&g_sigs);  /* holds its loop handles */
        vp_runtime_destroy(rt);
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

static PyObject* m_stand_in(PyObject* self, PyObject* a) {
    (void)self;
    if (!PyObject_TypeCheck(a, &ArrayType) || check_runtime() < 0) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "stand_in: a viproc array expected");
        }
        return NULL;
    }
    return vp_array_stand_in(array_of(a));
}

static PyObject* m_view_of(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *a, *fake, *r;
    if (!PyArg_ParseTuple(args, "O!OO", &ArrayType, &a, &fake, &r) || check_runtime() < 0) {
        return NULL;
    }
    int readonly = 0;
    vp_array* v = vp_array_view_of(g_rt, array_of(a), fake, r, &readonly);
    if (v == NULL) {
        Py_RETURN_NONE;
    }
    PyObject* w = wrap_array(v);
    if (w != NULL) {
        ((ArrayObject*)w)->readonly = ((ArrayObject*)a)->readonly || readonly;
    }
    return w;
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
    {"stand_in", m_stand_in, METH_O,
     "stand_in(a): ndarray with a's layout over a dummy address (metadata-only NumPy code)"},
    {"view_of", m_view_of, METH_VARARGS,
     "view_of(a, stand_in, r): r (a view of stand_in) as a view of a, or None"},
    {"gather", m_gather, METH_VARARGS,
     "gather(a, idx, idx_axes, result_axes, result_strides): a[idx...] (vp_gather)"},
    {"scatter", m_scatter, METH_VARARGS,
     "scatter(a, idx, idx_axes, result_axes, values): a[idx...] = values (vp_scatter)"},
    {"mask_assign", m_mask_assign, METH_VARARGS, "mask_assign(a, mask, value): a[mask] = value"},
    {"arg_reduce", m_arg_reduce, METH_VARARGS,
     "arg_reduce(argmax, a, out_shape): argmin/argmax along a's last axis (C-contiguous)"},
    {"astype", m_astype, METH_VARARGS,
     "astype(a, code, order): async cast (order 'K' or 'C'), or None if unsupported"},
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
    if (PyType_Ready(&ArrayType) < 0) {
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
        "An array op that ran asynchronously failed. Raised at the next sync "
        "point touching its result; the message names where the op was issued.",
        PyExc_RuntimeError, NULL);
    PyObject* bases = AsyncError != NULL ? PyTuple_Pack(2, AsyncError, PyExc_IndexError) : NULL;
    AsyncIndexError =
        bases != NULL
            ? PyErr_NewExceptionWithDoc("viproc._viproc.AsyncIndexError",
                                        "An index out of bounds in an asynchronous indexing op.",
                                        bases, NULL)
            : NULL;
    Py_XDECREF(bases);
    if (s_call == NULL || s_out == NULL || AsyncError == NULL || AsyncIndexError == NULL ||
        PyModule_AddObjectRef(m, "AsyncError", AsyncError) < 0 ||
        PyModule_AddObjectRef(m, "AsyncIndexError", AsyncIndexError) < 0 ||
        PyModule_AddObjectRef(m, "_Array", (PyObject*)&ArrayType) < 0 ||
        PyModule_AddStringConstant(m, "version", vp_version()) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
