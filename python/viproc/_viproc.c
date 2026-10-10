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

#define ERRLEN 1024
#define MAXDIMS 64
#define MAXIN 3 /* element-wise ufuncs have at most 3 inputs */

static vp_runtime* g_rt = NULL;
static unsigned long g_main_thread = 0;
static PyObject* AsyncError = NULL;

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
static PyObject* g_ufunc_info = NULL; /* ufunc -> (nin, nout, is_gufunc, {sigkey: resolved}) */
/* Code objects are keyed by address: hashing a code object hashes its whole
 * bytecode and constants. Each g_code_internal entry also holds the code
 * object, so its address cannot be reused while it is cached. */
static PyObject* g_code_internal = NULL; /* id(code) -> (code, is_internal) */
static PyObject* g_sites = NULL;         /* (id(code), lasti) -> interned site (per runtime) */

static PyObject* s_call = NULL; /* "__call__" */
static PyObject* s_out = NULL;  /* "out" */

/* --- _Array: base class of viproc.ndarray ---------------------------------- */

typedef struct {
    PyObject_HEAD vp_array* a;
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

static PyGetSetDef Array_getset[] = {
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
        PyErr_SetString(AsyncError, err);
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
        PyObject* id = PyLong_FromVoidPtr(code);
        PyObject* entry = id != NULL ? PyDict_GetItemWithError(g_code_internal, id) : NULL;
        PyObject* internal = entry != NULL ? PyTuple_GET_ITEM(entry, 1) : NULL;
        if (internal == NULL && id != NULL) {
            PyErr_Clear();
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
            internal = is_internal ? Py_True : Py_False;
            PyObject* e = PyTuple_Pack(2, (PyObject*)code, internal);
            if (e != NULL) {
                PyDict_SetItem(g_code_internal, id, e);
                Py_DECREF(e);
            }
        }
        if (internal == Py_False) {
            const char* site = NULL;
            PyObject* key = Py_BuildValue("(Oi)", id, PyFrame_GetLasti(f));
            PyObject* hit = key != NULL ? PyDict_GetItemWithError(g_sites, key) : NULL;
            if (hit != NULL) {
                site = (const char*)PyLong_AsVoidPtr(hit);
            } else if (key != NULL && !PyErr_Occurred()) {
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
                    PyObject* v = PyLong_FromVoidPtr((void*)site);
                    if (v != NULL) {
                        PyDict_SetItem(g_sites, key, v);
                        Py_DECREF(v);
                    }
                }
            }
            Py_XDECREF(key);
            Py_XDECREF(id);
            Py_DECREF(code);
            Py_DECREF(f);
            PyErr_Clear();
            return site != NULL ? site : vp_intern(g_rt, "<unknown>");
        }
        Py_XDECREF(id);
        Py_DECREF(code);
        PyFrameObject* back = PyFrame_GetBack(f); /* new ref */
        Py_DECREF(f);
        f = back;
    }
    return vp_intern(g_rt, "<unknown>");
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

/* Per-ufunc info: nin, nout, is gufunc, signature cache. Borrowed tuple. */
static PyObject* ufunc_info(PyObject* ufunc) {
    PyObject* info = PyDict_GetItemWithError(g_ufunc_info, ufunc);
    if (info != NULL || PyErr_Occurred()) {
        return info;
    }
    PyObject* nin = PyObject_GetAttrString(ufunc, "nin");
    PyObject* nout = PyObject_GetAttrString(ufunc, "nout");
    PyObject* sig = PyObject_GetAttrString(ufunc, "signature");
    if (nin == NULL || nout == NULL || sig == NULL) {
        Py_XDECREF(nin);
        Py_XDECREF(nout);
        Py_XDECREF(sig);
        return NULL;
    }
    info = Py_BuildValue("(NNON)", nin, nout, sig != Py_None ? Py_True : Py_False, PyDict_New());
    Py_DECREF(sig);
    if (info == NULL || PyDict_SetItem(g_ufunc_info, ufunc, info) < 0) {
        Py_XDECREF(info);
        return NULL;
    }
    Py_DECREF(info); /* owned by the cache */
    return info;
}

/* Resolved dtype codes for (ufunc, kinds): out[0..nin) inputs, out[nin] the
 * output, plus NumPy's loop for them. Returns 1 if supported, 0 if not, -1 on
 * error. Cached per runtime (the loop handles belong to it). */
static int resolve(PyObject* ufunc, PyObject* sigcache, int nin, const int* kinds, int* out,
                   const vp_loop** loop) {
    unsigned long long key = 0;
    for (int i = 0; i < nin; ++i) {
        key |= (unsigned long long)kinds[i] << (8 * i);
    }
    PyObject* k = PyLong_FromUnsignedLongLong(key);
    if (k == NULL) {
        return -1;
    }
    PyObject* hit = PyDict_GetItemWithError(sigcache, k);
    long long packed;
    *loop = NULL;
    if (hit != NULL) {
        packed = PyLong_AsLongLong(PyTuple_GET_ITEM(hit, 0));
        *loop = (const vp_loop*)PyLong_AsVoidPtr(PyTuple_GET_ITEM(hit, 1));
    } else if (PyErr_Occurred()) {
        Py_DECREF(k);
        return -1;
    } else {
        packed = -1;
        PyObject* spec = PyTuple_New(nin + 1);
        for (int i = 0; i < nin; ++i) {
            PyTuple_SET_ITEM(spec, i, Py_NewRef(kind_spec(kinds[i])));
        }
        PyTuple_SET_ITEM(spec, nin, Py_NewRef(Py_None));
        PyObject* name = PyUnicode_InternFromString("resolve_dtypes");
        PyObject* res = PyObject_CallMethodOneArg(ufunc, name, spec);
        Py_DECREF(name);
        Py_DECREF(spec);
        if (res == NULL) {
            PyErr_Clear(); /* no loop / promotion error: NumPy raises it in the fallback */
        } else {
            long long p = 0;
            int ok = 1;
            for (int i = 0; i <= nin && ok; ++i) {
                int c = dtype_code(PyTuple_GET_ITEM(res, i));
                /* Typed inputs must keep their dtype: anything else is a cast. */
                ok = c >= 0 && (i == nin || kinds[i] >= K_PYBOOL || kinds[i] == c);
                p |= (long long)c << (8 * i);
            }
            Py_DECREF(res);
            packed = ok ? p : -1;
        }
        if (packed >= 0) {
            vp_dtype dts[MAXIN + 1];
            for (int i = 0; i <= nin; ++i) {
                dts[i] = (vp_dtype)((packed >> (8 * i)) & 0xff);
            }
            char err[ERRLEN];
            if (vp_ufunc_loop(g_rt, ufunc, nin, dts, loop, err, sizeof err) != VP_OK) {
                packed = -1; /* no exact NumPy loop (or it needs the Python API) */
                *loop = NULL;
            }
        }
        PyObject* v = Py_BuildValue("(LN)", packed, PyLong_FromVoidPtr((void*)*loop));
        if (v == NULL || PyDict_SetItem(sigcache, k, v) < 0) {
            Py_XDECREF(v);
            Py_DECREF(k);
            return -1;
        }
        Py_DECREF(v);
    }
    Py_DECREF(k);
    if (packed < 0) {
        return 0;
    }
    for (int i = 0; i <= nin; ++i) {
        out[i] = (int)((packed >> (8 * i)) & 0xff);
    }
    return 1;
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
                      PyObject** result) {
    if (g_rt == NULL || g_array_cls == NULL || n > MAXIN ||
        PyThread_get_thread_ident() != g_main_thread) {
        return 0;
    }
    PyObject* info = ufunc_info(ufunc);
    if (info == NULL) {
        PyErr_Clear();
        return 0;
    }
    const long nin = PyLong_AsLong(PyTuple_GET_ITEM(info, 0));
    const long nout = PyLong_AsLong(PyTuple_GET_ITEM(info, 1));
    if (nin != n || nout != 1 || PyTuple_GET_ITEM(info, 2) == Py_True) {
        return 0;
    }
    int kinds[MAXIN];
    for (Py_ssize_t i = 0; i < n; ++i) {
        if ((kinds[i] = input_kind(inputs[i])) < 0) {
            return 0;
        }
    }
    int codes[MAXIN + 1];
    const vp_loop* loop = NULL;
    int r = resolve(ufunc, PyTuple_GET_ITEM(info, 3), (int)n, kinds, codes, &loop);
    if (r <= 0) {
        return r;
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
        if (!same) {
            status = 1; /* casting or broadcasting into out=: fallback */
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
        int r = fast_ufunc(args[0], inputs, n, out, &result);
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
    int r = fast_ufunc(ufunc, args, 2, NULL, &result);
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
    int r = fast_ufunc(ufunc, args, 2, a, &result);
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
    int r = fast_ufunc(ufunc, &a, 1, NULL, &result);
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
    if (!PyArg_ParseTuple(args, "|nn", &workers, &max_active)) {
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
    Py_RETURN_NONE;
}

static PyObject* m_shutdown(PyObject* self, PyObject* noargs) {
    (void)self;
    (void)noargs;
    if (g_rt != NULL) {
        vp_runtime* rt = g_rt;
        g_rt = NULL;
        PyDict_Clear(g_sites);      /* interned in the runtime being destroyed */
        PyDict_Clear(g_ufunc_info); /* holds its loop handles */
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
    if (!PyArg_ParseTuple(args, "O!OOL", &ArrayType, &base, &shape_o, &strides_o, &offset) ||
        check_runtime() < 0) {
        return NULL;
    }
    int64_t shape[MAXDIMS], strides[MAXDIMS];
    int nd = read_dims(shape_o, shape);
    if (nd < 0 || read_dims(strides_o, strides) != nd) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "shape and strides differ in length");
        }
        return NULL;
    }
    return wrap_array(vp_array_view(g_rt, array_of(base), nd, shape, strides, offset));
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

static PyObject* m_to_ndarray(PyObject* self, PyObject* o) {
    (void)self;
    vp_array* a = array_of(o);
    if (a == NULL || check_runtime() < 0 || wait_array(a) < 0) {
        return NULL;
    }
    return vp_array_to_ndarray(a);
}

static PyObject* m_assign(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *dst, *src;
    if (!PyArg_ParseTuple(args, "O!O!", &ArrayType, &dst, &ArrayType, &src) ||
        check_runtime() < 0) {
        return NULL;
    }
    char err[ERRLEN] = "";
    int status = vp_assign(g_rt, array_of(dst), array_of(src), issue_site(), err, sizeof err);
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
    {"init", m_init, METH_VARARGS, "init(workers=0, max_active_tasks=0)"},
    {"shutdown", m_shutdown, METH_NOARGS, "Wait for all tasks and destroy the runtime."},
    {"initialized", m_initialized, METH_NOARGS, NULL},
    {"workers", m_workers, METH_NOARGS, NULL},
    {"active_tasks", m_active_tasks, METH_NOARGS, NULL},
    {"wait_all", m_wait_all, METH_NOARGS, NULL},
    {"from_ndarray", m_from_ndarray, METH_O, "Wrap a NumPy array (no copy)."},
    {"view", m_view, METH_VARARGS, "view(a, shape, strides, offset)"},
    {"ready", m_ready, METH_O, NULL},
    {"same_storage", m_same_storage, METH_VARARGS, NULL},
    {"wait", m_wait, METH_O, "Sync point: wait for a, raise AsyncError on failure."},
    {"to_ndarray", m_to_ndarray, METH_O, "Sync point: copy of a as a NumPy array."},
    {"assign", m_assign, METH_VARARGS, "assign(dst, src): dst[...] = src"},
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
    g_ufunc_info = PyDict_New();
    g_code_internal = PyDict_New();
    g_sites = PyDict_New();
    s_call = PyUnicode_InternFromString("__call__");
    s_out = PyUnicode_InternFromString("out");
    AsyncError = PyErr_NewExceptionWithDoc(
        "viproc._viproc.AsyncError",
        "An array op that ran asynchronously failed. Raised at the next sync "
        "point touching its result; the message names where the op was issued.",
        PyExc_RuntimeError, NULL);
    if (g_ufunc_info == NULL || g_code_internal == NULL || g_sites == NULL || s_call == NULL ||
        s_out == NULL || AsyncError == NULL ||
        PyModule_AddObjectRef(m, "AsyncError", AsyncError) < 0 ||
        PyModule_AddObjectRef(m, "_Array", (PyObject*)&ArrayType) < 0 ||
        PyModule_AddStringConstant(m, "version", vp_version()) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
