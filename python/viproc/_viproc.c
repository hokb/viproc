/* CPython extension module `viproc._viproc`: thin binding of the C ABI in
 * include/viproc/viproc.h. All policy (dtype resolution, fallbacks, indexing)
 * lives in the Python package; this module only moves handles across. */

#include "viproc_python.h"

#include "viproc/viproc.h"

#define ERRLEN 1024

static vp_runtime* g_rt = NULL;
static unsigned long g_main_thread = 0;
static PyObject* AsyncError = NULL;

/* --- Handle type: owns one vp_array* ------------------------------------- */

typedef struct {
    PyObject_HEAD vp_array* a;
} Handle;

static void Handle_dealloc(Handle* self) {
    if (self->a != NULL) {
        vp_array_release(self->a);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyTypeObject HandleType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "viproc._viproc.Handle",
    .tp_basicsize = sizeof(Handle),
    .tp_dealloc = (destructor)Handle_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Opaque handle of a viproc runtime array.",
};

static PyObject* new_handle(vp_array* a) {
    Handle* h = PyObject_New(Handle, &HandleType);
    if (h == NULL) {
        vp_array_release(a);
        return NULL;
    }
    h->a = a;
    return (PyObject*)h;
}

static vp_array* handle_of(PyObject* o) {
    if (!PyObject_TypeCheck(o, &HandleType)) {
        PyErr_SetString(PyExc_TypeError, "expected a viproc handle");
        return NULL;
    }
    return ((Handle*)o)->a;
}

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

/* Reads a tuple/list of ints into `out` (at most 64). Returns count or -1. */
static int read_dims(PyObject* seq, int64_t* out) {
    PyObject* fast = PySequence_Fast(seq, "expected a sequence of ints");
    if (fast == NULL) {
        return -1;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n > 64) {
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

static PyObject* dims_tuple(int n, const int64_t* d) {
    PyObject* t = PyTuple_New(n);
    for (int i = 0; t != NULL && i < n; ++i) {
        PyTuple_SET_ITEM(t, i, PyLong_FromLongLong(d[i]));
    }
    return t;
}

/* --- module functions ----------------------------------------------------- */

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
    return new_handle(a);
}

static PyObject* m_view(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *h, *shape_o, *strides_o;
    long long offset;
    if (!PyArg_ParseTuple(args, "O!OOL", &HandleType, &h, &shape_o, &strides_o, &offset) ||
        check_runtime() < 0) {
        return NULL;
    }
    int64_t shape[64], strides[64];
    int nd = read_dims(shape_o, shape);
    if (nd < 0 || read_dims(strides_o, strides) != nd) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "shape and strides differ in length");
        }
        return NULL;
    }
    return new_handle(vp_array_view(g_rt, handle_of(h), nd, shape, strides, offset));
}

static PyObject* m_info(PyObject* self, PyObject* h) {
    (void)self;
    vp_array* a = handle_of(h);
    if (a == NULL) {
        return NULL;
    }
    const int nd = vp_array_ndim(a);
    PyObject* shape = dims_tuple(nd, vp_array_shape(a));
    PyObject* strides = dims_tuple(nd, vp_array_strides(a));
    return Py_BuildValue("(NNi)", shape, strides, (int)vp_array_dtype(a));
}

static PyObject* m_ready(PyObject* self, PyObject* h) {
    (void)self;
    vp_array* a = handle_of(h);
    if (a == NULL || check_runtime() < 0) {
        return NULL;
    }
    return PyBool_FromLong(vp_array_ready(a));
}

static PyObject* m_same_storage(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *x, *y;
    if (!PyArg_ParseTuple(args, "O!O!", &HandleType, &x, &HandleType, &y)) {
        return NULL;
    }
    return PyBool_FromLong(vp_array_same_storage(handle_of(x), handle_of(y)));
}

static int wait_handle(vp_array* a) {
    char err[ERRLEN] = "";
    int status;
    Py_BEGIN_ALLOW_THREADS;
    status = vp_array_wait(g_rt, a, err, sizeof err);
    Py_END_ALLOW_THREADS;
    return raise_status(status, err);
}

static PyObject* m_wait(PyObject* self, PyObject* h) {
    (void)self;
    vp_array* a = handle_of(h);
    if (a == NULL || check_runtime() < 0 || wait_handle(a) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* m_to_ndarray(PyObject* self, PyObject* h) {
    (void)self;
    vp_array* a = handle_of(h);
    if (a == NULL || check_runtime() < 0 || wait_handle(a) < 0) {
        return NULL;
    }
    return vp_array_to_ndarray(a);
}

static PyObject* m_ufunc(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *uf, *inputs, *shape_o, *out_o;
    int out_dtype;
    const char* site;
    if (!PyArg_ParseTuple(args, "OO!iOOs", &uf, &PyTuple_Type, &inputs, &out_dtype, &shape_o,
                          &out_o, &site) ||
        check_runtime() < 0) {
        return NULL;
    }
    const Py_ssize_t nin = PyTuple_GET_SIZE(inputs);
    if (nin > 32) {
        PyErr_SetString(PyExc_NotImplementedError, "too many inputs");
        return NULL;
    }
    vp_array* ins[32];
    for (Py_ssize_t i = 0; i < nin; ++i) {
        if ((ins[i] = handle_of(PyTuple_GET_ITEM(inputs, i))) == NULL) {
            return NULL;
        }
    }
    vp_array* out = NULL;
    if (out_o != Py_None && (out = handle_of(out_o)) == NULL) {
        return NULL;
    }
    int64_t shape[64];
    int nd = read_dims(shape_o, shape);
    if (nd < 0) {
        return NULL;
    }
    char err[ERRLEN] = "";
    vp_array* result = NULL;
    int status = vp_ufunc(g_rt, uf, (int)nin, ins, (vp_dtype)out_dtype, nd, shape, out, &result,
                          site, err, sizeof err);
    if (raise_status(status, err) < 0) {
        return NULL;
    }
    if (result == NULL) {
        Py_RETURN_NONE;
    }
    return new_handle(result);
}

static PyObject* m_assign(PyObject* self, PyObject* args) {
    (void)self;
    PyObject *dst, *src;
    const char* site;
    if (!PyArg_ParseTuple(args, "O!O!s", &HandleType, &dst, &HandleType, &src, &site) ||
        check_runtime() < 0) {
        return NULL;
    }
    char err[ERRLEN] = "";
    int status = vp_assign(g_rt, handle_of(dst), handle_of(src), site, err, sizeof err);
    if (raise_status(status, err) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"init", m_init, METH_VARARGS, "init(workers=0, max_active_tasks=0)"},
    {"shutdown", m_shutdown, METH_NOARGS, "Wait for all tasks and destroy the runtime."},
    {"initialized", m_initialized, METH_NOARGS, NULL},
    {"workers", m_workers, METH_NOARGS, NULL},
    {"active_tasks", m_active_tasks, METH_NOARGS, NULL},
    {"wait_all", m_wait_all, METH_NOARGS, NULL},
    {"from_ndarray", m_from_ndarray, METH_O, "Wrap a NumPy array (no copy)."},
    {"view", m_view, METH_VARARGS, "view(h, shape, strides, offset)"},
    {"info", m_info, METH_O, "info(h) -> (shape, strides, dtype_code)"},
    {"ready", m_ready, METH_O, NULL},
    {"same_storage", m_same_storage, METH_VARARGS, NULL},
    {"wait", m_wait, METH_O, "Sync point: wait for h, raise AsyncError on failure."},
    {"to_ndarray", m_to_ndarray, METH_O, "Sync point: copy of h as a NumPy array."},
    {"ufunc", m_ufunc, METH_VARARGS, "ufunc(uf, inputs, out_dtype, out_shape, out, site)"},
    {"assign", m_assign, METH_VARARGS, "assign(dst, src, site)"},
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
    if (PyType_Ready(&HandleType) < 0) {
        return NULL;
    }
    PyObject* m = PyModule_Create(&module);
    if (m == NULL) {
        return NULL;
    }
    AsyncError = PyErr_NewExceptionWithDoc(
        "viproc._viproc.AsyncError",
        "An array op that ran asynchronously failed. Raised at the next sync "
        "point touching its result; the message names where the op was issued.",
        PyExc_RuntimeError, NULL);
    if (AsyncError == NULL || PyModule_AddObjectRef(m, "AsyncError", AsyncError) < 0 ||
        PyModule_AddObjectRef(m, "Handle", (PyObject*)&HandleType) < 0 ||
        PyModule_AddStringConstant(m, "version", vp_version()) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
