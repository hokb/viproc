/* Public C ABI of the viproc runtime. Language bridges (Python first) use only
 * this header; no C++ types cross it.
 *
 * Threading: everything except vp_version() must be called from the one
 * thread that created the runtime (the "main thread"). Functions marked
 * [GIL] need the Python GIL; functions marked [no GIL] may be called with the
 * GIL released (and should be, since they can block). */
#ifndef VIPROC_VIPROC_H
#define VIPROC_VIPROC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Same type as Python.h's PyObject. */
typedef struct _object PyObject;

typedef struct vp_runtime vp_runtime;
typedef struct vp_array vp_array;

typedef enum vp_dtype {
    VP_BOOL,
    VP_INT32,
    VP_INT64,
    VP_FLOAT32,
    VP_FLOAT64,
    VP_COMPLEX64,
    VP_COMPLEX128,
} vp_dtype;

/* Status codes. On failure, functions taking `err`/`errlen` write a message. */
enum {
    VP_OK = 0,
    VP_ERROR = -1,       /* failure of an asynchronous task (raised at sync) */
    VP_UNSUPPORTED = -2, /* op/dtype/layout not supported: fall back to NumPy */
    VP_INVALID = -3,     /* invalid arguments */
};

/* Returns the runtime version as "MAJOR.MINOR.PATCH". Any thread. */
const char* vp_version(void);

/* --- Runtime ------------------------------------------------------------- */

/* [GIL] Creates the runtime. workers == 0: one per hardware thread.
 * max_active_tasks == 0: default look-ahead limit (1000). NumPy must be
 * importable. Returns NULL on failure. */
vp_runtime* vp_runtime_create(size_t workers, size_t max_active_tasks, char* err, size_t errlen);
/* [no GIL] Waits until no task is active. */
void vp_wait_all(vp_runtime* rt);
/* [GIL] Waits for all tasks and destroys the runtime. Arrays still alive
 * afterwards must only be released. */
void vp_runtime_destroy(vp_runtime* rt);
size_t vp_worker_count(const vp_runtime* rt);
size_t vp_active_tasks(const vp_runtime* rt);

/* --- Arrays -------------------------------------------------------------- */

/* [GIL] A ready array over the memory of a NumPy array (no copy). The
 * runtime keeps `ndarray` alive while its memory is in use. The caller must
 * not modify the ndarray's memory afterwards. */
int vp_array_from_ndarray(vp_runtime* rt, PyObject* ndarray, vp_array** out, char* err,
                          size_t errlen);
/* [GIL] A view of `base` sharing its storage. `offset` is in bytes, relative
 * to base's first element; strides are in bytes. Metadata only. */
vp_array* vp_array_view(vp_runtime* rt, const vp_array* base, int ndim, const int64_t* shape,
                        const int64_t* strides, int64_t offset);
/* [GIL] Releases the handle. Pending work continues. */
void vp_array_release(vp_array* a);

/* Metadata, available immediately (also for pending arrays). */
int vp_array_ndim(const vp_array* a);
const int64_t* vp_array_shape(const vp_array* a);
const int64_t* vp_array_strides(const vp_array* a);
vp_dtype vp_array_dtype(const vp_array* a);
/* 1 if the data is final (no pending task writes it). */
int vp_array_ready(vp_array* a);
/* 1 if both arrays share their storage (base and views). */
int vp_array_same_storage(const vp_array* a, const vp_array* b);

/* [no GIL] Sync point: waits until `a` is ready. Returns VP_ERROR with the
 * failing op's message and issue site if a task computing it failed. */
int vp_array_wait(vp_runtime* rt, vp_array* a, char* err, size_t errlen);
/* [GIL] New NumPy array with a copy of the data. `a` must be ready. */
PyObject* vp_array_to_ndarray(const vp_array* a);

/* --- Ops (all [GIL], all asynchronous) ------------------------------------ */

/* Issues `ufunc(inputs...)` with one output, using NumPy's loop for exactly
 * the dtypes (inputs..., out_dtype); no casting. Inputs broadcast to
 * out_shape.
 *  - out == NULL: creates the result array (*result).
 *  - out != NULL: writes into `out` (in/out semantics, `out=` / `a += b`);
 *    *result is set to NULL.
 * Returns VP_UNSUPPORTED if no matching loop exists or it needs the Python
 * API; the caller then falls back to NumPy. `site` names the issuing source
 * location; it is reported with errors. */
int vp_ufunc(vp_runtime* rt, PyObject* ufunc, int nin, vp_array* const* inputs, vp_dtype out_dtype,
             int out_ndim, const int64_t* out_shape, vp_array* out, vp_array** result,
             const char* site, char* err, size_t errlen);

/* Issues dst[...] = src (src broadcasts to dst's shape, same dtype). */
int vp_assign(vp_runtime* rt, vp_array* dst, vp_array* src, const char* site, char* err,
              size_t errlen);

#ifdef __cplusplus
}
#endif

#endif /* VIPROC_VIPROC_H */
