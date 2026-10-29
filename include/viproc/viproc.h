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

/* Same type as Python.h's PyObject (declared here only if Python.h is not
 * included yet; include Python.h first, if at all). */
#ifndef Py_PYTHON_H
typedef struct _object PyObject;
#endif

typedef struct vp_runtime vp_runtime;
typedef struct vp_array vp_array;
typedef struct vp_loop vp_loop;

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
    VP_PYERR = -4,       /* a Python exception is set */
};

/* Returns the runtime version as "MAJOR.MINOR.PATCH". Any thread. */
const char* vp_version(void);

/* --- Runtime ------------------------------------------------------------- */

/* [GIL] Creates the runtime. workers == 0: one per physical core
 * (SMT siblings count once).
 * max_active_tasks == 0: default look-ahead limit (1000). NumPy must be
 * importable. Returns NULL on failure. */
vp_runtime* vp_runtime_create(size_t workers, size_t max_active_tasks, char* err, size_t errlen);
/* [no GIL] Waits until no task is active. */
void vp_wait_all(vp_runtime* rt);
/* [GIL] Waits for all tasks and destroys the runtime. Arrays still alive
 * afterwards must only be released. */
void vp_runtime_destroy(vp_runtime* rt);
size_t vp_worker_count(const vp_runtime* rt);
/* [any thread] Physical cores this process may run on (SMT siblings once,
 * affinity). The default worker count is one less, at least 1. */
size_t vp_physical_core_count(void);
/* [main thread] NumPy time above which a small ready op of a timed loop
 * takes the task path, in ns (OffloadOptions::direct_budget_ns): 10 us
 * unless set; the Python bridge scales it to the machine. */
double vp_direct_budget_ns(const vp_runtime* rt);
void vp_set_direct_budget_ns(vp_runtime* rt, double ns);

/* Asynchronous error reporting (docs/design/error-handling.md). An op that
 * fails on any thread records its error in the runtime, the one place errors
 * live (arrays and tasks only refer to it). Bridges raise it in their main
 * thread as soon as possible: `fn` is called on the failing thread, without
 * locks or the GIL, when the runtime goes from no unreported error to one
 * (Python: Py_AddPendingCall); every call into viproc checks too.
 * [main thread, before issuing ops] */
void vp_set_error_notify(vp_runtime* rt, void (*fn)(void*), void* arg);
/* [main thread] 1 if a failed op's error waits to be reported (one atomic
 * load; nothing else). */
int vp_error_pending(const vp_runtime* rt);
/* [main thread] VP_ERROR with the error of the earliest issued op among those
 * that failed and were not reported yet ("<message> (in <op>, issued at
 * <site>)"; a message starting with "TypeName: " names the Python exception
 * type), VP_OK if there is none. Clears all pending reports. Errors raised
 * at a sync point (vp_array_wait, vp_array_prepare_write) are not reported
 * again here. */
int vp_take_error(vp_runtime* rt, char* err, size_t errlen);

/* [main thread] NumPy's errstate for the ops issued from now on: three bits
 * per category (NumPy's UFUNC_ERR_* modes: 0 ignore, 1 warn, 2 raise, 3
 * call, 4 print, 5 log), divide in bits 0-2, over 3-5, under 6-8, invalid
 * 9-11; `ctx` names the bridge's call/log object (0: none). Ops record their
 * FP errors under it (docs/design/error-handling.md, E8): raise fails the
 * op (a "FloatingPointError: ..." error), the other modes queue an event,
 * reported like an error (vp_error_pending, the error notifier). */
void vp_set_fp_mode(vp_runtime* rt, int mode, int ctx);
/* [main thread, GIL] Takes the queued FP events, in issue order: *out gets a
 * new list of tuples (report bits, status bits, mode, ctx, name, issue site);
 * FPE bits as NumPy's: 1 divide, 2 over, 4 under, 8 invalid. Returns their
 * number, or -1 with a Python error. */
int vp_take_fp_events(vp_runtime* rt, PyObject** out);

/* Dispatch policy for ops whose inputs are all ready (call before issuing):
 * ops with fewer than `sync_below` output elements run inline on the calling
 * thread; above that, `adaptive` != 0 lets each issue site switch to inline
 * execution while its results are usually waited on right after issuing.
 * adaptive == 0 gives a deterministic schedule. */
void vp_set_offload_policy(vp_runtime* rt, int64_t sync_below, int adaptive);
/* Number of ops the policy ran inline / handed to workers so far. */
void vp_offload_counts(const vp_runtime* rt, uint64_t* inline_ops, uint64_t* offloaded_ops);
/* Ops that ran directly on the main thread, without a task (small, all
 * inputs ready; Runtime::issue). */
uint64_t vp_direct_count(const vp_runtime* rt);
/* Tasks issued so far (main thread): every op that did not run directly,
 * whether its inputs were ready or pending, and copy-on-write copies. */
uint64_t vp_task_count(const vp_runtime* rt);
/* Sync points so far that found their array pending, so the main thread
 * blocked (vp_array_wait, vp_array_prepare_write, ...; vp_wait_all is not
 * counted). Waits on ready arrays (to read their error) are no syncs. */
uint64_t vp_sync_count(const vp_runtime* rt);
size_t vp_active_tasks(const vp_runtime* rt);

/* --- Element memory pool (process-wide; docs/design/memory-management.md) */

typedef struct {
    uint64_t hits;    /* requests served from the pool */
    uint64_t misses;  /* requests that allocated from the system */
    uint64_t cached;  /* bytes held by the pool now */
    uint64_t dropped; /* releases returned to the system (bound reached) */
    uint64_t limit;   /* upper bound for cached bytes */
} vp_pool_stats;

void vp_pool_get_stats(vp_pool_stats* out);
/* [main thread] Returns all cached blocks to the system. */
void vp_pool_trim(void);
/* [main thread] Sets the bound for cached bytes (0: no caching). */
void vp_pool_set_limit(uint64_t bytes);

/* --- Arrays -------------------------------------------------------------- */

/* [GIL] A ready array over the memory of a NumPy array (no copy). The
 * runtime keeps `ndarray` alive while its memory is in use. The caller must
 * not modify the ndarray's memory afterwards. */
int vp_array_from_ndarray(vp_runtime* rt, PyObject* ndarray, vp_array** out, char* err,
                          size_t errlen);
/* [GIL] A ready 0-d array holding `value` (a Python or NumPy scalar)
 * converted to `dtype` with NumPy's rules (e.g. OverflowError for a Python
 * int out of range). Returns VP_PYERR with the Python exception set. */
int vp_array_from_scalar(vp_runtime* rt, PyObject* value, vp_dtype dtype, vp_array** out);
/* [GIL] Like vp_array_from_scalar(), but converted as NumPy's `a[i] = value`
 * converts it (PyArray_Pack): NumPy's values and errors for assignments. */
int vp_array_from_item(vp_runtime* rt, PyObject* value, vp_dtype dtype, vp_array** out);
/* [GIL, main thread] NumPy's scalar of the first element of `a`, which must
 * be ready (vp_array_ready). New reference. */
PyObject* vp_array_scalar(const vp_array* a);
/* [main thread] NumPy's a.reshape(shape, order) as a view of `a` (order 'C'
 * or 'F'; `shape` without -1, same size as `a`), or NULL if NumPy would
 * copy. Metadata only, computed as NumPy computes it (PyArray_Newshape). */
vp_array* vp_array_reshape(vp_runtime* rt, const vp_array* a, int ndim, const int64_t* shape,
                           char order);
/* [GIL] A view of `base` sharing its storage. `offset` is in bytes, relative
 * to base's first element; strides are in bytes. Metadata only. */
vp_array* vp_array_view(vp_runtime* rt, const vp_array* base, vp_dtype dtype, int ndim,
                        const int64_t* shape, const int64_t* strides, int64_t offset);
/* [GIL] Releases the handle. Pending work continues. */
void vp_array_release(vp_array* a);

/* Metadata, available immediately (also for pending arrays). */
int vp_array_ndim(const vp_array* a);
const int64_t* vp_array_shape(const vp_array* a);
const int64_t* vp_array_strides(const vp_array* a);
vp_dtype vp_array_dtype(const vp_array* a);
/* Byte offset of the first element in the array's storage. */
int64_t vp_array_offset(const vp_array* a);
/* 1 if the data is final (no pending task writes it). */
int vp_array_ready(vp_array* a);
/* 1 if both arrays share their storage (base and views). */
int vp_array_same_storage(const vp_array* a, const vp_array* b);
/* [main thread] Size in bytes if `a`'s buffer may be overwritten by an op
 * that takes `a` as a temporary input (docs/design/memory-management.md, D5):
 * `a` is the only array on its storage (no views, one reference: the
 * caller's), the buffer is pool memory (not shared with NumPy), no issued
 * task still reads it, and `a` covers it contiguously from offset 0.
 * 0 otherwise. Whether the *caller's* object is a temporary is for the
 * bridge to decide. */
size_t vp_array_reusable_bytes(const vp_array* a);

/* [no GIL] Sync point: waits until `a` is ready. Returns VP_ERROR with the
 * failing op's message and issue site if a task computing it failed. */
int vp_array_wait(vp_runtime* rt, vp_array* a, char* err, size_t errlen);
/* [GIL, main thread] Copy-free export: a NumPy view of the data. `a` must be
 * ready. Read-only (writable == 0): the view counts as a reader of the buffer
 * while it lives, so later in-place writes to `a` copy-on-write and the view
 * keeps its data (copy semantics). Writable: only after
 * vp_array_prepare_write(), for NumPy code run synchronously on the main
 * thread (docs/design/memory-management.md, D7). */
PyObject* vp_array_export(const vp_array* a, int writable);
/* [GIL, main thread] A NumPy array over `a`'s data without an owner, for a
 * NumPy call made right away on ready data that only reads it (reductions):
 * valid while the caller keeps `a` and issues nothing that writes it.
 * Flagged writable as NumPy's own arrays are (NumPy copies read-only inputs
 * in some reductions); the callee must not write. If the array is still
 * referenced after the call, vp_array_pin() makes it a read-only export (it
 * then keeps the buffer alive and counts as a reader). `a` must be ready.
 * NULL with a Python error on failure. */
PyObject* vp_array_borrow(const vp_array* a);
/* [main thread] 1 if NumPy may write `a`'s buffer right now, on the main
 * thread with the GIL, without waiting: `a` is ready without a stored error
 * and no task and no NumPy view (export) reads its buffer. Then a writable
 * vp_array_borrow() view may be written during one NumPy call (index
 * assignment delegated to NumPy). Stays true until the main thread issues a
 * reader or writer of the storage. */
int vp_array_host_writable(vp_array* a);
int vp_array_pin(const vp_array* a, PyObject* view);
/* [no GIL, main thread] Sync point before NumPy code writes `a`'s buffer
 * directly: waits for its producer, copies-on-write if NumPy views of the
 * buffer are alive, waits until no task reads it. */
int vp_array_prepare_write(vp_runtime* rt, vp_array* a, char* err, size_t errlen);

/* --- Ops (all [GIL], all asynchronous) ------------------------------------ */

/* [GIL] Interns an issue-site description ("file:line in function") and
 * returns a copy that stays valid until the runtime is destroyed. Ops take
 * their `site` argument only from here: tasks keep the pointer, so passing
 * it costs nothing per op. Callers should cache the result per site. */
const char* vp_intern(vp_runtime* rt, const char* site);

/* Issues `ufunc(inputs...)` with one output, using NumPy's loop for exactly
 * the dtypes (inputs..., out_dtype); no casting. Inputs broadcast to
 * out_shape.
 *  - out == NULL: creates the result array (*result).
 *  - out != NULL: writes into `out` (in/out semantics, `out=` / `a += b`);
 *    *result is set to NULL.
 * Returns VP_UNSUPPORTED if no matching loop exists or it needs the Python
 * API; the caller then falls back to NumPy. `site` (from vp_intern) names
 * the issuing source location; it is reported with errors. */
int vp_ufunc(vp_runtime* rt, PyObject* ufunc, int nin, vp_array* const* inputs, vp_dtype out_dtype,
             int out_ndim, const int64_t* out_shape, vp_array* out, vp_array** result,
             const char* site, char* err, size_t errlen);

/* vp_ufunc_issue() with where=`mask` (a bool array broadcasting to
 * out_shape): as NumPy runs a masked ufunc, on its iterator (no copies of
 * short inputs), the loop called once per run of True values. Elements
 * where the mask is False keep the values of `out`; without `out` they are
 * undefined, as in NumPy's freshly allocated result. */
int vp_ufunc_issue_masked(vp_runtime* rt, const vp_loop* loop, vp_array* const* inputs,
                          vp_array* mask, int out_ndim, const int64_t* out_shape, vp_array* out,
                          vp_array** result, const char* site, char* err, size_t errlen);

/* Issues `ufunc(inputs...)` for a ufunc with several outputs (divmod, modf,
 * frexp) with NumPy's loop for exactly `dtypes` (nin inputs, then nout
 * outputs). As NumPy runs such ufuncs (never its trivial loop), the loop is
 * driven by NumPy's iterator, which casts inputs of another dtype in its
 * buffers, and the results get the layout that iterator allocates (order
 * K). Inputs broadcast to out_shape; results[0..nout) receive the new
 * arrays. Returns VP_UNSUPPORTED (reason in `err`) if no matching loop
 * exists or NumPy cannot run it so; the caller then falls back. */
int vp_ufunc_multi(vp_runtime* rt, PyObject* ufunc, int nin, vp_array* const* inputs, int nout,
                   const vp_dtype* dtypes, int out_ndim, const int64_t* out_shape,
                   vp_array** results, const char* site, char* err, size_t errlen);

/* Resolves NumPy's loop for `ufunc` with exactly `dtypes` (nin inputs, then
 * the output) once; the handle stays valid until the runtime is destroyed.
 * vp_ufunc_issue() then issues ops without any lookup. Returns
 * VP_UNSUPPORTED (and a reason in `err`) like vp_ufunc(). */
int vp_ufunc_loop(vp_runtime* rt, PyObject* ufunc, int nin, const vp_dtype* dtypes,
                  const vp_loop** loop, char* err, size_t errlen);
/* Like vp_ufunc(), with a loop from vp_ufunc_loop(). Inputs of another dtype
 * than the loop's are cast as NumPy casts them (short ones into copies
 * first, others in the iterator's buffers); an `out` of another dtype gets
 * the results cast into it, as NumPy's iterator does. */
int vp_ufunc_issue(vp_runtime* rt, const vp_loop* loop, vp_array* const* inputs, int out_ndim,
                   const int64_t* out_shape, vp_array* out, vp_array** result, const char* site,
                   char* err, size_t errlen);

/* Issues a cast of `in` to `dtype` (NumPy's a.astype(dtype, order)) with
 * NumPy's cast functions. `order` 'K' gives the result NumPy's layout for
 * astype (PyArray_NewLikeArray: C or F if `in` is, else its axis order),
 * 'C' C order, 'F' F order. Returns VP_UNSUPPORTED if NumPy cannot cast without the
 * Python API or the cast is empty. NumPy's warnings (ComplexWarning) are
 * issued during the call, as by NumPy (VP_UNSUPPORTED if a warning filter
 * turns one into an error: NumPy raises it in the fallback). */
int vp_astype(vp_runtime* rt, vp_array* in, vp_dtype dtype, char order, vp_array** result,
              const char* site, char* err, size_t errlen);

/* Advanced indexing (element copies, so any element order gives NumPy's
 * values). `idx` are `nidx` int32/int64 index arrays into the axes
 * `idx_axes` of the view `src` / `dst`; they broadcast together. Each of the
 * `nres` result axes comes from a non-indexed axis of the view
 * (result_axes[k] >= 0) or from dimension d of the broadcast index shape
 * (result_axes[k] = -1 - d); NumPy's placement rules are the caller's.
 * Out-of-bounds indices fail the task: the next sync point raises
 * AsyncIndexError (an IndexError). */
/* `mode` decides what an index outside [-n, n) does, as NumPy's take/put
 * modes: VP_INDEX_RAISE (the task fails), VP_INDEX_WRAP (i mod n),
 * VP_INDEX_CLIP (to 0 or n - 1; clip maps negative indices to 0 too). */
enum { VP_INDEX_RAISE = 0, VP_INDEX_WRAP = 1, VP_INDEX_CLIP = 2 };
/* r = src[idx...]: a new array with `result_strides` (NumPy's layout for the
 * indexing result; NULL for C order). */
int vp_gather(vp_runtime* rt, vp_array* src, int nidx, vp_array* const* idx, const int* idx_axes,
              int nres, const int* result_axes, const int64_t* result_strides, int mode,
              vp_array** result, const char* site, char* err, size_t errlen);
/* dst[idx...] = values (dst's dtype, broadcastable to the indexing result's
 * shape); the last of repeated indices wins, as in NumPy. */
int vp_scatter(vp_runtime* rt, vp_array* dst, int nidx, vp_array* const* idx, const int* idx_axes,
               int nres, const int* result_axes, vp_array* values, int mode, const char* site,
               char* err, size_t errlen);
/* dst[mask] = value: `value` 0-d with dst's dtype, `mask` bool broadcastable
 * to dst. */
int vp_mask_assign(vp_runtime* rt, vp_array* dst, vp_array* mask, vp_array* value, const char* site,
                   char* err, size_t errlen);

/* np.where(cond, x, y) with `cond` bool and x, y already of the result's
 * dtype (the caller casts as NumPy does); all three broadcast. The result
 * has NumPy's layout (order K over the three inputs). */
int vp_select(vp_runtime* rt, vp_array* cond, vp_array* x, vp_array* y, vp_array** result,
              const char* site, char* err, size_t errlen);

/* Issues a sort (`arg` 0) or argsort (`arg` != 0) of `in` along `axis` with
 * NumPy's per-dtype sort functions (src/kernels/numpy_sort.hpp); `kind` is
 * NumPy's sort table index (0 the default, 2 stable; NumPy maps heapsort to
 * 0). A sort writes a new array, dense with its axes in `out_order`
 * (order_n == ndim entries, outermost first; order_n == 0: the layout of
 * np.sort's copy, empty_like(in) in order K), or,
 * with `in_place`, sorts `in` itself (`a.sort()`, *result NULL); an argsort
 * gives a new C-ordered int64 array. Returns VP_UNSUPPORTED for other
 * arguments. */
int vp_sort(vp_runtime* rt, vp_array* in, int axis, int kind, int arg, int in_place, int order_n,
            const int* out_order, vp_array** result, const char* site, char* err, size_t errlen);

/* Issues argmax (`argmax` != 0) or argmin of `in` along its last axis with
 * NumPy's per-dtype functions, one call per row, as PyArray_ArgMax calls
 * them. `in` must be C-contiguous with at least one dimension (the bridge
 * moves the axis last and copies as NumPy does); the result is an intp
 * (int64) array of shape `out_shape` with one element per row. */
int vp_arg_reduce(vp_runtime* rt, int argmax, vp_array* in, int out_ndim, const int64_t* out_shape,
                  vp_array** result, const char* site, char* err, size_t errlen);

/* Issues a reduction of `in` with the binary `ufunc` (add, multiply, minimum,
 * maximum, logical_and, ...) over the dimensions flagged in `axes` (one entry
 * per dimension of `in`; nonzero = reduced). viproc calls NumPy's own
 * reduction loop, in the blocks NumPy's reduction passes to it, so results
 * are bit-identical (NumPy >= 2.3; the caller checks the version). The
 * result has the input's dtype; reduced dimensions stay with extent 1 if
 * `keepdims`. `identity` is the value the accumulator starts at (the
 * ufunc's identity), or Py_None to start at the first element along the
 * reduced axes (minimum, maximum). `dtype` (Py_None or a dtype) is the
 * reduction's dtype= argument; NumPy's rules pick the accumulator (int32
 * sums in int64). Reductions that cast or whose reduced axes do not coalesce
 * run on NumPy's own iterator (NpyIter, buffered as in NumPy); the result is
 * then the array that iterator allocates, in NumPy's layout. Returns
 * VP_UNSUPPORTED (reason in `err`) if the input is empty, the result dtype is
 * not supported, the loop needs the Python API, or a reduction without
 * initial value casts; the caller then falls back to NumPy. */
int vp_reduce(vp_runtime* rt, PyObject* ufunc, vp_array* in, const unsigned char* axes,
              int keepdims, PyObject* dtype, PyObject* identity, vp_array** result,
              const char* site, char* err, size_t errlen);

/* Issues ufunc.accumulate(in, axis=axis) (np.cumsum, np.cumprod, ...) for a
 * binary ufunc without casts: input and result have in's dtype, which must
 * be the accumulator dtype NumPy picks (the caller casts first, as NumPy's
 * iterator copies a cast input). The loop is the reduction loop called as
 * PyUFunc_Accumulate calls it: per row along `axis`, the first element is
 * copied, then one call over the rest of the row with args {out, in + 1,
 * out + 1}. The result has NumPy's layout (C-contiguous for 1-d, else the
 * input's axis order, as NumPy's iterator allocates it). Returns
 * VP_UNSUPPORTED (reason in `err`) for an empty input, 0-d input, or a loop
 * that is unavailable or needs the Python API; the caller then falls back. */
int vp_accumulate(vp_runtime* rt, PyObject* ufunc, vp_array* in, int axis, vp_array** result,
                  const char* site, char* err, size_t errlen);

/* vp_reduce() into an existing array `out` (in place, *result set to NULL):
 * only where `out` has the dtype and layout of the result NumPy would
 * allocate (strides of extents > 1), so the reduction iterates exactly as
 * without out=; VP_UNSUPPORTED otherwise. */
int vp_reduce_out(vp_runtime* rt, PyObject* ufunc, vp_array* in, const unsigned char* axes,
                  int keepdims, PyObject* dtype, PyObject* identity, vp_array* out,
                  vp_array** result, const char* site, char* err, size_t errlen);

/* Issues ufunc.reduceat(in, indices, axis) for a binary ufunc without casts
 * (in has the reduction loop's dtype: the caller casts first), as
 * PyUFunc_Reduceat runs it (make_reduceat_kernel). `indices` (host data,
 * nindices >= 1) must lie in [0, extent); the result has extent nindices
 * along `axis` and the dense byte strides `out_strides` (NumPy's layout,
 * computed by the caller; NULL: C order). Returns
 * VP_UNSUPPORTED (reason in `err`) for an index out of range, a 0-d input,
 * an empty result or a loop that is not available; the caller then falls
 * back (and NumPy raises its errors). */
int vp_reduceat(vp_runtime* rt, PyObject* ufunc, vp_array* in, int axis, const int64_t* indices,
                int nindices, const int64_t* out_strides, vp_array** result, const char* site,
                char* err, size_t errlen);

/* Issues ufunc.at(a, indices, values) in place on `a` with `loop` (every
 * operand of a's dtype; vp_ufunc_loop): `indices` a 1-d int32/int64 array
 * into a's first axis, `values` (NULL for a unary ufunc) broadcasting to
 * indices.shape + a.shape[1:]. Runs as NumPy's ufunc_at without casts
 * (make_at_kernel): repeated indices accumulate; an index out of bounds
 * fails the op (AsyncIndexError) before anything is written. Returns
 * VP_UNSUPPORTED (reason in `err`) for other operands. */
int vp_ufunc_at(vp_runtime* rt, const vp_loop* loop, vp_array* a, vp_array* indices,
                vp_array* values, const char* site, char* err, size_t errlen);

/* Issues a generalized ufunc (`ufunc.signature` set: vecdot, matvec, vecmat,
 * numpy.linalg's and numpy.fft's gufuncs) with NumPy's loop for exactly
 * `dtypes` (nin inputs, then nout outputs), called as
 * PyUFunc_GeneralizedFunctionInternal calls it (src/kernels/numpy_gufunc.hpp).
 *  - Operands have their core axes last (the caller applies axes=/axis=).
 *    `core_n[i]` is the number of core axes of operand i in the signature,
 *    `core_axes` (concatenated) their label indices, or -1 - label for a
 *    missing flexible dimension (`m?` of a 1-d matmul operand).
 *  - `sizes` (nlabels entries, -1 if unknown) receives each label's size;
 *    a label only an output has needs its size from the caller or `outs`.
 *  - Inputs of another dtype are cast into copies first, laid out as
 *    NumPy's iterator lays them out.
 *  - outs == NULL or outs[0] == NULL: new outputs into results[0..nout), in
 *    NumPy's iterator's layout, or, with order_n > 0, dense with their axes
 *    in `out_order` (order_n entries, outermost first; every output has
 *    order_n axes): numpy.fft allocates `out` with empty_like (order K).
 *    Otherwise all outputs are given (in/out semantics, exactly the loop's
 *    dtypes and the batch shape) and results[] are set to NULL.
 *  - `invalid_error` (or NULL): the loop reports failure through the
 *    floating-point invalid flag (numpy.linalg); the op then fails with this
 *    message ("LinAlgError: Singular matrix").
 *  - VP_GUFUNC_CHECK_FINITE in `flags`: the op fails with "LinAlgError: Array
 *    must not contain infs or NaNs" if input 0 has a non-finite element,
 *    before the loop runs (numpy.linalg's _assert_finite, eig/eigvals).
 *  - VP_GUFUNC_SERIAL in `flags`: the loop calls of all ops with this flag
 *    run one at a time (LAPACK on matrices OpenBLAS threads itself: several
 *    threaded calls at once oversubscribe its threads).
 *  - `work` sizes the op for the dispatch policy (element-wise op units).
 * Returns VP_UNSUPPORTED (reason in `err`) where NumPy raises or copies an
 * operand for another reason (misaligned, out= overlapping an input, out=
 * of another dtype or shape, shapes that do not match); the caller then
 * falls back to NumPy. */
enum { VP_GUFUNC_CHECK_FINITE = 2, VP_GUFUNC_SERIAL = 4 };
int vp_gufunc(vp_runtime* rt, PyObject* ufunc, int nin, vp_array* const* inputs, int nout,
              vp_array* const* outs, const vp_dtype* dtypes, const int* core_n,
              const int* core_axes, int nlabels, int64_t* sizes, int flags, int order_n,
              const int* out_order, const char* invalid_error, int64_t work, vp_array** results,
              const char* site, char* err, size_t errlen);

/* Issues np.matmul(a, b) (the `@` operator) with NumPy's matmul gufunc loop
 * for (t, t) -> t, where t is the dtype of both operands; the loop is
 * called with the strides NumPy's gufunc machinery passes, so it makes the
 * same BLAS calls (src/kernels/numpy_matmul.hpp). The result has NumPy's
 * shape and layout. Returns VP_UNSUPPORTED (reason in `err`) where NumPy
 * would cast or copy an operand (different dtypes, misaligned data), the
 * loop needs the Python API, or NumPy raises (0-d operands, mismatching
 * core dimensions, batch shapes that do not broadcast); the caller then
 * falls back to NumPy. */
int vp_matmul(vp_runtime* rt, PyObject* ufunc, vp_array* a, vp_array* b, vp_array** result,
              const char* site, char* err, size_t errlen);

/* Issues dst[...] = src (src broadcasts to dst's shape; another dtype is
 * cast as NumPy's assignment casts, VP_UNSUPPORTED if NumPy cannot). `site`
 * comes from vp_intern. */
int vp_assign(vp_runtime* rt, vp_array* dst, vp_array* src, const char* site, char* err,
              size_t errlen);

/* Issues a new array (*result) of `dtype` and `shape` (`ndim` dimensions)
 * with the dense byte `strides` (NULL: C order) assembled from `nparts`
 * parts: part i is the view of the result with `part_ndim[i]` dimensions,
 * the shapes and byte strides at `part_shapes` / `part_strides` (all parts'
 * dimensions one after the other) and the byte offset `part_offsets[i]` from
 * the result's first element; srcs[i] (`dtype`, broadcastable to the part's
 * shape) is copied into it. One task copies all parts in order (a later part
 * overwrites an earlier one); elements no part covers are undefined. Used for
 * concatenate, stack, tile, repeat, pad, roll, delete and insert. */
int vp_place(vp_runtime* rt, vp_dtype dtype, int ndim, const int64_t* shape, const int64_t* strides,
             int nparts, vp_array* const* srcs, const int* part_ndim, const int64_t* part_shapes,
             const int64_t* part_strides, const int64_t* part_offsets, vp_array** result,
             const char* site, char* err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif /* VIPROC_VIPROC_H */
