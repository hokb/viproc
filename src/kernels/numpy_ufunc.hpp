// Kernel adapter: runs element-wise array ops with NumPy's own ufunc loops.
#pragma once

#include "viproc_python.h"

#include "runtime/runtime.hpp"

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace viproc::numpy {

// Must be called once, with the GIL held, before anything else in this file.
bool init();

int type_num(DType dtype);

// A NumPy strided loop for one ufunc and one exact dtype signature, obtained via
// ufunc._resolve_dtypes_and_context() / ufunc._get_strided_loop().
//
// Created and destroyed on the main thread with the GIL held (it owns a Python
// capsule). call() is safe on any thread without the GIL.
class UfuncLoop {
  public:
    ~UfuncLoop();
    UfuncLoop(const UfuncLoop&) = delete;
    UfuncLoop& operator=(const UfuncLoop&) = delete;

    // `dtypes` lists all operands (inputs, then outputs). Fails if NumPy would
    // resolve a different signature (casting is not supported yet).
    static std::unique_ptr<UfuncLoop> resolve(PyObject* ufunc, std::span<const DType> dtypes,
                                              std::string& error);
    // The loop NumPy uses to reduce `dtype` with the binary `ufunc` (resolved
    // with reduction=True), for the inner strides `fixed_strides` (acc, in,
    // acc), as NumPy's reduction passes them to get_strided_loop. Fails if
    // the reduction would cast (the accumulator dtype differs from `dtype`).
    static std::unique_ptr<UfuncLoop> resolve_reduction(PyObject* ufunc, DType dtype,
                                                        const std::array<std::intptr_t, 3>& fixed,
                                                        std::string& error);
    // The loop of a call info capsule from ufunc._resolve_dtypes_and_context,
    // filled by ufunc._get_strided_loop for `fixed` strides (any number of
    // operands). Takes a reference to `capsule`.
    static std::unique_ptr<UfuncLoop> from_call_info(PyObject* ufunc, PyObject* capsule,
                                                     std::span<const std::intptr_t> fixed,
                                                     std::string& error);

    int call(char* const* data, const std::intptr_t* dims, const std::intptr_t* strides) const;

    int nin() const { return nin_; }
    // Element-wise kernel for this loop, shared by all tasks using it.
    const std::shared_ptr<Kernel>& elementwise_kernel() const { return kernel_; }
    int nout() const { return nout_; }
    bool requires_pyapi() const;
    // The ufunc (borrowed; ufuncs live as long as NumPy).
    PyObject* ufunc() const { return ufunc_; }

  private:
    UfuncLoop() = default;
    PyObject* capsule_ = nullptr;
    PyObject* ufunc_ = nullptr;
    void* info_ = nullptr; // UfuncCallInfoPrefix inside the capsule
    int nin_ = 0;
    int nout_ = 0;
    std::shared_ptr<Kernel> kernel_;
};

// Runs `release` on the main thread with the GIL: now, if the caller holds
// the GIL, else at the next C ABI call. Kernels owning Python objects
// release them through it (a kernel may be destroyed on a worker).
using DeferRelease = std::function<void(std::function<void()>)>;

// An element-wise iterator template and its pool of reusable copies
// (numpy_ufunc.cpp).
struct IterTemplate;

// Caches resolved loops per (ufunc, dtype signature). Main thread with the GIL.
// Must outlive every task that uses one of its loops (wait for the runtime
// first, then destroy the cache).
class LoopCache {
  public:
    const UfuncLoop* get(PyObject* ufunc, std::span<const DType> dtypes, std::string& error);
    // Reduction loop (UfuncLoop::resolve_reduction), cached per ufunc, dtype
    // and inner strides.
    const UfuncLoop* get_reduction(PyObject* ufunc, DType dtype,
                                   const std::array<std::intptr_t, 3>& fixed, std::string& error);

    // Element-wise kernel driven by NumPy's own iterator, for layouts where
    // NumPy's ufunc machinery does not make a single loop call (strided or
    // negatively strided operands, broadcasting, mixed axis orders). NumPy's
    // iterator decides there how the loop is called: which axes are
    // coalesced, the traversal order, and whether operands are copied to
    // contiguous buffers first. That changes results: NumPy's SIMD loops
    // (complex multiply, float32 exp/log/sin, ...) round differently from
    // their scalar fallbacks, and which one runs depends on the strides.
    //
    // The iterator is set up as NumPy's execute_ufunc_loop() does (same
    // flags, order K, the current np.getbufsize()) over template arrays with
    // the operands' layouts, once per (loop, layouts, buffer size); each
    // kernel gets a copy (NpyIter_Copy), which it points at the real buffers
    // (NpyIter_ResetBasePointers) and runs NumPy's loop over without the GIL.
    // `out_given`: the output is an existing array (out=, in-place
    // operators); otherwise viproc allocated it in NumPy's layout, and the
    // iterator must not reverse axes, as NumPy's would not with an output it
    // allocates itself. Returns nullptr (reason in `error`, Python error
    // cleared) if NumPy cannot. Main thread with the GIL.
    // `dtypes`: the loop's operand dtypes (inputs, then the output); inputs
    // of another dtype are cast in the iterator's buffers, as NumPy does.
    std::shared_ptr<Kernel> iter_elementwise(const UfuncLoop* loop, std::span<const DType> dtypes,
                                             std::span<const Layout> inputs, const Layout& out,
                                             bool out_given, DeferRelease defer,
                                             std::string& error);

    LoopCache() = default;
    LoopCache(const LoopCache&) = delete;
    LoopCache& operator=(const LoopCache&) = delete;
    ~LoopCache();

    // np.getbufsize(), re-read only when NumPy's extobj context changes.
    long buffersize();

  private:
    std::map<std::pair<PyObject*, std::vector<DType>>, std::unique_ptr<UfuncLoop>> loops_;
    std::map<std::tuple<PyObject*, DType, std::array<std::intptr_t, 3>>, std::unique_ptr<UfuncLoop>>
        reduction_loops_;
    std::unordered_map<std::string, std::shared_ptr<IterTemplate>> iter_templates_;
    PyObject* extobj_var_ = nullptr; // NumPy's extobj context variable
    PyObject* extobj_ = nullptr;     // its value at the last read
    long buffersize_ = 0;
};

// Element-wise kernel: calls `loop` once over all elements where
// single_call() holds, with its strides, else once per innermost row.
// Operand order follows Runtime::issue(). The bridge uses it only where
// single_call() holds; everything else goes through
// LoopCache::iter_elementwise().
std::shared_ptr<Kernel> make_elementwise_kernel(const UfuncLoop* loop);

// True if NumPy calls the loop of an element-wise ufunc with one output once
// over all elements with these operand layouts; `strides` (inputs, then the
// output) receives the strides NumPy passes. That is NumPy's trivial loop
// (0-d inputs; operands with the output's shape, 1-d with their own strides
// or n-d with the same C/F contiguity) and the iterator where it coalesces
// everything (operands dense with the output's strides, or single elements).
// The strides matter even for one element: NumPy's SIMD loops are chosen by
// stride and round differently from their scalar fallbacks.
bool single_call(std::span<const Layout* const> inputs, const Layout& out, std::intptr_t* strides);

// Casts `in` into `out` (same shape, another dtype) with NumPy's cast
// functions (inputs = {in}, outputs = {out}); casts convert each element on
// its own, so any layouts work. Used for astype() and for the inputs NumPy's
// ufuncs cast before their single loop call (0-d and short 1-d ones,
// check_for_trivial_loop). Main thread with the GIL; nullptr (reason in
// `error`, Python error cleared by the caller) if NumPy cannot. NumPy's
// warnings (ComplexWarning) are raised here, at issue, as in NumPy's call.
std::shared_ptr<Kernel> make_cast_kernel(const Layout& in, const Layout& out, long buffersize,
                                         DeferRelease defer, std::string& error);

// NumPy's scalar of the element at `data` (dtype `dtype`). GIL held.
PyObject* scalar_object(DType dtype, const void* data);

// A writable ndarray with `l`'s dtype, shape and strides over a dummy
// address (see vp_array_stand_in). New reference or nullptr. GIL held.
PyObject* stand_in(const Layout& l);
// If `r` is an ndarray view of `stand_in` with a supported dtype: its layout
// (offset relative to the stand-in's first element) and writability.
bool view_of(PyObject* stand_in, PyObject* r, Layout& layout, bool& writable);

// Wraps the memory of a NumPy array as a ready runtime array without copying.
// `owner` keeps the memory alive (e.g. holds a reference to `arr`); with a
// null owner the caller keeps `arr` alive until no task reads it any more.
ArrayPtr wrap_ndarray(Runtime& rt, PyObject* arr, std::string& error,
                      std::shared_ptr<void> owner = nullptr);
// Same checks as wrap_ndarray(), without wrapping.
bool supported_ndarray(PyObject* arr);
// The memory of a supported NumPy array: its lowest address, the bytes its
// elements span, and its layout relative to that address.
bool ndarray_layout(PyObject* arr, Layout& layout, void*& data, std::size_t& nbytes);

// A ready 0-d array holding `value` converted to `dtype` by NumPy (nullptr
// with a Python exception set on failure).
ArrayPtr scalar_array(Runtime& rt, PyObject* value, DType dtype);

// NumPy's resolution of a reduction of `in_descr` with the binary `ufunc`
// (ufunc._resolve_dtypes_and_context((None, in_descr, None), signature=
// (dtype, None, None), reduction=True)): a new reference to the tuple
// (descriptors, call info capsule), or nullptr (error in `error`, Python
// error cleared). `dtype` may be Py_None. GIL held.
PyObject* resolve_reduction_info(PyObject* ufunc, PyObject* in_descr, PyObject* dtype,
                                 std::string& error);

// The bytes of `value` converted to `dtype` by NumPy (false with a Python
// exception set on failure). GIL held.
bool scalar_bytes(PyObject* value, DType dtype, std::vector<std::byte>& out);

// Copy-free export (docs/design/memory-management.md, D7): a NumPy view of
// `a`'s buffer, with `a`'s shape and strides. `a` must be ready.
// - Read-only (writable == false): while the view (or any NumPy view of it)
//   lives, it counts as a reader of the buffer (Buffer::host_views), so a later
//   in-place write to `a` switches `a` to a new buffer (copy-on-write) and the
//   NumPy view keeps the data it showed: copy semantics without a copy.
// - Writable: only after Runtime::prepare_host_write(a), for NumPy code the
//   bridge runs synchronously on the main thread (eager fallbacks); writes go
//   straight into `a`, exactly as NumPy on an ndarray.
// Main thread, with the GIL.
PyObject* export_ndarray(const Array& a, bool writable);

} // namespace viproc::numpy
