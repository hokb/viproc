// Kernel adapter: runs element-wise array ops with NumPy's own ufunc loops.
#pragma once

#include "viproc_python.h"

#include "runtime/runtime.hpp"

#include <map>
#include <memory>
#include <span>
#include <string>
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

    int call(char* const* data, const std::intptr_t* dims, const std::intptr_t* strides) const;

    int nin() const { return nin_; }
    int nout() const { return nout_; }
    bool requires_pyapi() const;

  private:
    UfuncLoop() = default;
    PyObject* capsule_ = nullptr;
    void* info_ = nullptr; // UfuncCallInfoPrefix inside the capsule
    int nin_ = 0;
    int nout_ = 0;
};

// Caches resolved loops per (ufunc, dtype signature). Main thread with the GIL.
// Must outlive every task that uses one of its loops (wait for the runtime
// first, then destroy the cache).
class LoopCache {
  public:
    const UfuncLoop* get(PyObject* ufunc, std::span<const DType> dtypes, std::string& error);

  private:
    std::map<std::pair<PyObject*, std::vector<DType>>, std::unique_ptr<UfuncLoop>> loops_;
};

// Element-wise kernel: broadcasts the inputs to the shape of output 0 and calls
// `loop` once per innermost row. Operand order follows Runtime::issue().
std::shared_ptr<Kernel> make_elementwise_kernel(const UfuncLoop* loop);

// Wraps the memory of a NumPy array as a ready runtime array without copying.
// `owner` keeps the memory alive (e.g. holds a reference to `arr`); with a
// null owner the caller keeps `arr` alive until no task reads it any more.
ArrayPtr wrap_ndarray(Runtime& rt, PyObject* arr, std::string& error,
                      std::shared_ptr<void> owner = nullptr);
// Same checks as wrap_ndarray(), without wrapping.
bool supported_ndarray(PyObject* arr);

// New NumPy array with a copy of `a`'s data. `a` must be ready.
PyObject* to_ndarray(const Array& a);

} // namespace viproc::numpy
