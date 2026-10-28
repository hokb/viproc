// Internal to the NumPy kernel adapters (numpy_ufunc.cpp, numpy_reduce.cpp):
// pools of NumPy iterators built once per template. Include after NumPy's
// headers.
#pragma once

#include "kernels/numpy_ufunc.hpp"

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace viproc::numpy {

// The pending Python error as a string (cleared). GIL held.
std::string fetch_python_error();

// An iterator copy with the accessors a kernel calls per inner loop.
struct PooledIter {
    NpyIter* iter = nullptr;
    NpyIter_IterNextFunc* iternext = nullptr;
    char** dataptr = nullptr;
    npy_intp* strideptr = nullptr;
    npy_intp* countptr = nullptr;
};

// An iterator built over template arrays (a dummy address, the operands'
// layouts) and its copies. A kernel takes a copy when it runs, points it at
// the real buffers (NpyIter_ResetBasePointers, no GIL) and gives it back
// when it is done, so the copies, and the buffers they allocate at their
// first reset (DELAY_BUFALLOC), are reused while they are warm. At most one
// kernel per thread runs at a time, so `target` copies (workers + main
// thread + 1) are enough: the main thread makes one more per issue until
// there are that many (`reserve`, with the GIL), never one per op in flight.
// Before, each op took a copy at issue: with up to 1,000 ops in flight the
// pool (16) ran dry, and nearly every op ran on a new copy whose buffers
// faulted in on the worker (bool * float64 of 1e4 elements: ~25 µs per op
// in page faults, NumPy takes 8 µs for the whole op).
struct IterTemplate {
    NpyIter* templ = nullptr;
    DeferRelease defer;
    std::size_t target = 0;
    std::atomic<std::size_t> alive{0}; // copies that exist (free or running)
    std::mutex mutex;
    std::vector<PooledIter> free; // finished copies, ready for a reset
    // Set for iterators NpyIter_Copy cannot copy: a new one is made this
    // way instead (where= masks: NumPy 2.x's copy of a masked buffered
    // transfer frees its data twice). `owned`: Python references `fresh`
    // uses, released with the template.
    std::function<NpyIter*()> fresh;
    std::vector<PyObject*> owned;

    ~IterTemplate() {
        std::vector<NpyIter*> all;
        for (const PooledIter& p : free) {
            all.push_back(p.iter);
        }
        all.push_back(templ);
        std::vector<PyObject*> refs = std::move(owned);
        defer([all, refs] {
            for (NpyIter* it : all) {
                NpyIter_Deallocate(it);
            }
            for (PyObject* o : refs) {
                Py_XDECREF(o);
            }
        });
    }

    // One more copy while there are fewer than `target` (a kernel about to
    // be issued may need it). False with a Python error set if NumPy fails.
    // Main thread with the GIL.
    bool reserve() {
        if (alive.load(std::memory_order_relaxed) >= target) {
            return true;
        }
        PooledIter p;
        if (!make(p)) {
            return false;
        }
        std::lock_guard lock(mutex);
        free.push_back(p);
        return true;
    }

    // A copy for a kernel that runs now. Any thread, without the GIL; only
    // when none is free (all `target` copies running, or some dropped after
    // an incomplete iteration) it takes the GIL and makes one. False with
    // the reason in `error` if NumPy fails.
    bool acquire(PooledIter& p, std::string& error) {
        {
            std::lock_guard lock(mutex);
            if (!free.empty()) {
                p = free.back();
                free.pop_back();
                return true;
            }
        }
        const PyGILState_STATE g = PyGILState_Ensure();
        const bool ok = make(p);
        if (!ok) {
            error = fetch_python_error();
        }
        PyGILState_Release(g);
        return ok;
    }

    // Takes a copy back after a complete iteration (nothing left in its
    // buffers to write back); others are released. Any thread.
    void give_back(const PooledIter& p, bool clean) {
        if (clean) {
            std::lock_guard lock(mutex);
            free.push_back(p);
            return;
        }
        alive.fetch_sub(1, std::memory_order_relaxed);
        defer([it = p.iter] { NpyIter_Deallocate(it); });
    }

  private:
    // GIL held.
    bool make(PooledIter& p) {
        NpyIter* it = fresh ? fresh() : NpyIter_Copy(templ);
        if (it == nullptr) {
            return false;
        }
        p.iternext = NpyIter_GetIterNext(it, nullptr);
        if (p.iternext == nullptr) {
            NpyIter_Deallocate(it);
            return false;
        }
        p.iter = it;
        p.dataptr = NpyIter_GetDataPtrArray(it);
        p.strideptr = NpyIter_GetInnerStrideArray(it);
        p.countptr = NpyIter_GetInnerLoopSizePtr(it);
        alive.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
};

// A new, empty template whose `target` is the number of threads that run
// kernels (set_kernel_threads) plus one. Main thread.
std::shared_ptr<IterTemplate> new_iter_template();

// An array with `l`'s shape and strides over `data` (not owned, never read
// or written: a template for an iterator's setup). New reference, nullptr
// with a Python error set on failure. GIL held.
PyArrayObject* template_array(const Layout& l, char* data, bool writable);

// Element-aligned dummy address for template arrays: nothing is read or
// written through it.
char* template_address();

} // namespace viproc::numpy
