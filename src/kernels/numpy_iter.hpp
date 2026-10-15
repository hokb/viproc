// Internal to the NumPy kernel adapters (numpy_ufunc.cpp, numpy_reduce.cpp):
// pools of NumPy iterators built once per template. Include after NumPy's
// headers.
#pragma once

#include "kernels/numpy_ufunc.hpp"

#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace viproc::numpy {

// Iterators of kernels that finished are kept per template and reused: a
// new copy (NpyIter_Copy) per op would also allocate its buffers on the
// worker at the first reset (DELAY_BUFALLOC), and freeing them later on the
// main thread keeps that memory cold. Main thread pops, any thread returns.
constexpr std::size_t kMaxPooledIters = 16;

// An iterator built over template arrays (a dummy address, the operands'
// layouts) and its finished copies. Kernels point a copy at the real
// buffers (NpyIter_ResetBasePointers, no GIL) when they run.
struct IterTemplate {
    NpyIter* templ = nullptr;
    DeferRelease defer;
    std::mutex mutex;
    std::vector<NpyIter*> free; // finished iterators, ready for a reset

    ~IterTemplate() {
        std::vector<NpyIter*> all = std::move(free);
        all.push_back(templ);
        defer([all] {
            for (NpyIter* it : all) {
                NpyIter_Deallocate(it);
            }
        });
    }

    // A clean iterator to reuse, else a new copy of the template (nullptr
    // with a Python error set if NumPy fails). Main thread with the GIL.
    NpyIter* get() {
        {
            std::lock_guard lock(mutex);
            if (!free.empty()) {
                NpyIter* it = free.back();
                free.pop_back();
                return it;
            }
        }
        return NpyIter_Copy(templ);
    }

    // Takes back an iterator after a complete iteration (nothing left in
    // its buffers to write back); others are released. Any thread.
    void give_back(NpyIter* it, bool clean) {
        if (clean) {
            std::lock_guard lock(mutex);
            if (free.size() < kMaxPooledIters) {
                free.push_back(it);
                return;
            }
        }
        defer([it] { NpyIter_Deallocate(it); });
    }
};

// An array with `l`'s shape and strides over `data` (not owned, never read
// or written: a template for an iterator's setup). New reference, nullptr
// with a Python error set on failure. GIL held.
PyArrayObject* template_array(const Layout& l, char* data, bool writable);

// The pending Python error as a string (cleared). GIL held.
std::string fetch_python_error();

// Element-aligned dummy address for template arrays: nothing is read or
// written through it.
char* template_address();

} // namespace viproc::numpy
