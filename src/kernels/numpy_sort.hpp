// Kernel adapter: sort and argsort along one axis with NumPy's per-dtype
// sort functions (PyDataType_GetArrFuncs(descr)->sort / ->argsort: the
// quicksort, heapsort and timsort/radixsort instances NumPy's sort
// ArrayMethods call, including their SIMD dispatch).
//
// NumPy (PyArray_Sort, _new_sortlike) sorts each 1-d row along the axis on
// its own: in place if the row is contiguous and aligned, else in a
// contiguous buffer it copies the row into and back from. argsort
// (_new_argsortlike) fills 0..N-1 and sorts the indices by the row's
// values. A row's result depends only on its values and the function, so
// viproc's calls (same function, same values, contiguous) give NumPy's
// results, including the order of equal elements, NaNs and signed zeros.
#pragma once

#include "kernels/numpy_ufunc.hpp"

#include <memory>
#include <string>

namespace viproc::numpy {

// Sort (`arg` false) or argsort (`arg` true) of `dtype` along `axis`.
// `kind`: NumPy's table index (0 quicksort, 1 heapsort, 2 stable).
// Operands: sort: input (any layout), output (same shape, any layout) or
// one in/out (sorted in place); argsort: input, int64 output of the same
// shape. Any thread, without the GIL.
std::shared_ptr<Kernel> make_sort_kernel(DType dtype, int kind, bool arg, int axis,
                                         std::string& error);

} // namespace viproc::numpy
