// Element copy kernel: dst[...] = src (with broadcasting, same dtype).
#pragma once

#include "task.hpp"

#include <memory>

namespace viproc {

// Kernel operands: inputs = [src, dst (as in/out)], outputs = [dst].
std::shared_ptr<Kernel> make_assign_kernel();

} // namespace viproc
