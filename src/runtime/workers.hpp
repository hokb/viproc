// The number of worker threads.
#pragma once

#include <cstddef>

namespace numpy2x {

// The runtime's worker threads: 2, besides the main thread that issues the
// program's ops. Fixed: there is no option, environment variable or API
// argument for it. Thread-safe.
std::size_t worker_threads();

} // namespace numpy2x
