// OpenCL and CLBlast, loaded at run time (docs/design/opencl.md, "Loading").
//
// viproc does not link the OpenCL ICD loader or CLBlast: `import viproc` must
// work on machines without either. load_opencl() opens the ICD loader
// (libOpenCL.so.1, OpenCL.dll, the OpenCL framework) once per process; the
// cl* functions viproc and VkFFT call are defined in cl_loader.cpp as
// forwarders to the loaded library (hidden symbols, so they never interpose
// another module's OpenCL). load_clblast() opens CLBlast's shared library
// (env VIPROC_CLBLAST names its path) and resolves the routines the device
// kernels use; its C API types are mirrored here, so no CLBlast header is
// needed to build.
#pragma once

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <string>

namespace viproc::opencl {

// True if the OpenCL ICD loader is loaded (the first call loads it; any
// thread, thread-safe). `why` receives the reason when it is not.
bool load_opencl(std::string* why = nullptr);

// CLBlast's C API (clblast_c.h), the routines viproc calls. Enum arguments
// are passed as their int values.
enum : int {
    kRowMajor = 101,
    kNoTrans = 111,
    kTrans = 112,
    kConjTrans = 113,
    kUpper = 121,
    kLower = 122,
    kNonUnit = 131,
    kUnit = 132,
    kLeft = 141,
};

struct Clblast {
    using Gemm32 = int (*)(int, int, int, size_t, size_t, size_t, float, cl_mem, size_t, size_t,
                           cl_mem, size_t, size_t, float, cl_mem, size_t, size_t, cl_command_queue*,
                           cl_event*);
    using Gemm64 = int (*)(int, int, int, size_t, size_t, size_t, double, cl_mem, size_t, size_t,
                           cl_mem, size_t, size_t, double, cl_mem, size_t, size_t,
                           cl_command_queue*, cl_event*);
    using GemmC = int (*)(int, int, int, size_t, size_t, size_t, cl_float2, cl_mem, size_t, size_t,
                          cl_mem, size_t, size_t, cl_float2, cl_mem, size_t, size_t,
                          cl_command_queue*, cl_event*);
    using GemmZ = int (*)(int, int, int, size_t, size_t, size_t, cl_double2, cl_mem, size_t, size_t,
                          cl_mem, size_t, size_t, cl_double2, cl_mem, size_t, size_t,
                          cl_command_queue*, cl_event*);
    using Batched32 = int (*)(int, int, int, size_t, size_t, size_t, float, cl_mem, size_t, size_t,
                              size_t, cl_mem, size_t, size_t, size_t, float, cl_mem, size_t, size_t,
                              size_t, size_t, cl_command_queue*, cl_event*);
    using Batched64 = int (*)(int, int, int, size_t, size_t, size_t, double, cl_mem, size_t, size_t,
                              size_t, cl_mem, size_t, size_t, size_t, double, cl_mem, size_t,
                              size_t, size_t, size_t, cl_command_queue*, cl_event*);
    using BatchedC = int (*)(int, int, int, size_t, size_t, size_t, cl_float2, cl_mem, size_t,
                             size_t, size_t, cl_mem, size_t, size_t, size_t, cl_float2, cl_mem,
                             size_t, size_t, size_t, size_t, cl_command_queue*, cl_event*);
    using BatchedZ = int (*)(int, int, int, size_t, size_t, size_t, cl_double2, cl_mem, size_t,
                             size_t, size_t, cl_mem, size_t, size_t, size_t, cl_double2, cl_mem,
                             size_t, size_t, size_t, size_t, cl_command_queue*, cl_event*);
    using Trsm32 = int (*)(int, int, int, int, int, size_t, size_t, float, cl_mem, size_t, size_t,
                           cl_mem, size_t, size_t, cl_command_queue*, cl_event*);
    using Trsm64 = int (*)(int, int, int, int, int, size_t, size_t, double, cl_mem, size_t, size_t,
                           cl_mem, size_t, size_t, cl_command_queue*, cl_event*);
    using TrsmC = int (*)(int, int, int, int, int, size_t, size_t, cl_float2, cl_mem, size_t,
                          size_t, cl_mem, size_t, size_t, cl_command_queue*, cl_event*);
    using TrsmZ = int (*)(int, int, int, int, int, size_t, size_t, cl_double2, cl_mem, size_t,
                          size_t, cl_mem, size_t, size_t, cl_command_queue*, cl_event*);
    // syrk (real) and herk (complex): alpha and beta are real
    using Rk32 = int (*)(int, int, int, size_t, size_t, float, cl_mem, size_t, size_t, float,
                         cl_mem, size_t, size_t, cl_command_queue*, cl_event*);
    using Rk64 = int (*)(int, int, int, size_t, size_t, double, cl_mem, size_t, size_t, double,
                         cl_mem, size_t, size_t, cl_command_queue*, cl_event*);

    Gemm32 sgemm = nullptr;
    Gemm64 dgemm = nullptr;
    GemmC cgemm = nullptr;
    GemmZ zgemm = nullptr;
    Batched32 sgemm_batched = nullptr;
    Batched64 dgemm_batched = nullptr;
    BatchedC cgemm_batched = nullptr;
    BatchedZ zgemm_batched = nullptr;
    Trsm32 strsm = nullptr;
    Trsm64 dtrsm = nullptr;
    TrsmC ctrsm = nullptr;
    TrsmZ ztrsm = nullptr;
    Rk32 ssyrk = nullptr;
    Rk64 dsyrk = nullptr;
    Rk32 cherk = nullptr;
    Rk64 zherk = nullptr;
};

// CLBlast's routines, or null if its library is not found (first call
// loads it; any thread). `why` receives the reason.
const Clblast* load_clblast(std::string* why = nullptr);

} // namespace viproc::opencl
