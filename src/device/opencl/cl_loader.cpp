#include "device/opencl/cl_loader.hpp"

#include <cstdlib>
#include <mutex>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace viproc::opencl {

namespace {

// The platform's shared-library functions: one implementation per platform.
void* open_library(const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(LoadLibraryA(name));
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void* library_symbol(void* lib, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

void* open_first(std::initializer_list<const char*> names) {
    for (const char* n : names) {
        if (void* lib = open_library(n)) {
            return lib;
        }
    }
    return nullptr;
}

} // namespace

// The OpenCL functions viproc and VkFFT call: (return type, name, parameters,
// arguments). Each gets a pointer, loaded by load_opencl(), and a forwarder
// with the function's own name below.
#define VIPROC_CL_FUNCTIONS(X)                                                                     \
    X(cl_int, clGetPlatformIDs, (cl_uint a, cl_platform_id * b, cl_uint * c), (a, b, c))           \
    X(cl_int, clGetPlatformInfo,                                                                   \
      (cl_platform_id a, cl_platform_info b, size_t c, void* d, size_t* e), (a, b, c, d, e))       \
    X(cl_int, clGetDeviceIDs,                                                                      \
      (cl_platform_id a, cl_device_type b, cl_uint c, cl_device_id * d, cl_uint * e),              \
      (a, b, c, d, e))                                                                             \
    X(cl_int, clGetDeviceInfo, (cl_device_id a, cl_device_info b, size_t c, void* d, size_t* e),   \
      (a, b, c, d, e))                                                                             \
    X(cl_context, clCreateContext,                                                                 \
      (const cl_context_properties* a, cl_uint b, const cl_device_id* c,                           \
       void(CL_CALLBACK * d)(const char*, const void*, size_t, void*), void* e, cl_int* f),        \
      (a, b, c, d, e, f))                                                                          \
    X(cl_int, clReleaseContext, (cl_context a), (a))                                               \
    X(cl_command_queue, clCreateCommandQueue,                                                      \
      (cl_context a, cl_device_id b, cl_command_queue_properties c, cl_int * d), (a, b, c, d))     \
    X(cl_int, clReleaseCommandQueue, (cl_command_queue a), (a))                                    \
    X(cl_mem, clCreateBuffer, (cl_context a, cl_mem_flags b, size_t c, void* d, cl_int* e),        \
      (a, b, c, d, e))                                                                             \
    X(cl_int, clReleaseMemObject, (cl_mem a), (a))                                                 \
    X(cl_int, clEnqueueWriteBuffer,                                                                \
      (cl_command_queue a, cl_mem b, cl_bool c, size_t d, size_t e, const void* f, cl_uint g,      \
       const cl_event* h, cl_event* i),                                                            \
      (a, b, c, d, e, f, g, h, i))                                                                 \
    X(cl_int, clEnqueueReadBuffer,                                                                 \
      (cl_command_queue a, cl_mem b, cl_bool c, size_t d, size_t e, void* f, cl_uint g,            \
       const cl_event* h, cl_event* i),                                                            \
      (a, b, c, d, e, f, g, h, i))                                                                 \
    X(cl_program, clCreateProgramWithSource,                                                       \
      (cl_context a, cl_uint b, const char** c, const size_t* d, cl_int* e), (a, b, c, d, e))      \
    X(cl_program, clCreateProgramWithBinary,                                                       \
      (cl_context a, cl_uint b, const cl_device_id* c, const size_t* d, const unsigned char** e,   \
       cl_int* f, cl_int* g),                                                                      \
      (a, b, c, d, e, f, g))                                                                       \
    X(cl_int, clBuildProgram,                                                                      \
      (cl_program a, cl_uint b, const cl_device_id* c, const char* d,                              \
       void(CL_CALLBACK * e)(cl_program, void*), void* f),                                         \
      (a, b, c, d, e, f))                                                                          \
    X(cl_int, clGetProgramBuildInfo,                                                               \
      (cl_program a, cl_device_id b, cl_program_build_info c, size_t d, void* e, size_t* f),       \
      (a, b, c, d, e, f))                                                                          \
    X(cl_int, clGetProgramInfo, (cl_program a, cl_program_info b, size_t c, void* d, size_t* e),   \
      (a, b, c, d, e))                                                                             \
    X(cl_int, clReleaseProgram, (cl_program a), (a))                                               \
    X(cl_kernel, clCreateKernel, (cl_program a, const char* b, cl_int* c), (a, b, c))              \
    X(cl_int, clReleaseKernel, (cl_kernel a), (a))                                                 \
    X(cl_int, clSetKernelArg, (cl_kernel a, cl_uint b, size_t c, const void* d), (a, b, c, d))     \
    X(cl_int, clGetKernelWorkGroupInfo,                                                            \
      (cl_kernel a, cl_device_id b, cl_kernel_work_group_info c, size_t d, void* e, size_t* f),    \
      (a, b, c, d, e, f))                                                                          \
    X(cl_int, clEnqueueNDRangeKernel,                                                              \
      (cl_command_queue a, cl_kernel b, cl_uint c, const size_t* d, const size_t* e,               \
       const size_t* f, cl_uint g, const cl_event* h, cl_event* i),                                \
      (a, b, c, d, e, f, g, h, i))                                                                 \
    X(cl_int, clFinish, (cl_command_queue a), (a))                                                 \
    X(cl_int, clWaitForEvents, (cl_uint a, const cl_event* b), (a, b))                             \
    X(cl_int, clReleaseEvent, (cl_event a), (a))

namespace {

struct ClTable {
#define VIPROC_CL_POINTER(ret, name, params, args) ret(CL_API_CALL* name) params = nullptr;
    VIPROC_CL_FUNCTIONS(VIPROC_CL_POINTER)
#undef VIPROC_CL_POINTER
};

ClTable g_cl;
bool g_cl_loaded = false;
std::string g_cl_why;
std::once_flag g_cl_once;

void load_cl_once() {
    void* lib = nullptr;
    if (const char* path = std::getenv("VIPROC_OPENCL_LIBRARY"); path != nullptr && *path) {
        lib = open_library(path);
    }
    if (lib == nullptr) {
#ifdef _WIN32
        lib = open_first({"OpenCL.dll"});
#elif defined(__APPLE__)
        lib = open_first({"/System/Library/Frameworks/OpenCL.framework/OpenCL"});
#else
        lib = open_first({"libOpenCL.so.1", "libOpenCL.so"});
#endif
    }
    if (lib == nullptr) {
        g_cl_why = "no OpenCL ICD loader (libOpenCL.so.1 / OpenCL.dll) found";
        return;
    }
    bool ok = true;
#define VIPROC_CL_LOAD(ret, name, params, args)                                                    \
    g_cl.name = reinterpret_cast<ret(CL_API_CALL*) params>(library_symbol(lib, #name));            \
    if (g_cl.name == nullptr) {                                                                    \
        ok = false;                                                                                \
        g_cl_why = "the OpenCL library lacks " #name;                                              \
    }
    VIPROC_CL_FUNCTIONS(VIPROC_CL_LOAD)
#undef VIPROC_CL_LOAD
    g_cl_loaded = ok;
}

Clblast g_blas;
const Clblast* g_blas_ptr = nullptr;
std::string g_blas_why;
std::once_flag g_blas_once;

template <class F> bool resolve(void* lib, F& f, const char* name) {
    f = reinterpret_cast<F>(library_symbol(lib, name));
    if (f == nullptr) {
        g_blas_why = std::string("CLBlast lacks ") + name;
    }
    return f != nullptr;
}

void load_clblast_once() {
    void* lib = nullptr;
    if (const char* path = std::getenv("VIPROC_CLBLAST"); path != nullptr && *path) {
        lib = open_library(path);
    }
    if (lib == nullptr) {
#ifdef _WIN32
        lib = open_first({"clblast.dll", "libclblast.dll"});
#elif defined(__APPLE__)
        lib = open_first({"libclblast.dylib", "libclblast.1.dylib"});
#else
        lib = open_first({"libclblast.so.1", "libclblast.so"});
#endif
    }
    if (lib == nullptr) {
        g_blas_why = "CLBlast not found (libclblast.so.1 / clblast.dll; env VIPROC_CLBLAST)";
        return;
    }
    Clblast& b = g_blas;
    const bool ok =
        resolve(lib, b.sgemm, "CLBlastSgemm") && resolve(lib, b.dgemm, "CLBlastDgemm") &&
        resolve(lib, b.cgemm, "CLBlastCgemm") && resolve(lib, b.zgemm, "CLBlastZgemm") &&
        resolve(lib, b.sgemm_batched, "CLBlastSgemmStridedBatched") &&
        resolve(lib, b.dgemm_batched, "CLBlastDgemmStridedBatched") &&
        resolve(lib, b.cgemm_batched, "CLBlastCgemmStridedBatched") &&
        resolve(lib, b.zgemm_batched, "CLBlastZgemmStridedBatched") &&
        resolve(lib, b.strsm, "CLBlastStrsm") && resolve(lib, b.dtrsm, "CLBlastDtrsm") &&
        resolve(lib, b.ctrsm, "CLBlastCtrsm") && resolve(lib, b.ztrsm, "CLBlastZtrsm") &&
        resolve(lib, b.ssyrk, "CLBlastSsyrk") && resolve(lib, b.dsyrk, "CLBlastDsyrk") &&
        resolve(lib, b.cherk, "CLBlastCherk") && resolve(lib, b.zherk, "CLBlastZherk");
    if (ok) {
        g_blas_ptr = &g_blas;
    }
}

} // namespace

bool load_opencl(std::string* why) {
    std::call_once(g_cl_once, load_cl_once);
    if (!g_cl_loaded && why != nullptr) {
        *why = g_cl_why;
    }
    return g_cl_loaded;
}

const Clblast* load_clblast(std::string* why) {
    if (!load_opencl(why)) {
        return nullptr;
    }
    std::call_once(g_blas_once, load_clblast_once);
    if (g_blas_ptr == nullptr && why != nullptr) {
        *why = g_blas_why;
    }
    return g_blas_ptr;
}

} // namespace viproc::opencl

// The forwarders. Hidden on ELF targets: they serve viproc's own calls (and
// VkFFT's, compiled into viproc) and never another module's.
#if defined(__GNUC__) && !defined(_WIN32)
#define VIPROC_CL_HIDDEN __attribute__((visibility("hidden")))
#else
#define VIPROC_CL_HIDDEN
#endif

extern "C" {
#define VIPROC_CL_FORWARD(ret, name, params, args)                                                 \
    VIPROC_CL_HIDDEN ret CL_API_CALL name params { return viproc::opencl::g_cl.name args; }
VIPROC_CL_FUNCTIONS(VIPROC_CL_FORWARD)
#undef VIPROC_CL_FORWARD
}
