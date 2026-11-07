// The OpenCL device viproc offloads to (optimization level O4;
// docs/design/opencl.md).
//
// One device per process, chosen by select() from a spec ("gpu", "cpu",
// "none", a platform:device index pair, a device index or part of the
// device's name). A Context owns the device's OpenCL context and the
// programs built for it; device kernels keep a shared_ptr to the context
// they were made for, so selecting another device never invalidates a task
// in flight. Each thread that runs device work gets its own in-order command
// queue (queue()); no lock is held while the device computes.
#pragma once

#include "device/opencl/cl_loader.hpp"
#include "runtime/layout.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace viproc::opencl {

// An OpenCL call failed (out of device memory, a build error, ...). Device
// kernels catch it and run their CPU kernel instead.
struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Throws Error for a failed call: "<what>: OpenCL error <code>".
void check(cl_int status, const char* what);

class Context {
  public:
    Context(cl_platform_id platform, cl_device_id device);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    cl_context context() const { return context_; }
    cl_device_id device() const { return device_; }
    const std::string& name() const { return name_; }
    const std::string& platform_name() const { return platform_name_; }
    bool fp64() const { return fp64_; }
    bool is_gpu() const { return gpu_; }
    std::size_t max_work_group() const { return max_wg_; }
    std::uint64_t global_memory() const { return global_mem_; }
    std::uint64_t max_allocation() const { return max_alloc_; }
    // Unique per context in the process (thread queues are keyed by it).
    std::uint64_t id() const { return id_; }

    // The calling thread's command queue on this context (in order).
    cl_command_queue queue();
    // viproc's own kernels (linalg.cl) for `dtype` (Float32, Float64,
    // Complex64, Complex128), built on first use.
    cl_program program(DType dtype);
    // A work-group size for `kernel` (the device's and kernel's maximum,
    // at most `want`, a power of two).
    std::size_t work_group(cl_kernel kernel, std::size_t want);

  private:
    cl_context context_ = nullptr;
    cl_device_id device_ = nullptr;
    std::string name_, platform_name_;
    bool fp64_ = false, gpu_ = false;
    std::size_t max_wg_ = 1;
    std::uint64_t global_mem_ = 0, max_alloc_ = 0;
    std::uint64_t id_;
    std::mutex mutex_;
    cl_program programs_[4] = {};
    std::string build_errors_[4];
};

// A device buffer, released with the object.
class Buffer {
  public:
    Buffer() = default;
    Buffer(Context& ctx, std::size_t bytes);
    ~Buffer();
    Buffer(Buffer&& o) noexcept : mem_(o.mem_), bytes_(o.bytes_) { o.mem_ = nullptr; }
    Buffer& operator=(Buffer&& o) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    cl_mem get() const { return mem_; }
    cl_mem* ptr() { return &mem_; }
    std::size_t bytes() const { return bytes_; }

  private:
    cl_mem mem_ = nullptr;
    std::size_t bytes_ = 0;
};

void write(cl_command_queue q, Buffer& b, const void* src, std::size_t bytes);
void read(cl_command_queue q, const Buffer& b, void* dst, std::size_t bytes);

// A kernel object of one of viproc's programs, released with the object.
// Not shared between threads (clSetKernelArg).
class KernelCall {
  public:
    KernelCall(cl_program program, const char* name);
    ~KernelCall();
    KernelCall(const KernelCall&) = delete;
    KernelCall& operator=(const KernelCall&) = delete;
    template <class T> KernelCall& arg(const T& v) {
        check(clSetKernelArg(kernel_, next_++, sizeof(T), &v), "clSetKernelArg");
        return *this;
    }
    KernelCall& local(std::size_t bytes) {
        check(clSetKernelArg(kernel_, next_++, bytes, nullptr), "clSetKernelArg");
        return *this;
    }
    cl_kernel get() const { return kernel_; }
    // 1-d range of `groups` work-groups of `wg` work-items.
    void run(cl_command_queue q, std::size_t groups, std::size_t wg);

  private:
    cl_kernel kernel_ = nullptr;
    cl_uint next_ = 0;
};

// The selected device: null if none (not selected, "none", no OpenCL, no
// matching device). Any thread.
std::shared_ptr<Context> current();

// Selects the device for spec (null or "": env VIPROC_OPENCL_DEVICE, else
// "gpu"). Returns the context or null; `why` receives the reason for null.
// Main thread; device kernels issued before keep their context.
std::shared_ptr<Context> select(const char* spec, std::string* why);

// Ops offloaded smaller than this (in the op's flop estimate) run their CPU
// kernel (env VIPROC_OPENCL_MIN_WORK; default 2^24, about a 256^3 product).
std::int64_t min_work();
void set_min_work(std::int64_t w);

// Counters (offload_stats): ops a device kernel ran on the device, and ones
// it handed to its CPU kernel because the device failed (Error).
extern std::atomic<std::uint64_t> device_ops;
extern std::atomic<std::uint64_t> device_fallbacks;

} // namespace viproc::opencl
