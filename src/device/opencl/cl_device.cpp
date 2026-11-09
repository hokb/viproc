#include "device/opencl/cl_device.hpp"

#include "device/opencl/cl_fft.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace viproc::opencl {

extern const char* const kLinalgSource; // cl_linalg_source.cpp

std::atomic<std::uint64_t> device_ops{0};
std::atomic<std::uint64_t> device_fallbacks{0};

void check(cl_int status, const char* what) {
    if (status != CL_SUCCESS) {
        throw Error(std::string(what) + ": OpenCL error " + std::to_string(status));
    }
}

namespace {

std::string device_string(cl_device_id d, cl_device_info what) {
    std::size_t n = 0;
    if (clGetDeviceInfo(d, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) {
        return {};
    }
    std::string s(n, '\0');
    clGetDeviceInfo(d, what, n, s.data(), nullptr);
    while (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    return s;
}

std::string platform_string(cl_platform_id p, cl_platform_info what) {
    std::size_t n = 0;
    if (clGetPlatformInfo(p, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) {
        return {};
    }
    std::string s(n, '\0');
    clGetPlatformInfo(p, what, n, s.data(), nullptr);
    while (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    return s;
}

template <class T> T device_value(cl_device_id d, cl_device_info what) {
    T v{};
    clGetDeviceInfo(d, what, sizeof v, &v, nullptr);
    return v;
}

// Context generations: a thread's queue belongs to one context.
struct ThreadQueue {
    std::uint64_t ctx = 0;
    cl_command_queue queue = nullptr;
    ~ThreadQueue() {
        if (queue != nullptr) {
            clReleaseCommandQueue(queue);
        }
    }
};

thread_local ThreadQueue t_queue;
std::atomic<std::uint64_t> g_context_ids{0};

std::mutex g_select_mutex;
// Never destroyed: releasing OpenCL objects from static destructors at exit
// races with the OpenCL implementation's own teardown.
std::shared_ptr<Context>& g_current = *new std::shared_ptr<Context>;

std::string lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

} // namespace

Context::Context(cl_platform_id platform, cl_device_id device)
    : device_(device), id_(++g_context_ids) {
    cl_int err = CL_SUCCESS;
    const cl_context_properties props[] = {CL_CONTEXT_PLATFORM,
                                           reinterpret_cast<cl_context_properties>(platform), 0};
    context_ = clCreateContext(props, 1, &device_, nullptr, nullptr, &err);
    check(err, "clCreateContext");
    name_ = device_string(device, CL_DEVICE_NAME);
    platform_name_ = platform_string(platform, CL_PLATFORM_NAME);
    const std::string ext = device_string(device, CL_DEVICE_EXTENSIONS);
    fp64_ = ext.find("cl_khr_fp64") != std::string::npos ||
            device_value<cl_device_fp_config>(device, CL_DEVICE_DOUBLE_FP_CONFIG) != 0;
    gpu_ = (device_value<cl_device_type>(device, CL_DEVICE_TYPE) &
            (CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR)) != 0;
    max_wg_ =
        std::max<std::size_t>(device_value<std::size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE), 1);
    global_mem_ = device_value<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
    max_alloc_ = device_value<cl_ulong>(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    compute_units_ =
        std::max<cl_uint>(device_value<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS), 1);
    clock_mhz_ = std::max<cl_uint>(device_value<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY), 1);
}

Context::~Context() {
    drop_fft_plans(this);
    for (cl_program p : programs_) {
        if (p != nullptr) {
            clReleaseProgram(p);
        }
    }
    if (context_ != nullptr) {
        clReleaseContext(context_);
    }
}

cl_command_queue Context::queue() {
    if (t_queue.ctx != id_) {
        if (t_queue.queue != nullptr) {
            clReleaseCommandQueue(t_queue.queue);
            t_queue.queue = nullptr;
        }
        cl_int err = CL_SUCCESS;
        t_queue.queue = clCreateCommandQueue(context_, device_, 0, &err);
        check(err, "clCreateCommandQueue");
        t_queue.ctx = id_;
    }
    return t_queue.queue;
}

cl_program Context::program(DType dtype) {
    int i = 0;
    const char* define = nullptr;
    switch (dtype) {
    case DType::Float32:
        i = 0, define = "-DVP_F32";
        break;
    case DType::Float64:
        i = 1, define = "-DVP_F64";
        break;
    case DType::Complex64:
        i = 2, define = "-DVP_C64";
        break;
    case DType::Complex128:
        i = 3, define = "-DVP_C128";
        break;
    default:
        throw Error("no device program for this dtype");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (programs_[i] != nullptr) {
        return programs_[i];
    }
    if (!build_errors_[i].empty()) {
        throw Error(build_errors_[i]);
    }
    cl_int err = CL_SUCCESS;
    const char* src = kLinalgSource;
    cl_program p = clCreateProgramWithSource(context_, 1, &src, nullptr, &err);
    check(err, "clCreateProgramWithSource");
    err = clBuildProgram(p, 1, &device_, define, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::size_t n = 0;
        clGetProgramBuildInfo(p, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        clGetProgramBuildInfo(p, device_, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        clReleaseProgram(p);
        build_errors_[i] = "building viproc's OpenCL kernels failed: " + log;
        throw Error(build_errors_[i]);
    }
    programs_[i] = p;
    return p;
}

std::size_t Context::work_group(cl_kernel kernel, std::size_t want) {
    std::size_t k = max_wg_;
    std::size_t kmax = 0;
    if (clGetKernelWorkGroupInfo(kernel, device_, CL_KERNEL_WORK_GROUP_SIZE, sizeof kmax, &kmax,
                                 nullptr) == CL_SUCCESS &&
        kmax > 0) {
        k = std::min(k, kmax);
    }
    k = std::min(k, want);
    std::size_t p = 1;
    while (p * 2 <= k) {
        p *= 2;
    }
    return p;
}

Buffer::Buffer(Context& ctx, std::size_t bytes) : bytes_(bytes) {
    if (bytes > ctx.max_allocation()) {
        throw Error("buffer larger than the device's largest allocation");
    }
    cl_int err = CL_SUCCESS;
    mem_ = clCreateBuffer(ctx.context(), CL_MEM_READ_WRITE, std::max<std::size_t>(bytes, 16),
                          nullptr, &err);
    check(err, "clCreateBuffer");
}

Buffer::~Buffer() {
    if (mem_ != nullptr) {
        clReleaseMemObject(mem_);
    }
}

Buffer& Buffer::operator=(Buffer&& o) noexcept {
    if (this != &o) {
        if (mem_ != nullptr) {
            clReleaseMemObject(mem_);
        }
        mem_ = o.mem_;
        bytes_ = o.bytes_;
        o.mem_ = nullptr;
    }
    return *this;
}

void write(cl_command_queue q, Buffer& b, const void* src, std::size_t bytes) {
    if (bytes > 0) {
        // blocking: the host staging memory is freed right after
        check(clEnqueueWriteBuffer(q, b.get(), CL_TRUE, 0, bytes, src, 0, nullptr, nullptr),
              "clEnqueueWriteBuffer");
    }
}

void read(cl_command_queue q, const Buffer& b, void* dst, std::size_t bytes) {
    if (bytes > 0) {
        check(clEnqueueReadBuffer(q, b.get(), CL_TRUE, 0, bytes, dst, 0, nullptr, nullptr),
              "clEnqueueReadBuffer");
    }
}

KernelCall::KernelCall(cl_program program, const char* name) {
    cl_int err = CL_SUCCESS;
    kernel_ = clCreateKernel(program, name, &err);
    check(err, "clCreateKernel");
}

KernelCall::~KernelCall() {
    if (kernel_ != nullptr) {
        clReleaseKernel(kernel_);
    }
}

void KernelCall::run(cl_command_queue q, std::size_t groups, std::size_t wg) {
    if (groups == 0) {
        return;
    }
    const std::size_t global = groups * wg;
    check(clEnqueueNDRangeKernel(q, kernel_, 1, nullptr, &global, &wg, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel");
}

std::shared_ptr<Context> current() {
    std::lock_guard<std::mutex> lock(g_select_mutex);
    return g_current;
}

std::shared_ptr<Context> select(const char* spec_in, std::string* why) {
    std::string spec = spec_in != nullptr ? spec_in : "";
    if (spec.empty()) {
        const char* env = std::getenv("VIPROC_OPENCL_DEVICE");
        spec = env != nullptr && *env ? env : "gpu";
    }
    spec = lower(spec);
    std::lock_guard<std::mutex> lock(g_select_mutex);
    g_current.reset();
    auto fail = [&](std::string w) {
        if (why != nullptr) {
            *why = std::move(w);
        }
        return std::shared_ptr<Context>();
    };
    if (spec == "none" || spec == "off" || spec == "0" || spec == "false") {
        return fail("no device selected (" + spec + ")");
    }
    std::string w;
    if (!load_opencl(&w)) {
        return fail(w);
    }
    cl_uint np = 0;
    if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) {
        return fail("no OpenCL platform");
    }
    std::vector<cl_platform_id> platforms(np);
    clGetPlatformIDs(np, platforms.data(), nullptr);
    struct Candidate {
        cl_platform_id p;
        cl_device_id d;
        std::size_t pi, di, flat;
        cl_device_type type;
        std::string name;
    };
    std::vector<Candidate> all;
    for (std::size_t pi = 0; pi < platforms.size(); ++pi) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS ||
            nd == 0) {
            continue;
        }
        std::vector<cl_device_id> devs(nd);
        clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
        for (std::size_t di = 0; di < devs.size(); ++di) {
            all.push_back(Candidate{platforms[pi], devs[di], pi, di, all.size(),
                                    device_value<cl_device_type>(devs[di], CL_DEVICE_TYPE),
                                    lower(device_string(devs[di], CL_DEVICE_NAME))});
        }
    }
    const Candidate* pick = nullptr;
    for (const Candidate& c : all) {
        bool match = false;
        if (spec == "gpu") {
            match = (c.type & (CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR)) != 0;
        } else if (spec == "cpu") {
            match = (c.type & CL_DEVICE_TYPE_CPU) != 0;
        } else if (spec == "any") {
            match = true;
        } else if (const std::size_t colon = spec.find(':');
                   colon != std::string::npos &&
                   std::isdigit(static_cast<unsigned char>(spec[0]))) {
            match = std::to_string(c.pi) + ":" + std::to_string(c.di) == spec;
        } else if (std::all_of(spec.begin(), spec.end(), [](char ch) {
                       return std::isdigit(static_cast<unsigned char>(ch));
                   })) {
            match = std::to_string(c.flat) == spec;
        } else {
            match = c.name.find(spec) != std::string::npos;
        }
        if (match) {
            pick = &c;
            break;
        }
    }
    if (pick == nullptr) {
        return fail("no OpenCL device matches \"" + spec + "\"");
    }
    try {
        g_current = std::make_shared<Context>(pick->p, pick->d);
    } catch (const Error& e) {
        return fail(e.what());
    }
    return g_current;
}

} // namespace viproc::opencl
