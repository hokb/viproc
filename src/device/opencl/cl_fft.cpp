// VkFFT is third-party code: this file is compiled without viproc's warning
// flags (CMakeLists.txt). Keep it to the wrapper.
#include "device/opencl/cl_fft.hpp"

#define VKFFT_BACKEND 3
#include "vkFFT.h"

#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>

namespace viproc::opencl {

namespace {

struct Plan {
    // VkFFT keeps pointers to these: the plan's own, stable copies.
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue init_queue = nullptr;
    cl_mem init_buffer = nullptr;
    pfUINT buffer_size = 0;
    VkFFTApplication app = {};
    bool made = false;
    std::mutex run; // VkFFT sets the kernels' arguments per run
    ~Plan() {
        if (made) {
            deleteVkFFT(&app);
        }
    }
};

using Key = std::tuple<const Context*, int, bool, std::uint64_t, std::uint64_t>;

std::mutex g_mutex;
// Most recently used first; at most kPlans plans. Never destroyed (no
// OpenCL calls from static destructors at exit).
std::list<std::pair<Key, std::shared_ptr<Plan>>>& g_plans =
    *new std::list<std::pair<Key, std::shared_ptr<Plan>>>;
constexpr std::size_t kPlans = 64;

std::shared_ptr<Plan> plan_for(Context& ctx, cl_command_queue q, FftKind kind, bool dbl,
                               std::uint64_t n, std::uint64_t batch, Buffer& buf) {
    const bool real = kind == FftKind::RealForward || kind == FftKind::RealBackward;
    const Key key{&ctx, real ? 1 : 0, dbl, n, batch};
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto it = g_plans.begin(); it != g_plans.end(); ++it) {
        if (it->first == key) {
            g_plans.splice(g_plans.begin(), g_plans, it);
            return it->second;
        }
    }
    auto p = std::make_shared<Plan>();
    p->device = ctx.device();
    p->context = ctx.context();
    p->init_queue = q;
    p->init_buffer = buf.get();
    p->buffer_size = static_cast<pfUINT>(buf.bytes());
    VkFFTConfiguration c = {};
    c.FFTdim = 1;
    c.size[0] = n;
    c.numberBatches = batch;
    c.doublePrecision = dbl ? 1 : 0;
    c.useLUT = 1; // precomputed twiddles: accuracy close to pocketfft's
    c.performR2C = real ? 1 : 0;
    c.device = &p->device;
    c.context = &p->context;
    c.commandQueue = &p->init_queue;
    c.buffer = &p->init_buffer;
    c.bufferSize = &p->buffer_size;
    const VkFFTResult r = initializeVkFFT(&p->app, c);
    if (r != VKFFT_SUCCESS) {
        throw Error(std::string("VkFFT plan: ") + getVkFFTErrorString(r));
    }
    p->made = true;
    g_plans.emplace_front(key, p);
    if (g_plans.size() > kPlans) {
        g_plans.pop_back();
    }
    return p;
}

} // namespace

void fft(Context& ctx, cl_command_queue q, FftKind kind, bool dbl, std::uint64_t n,
         std::uint64_t batch, Buffer& buf) {
    std::shared_ptr<Plan> p = plan_for(ctx, q, kind, dbl, n, batch, buf);
    std::lock_guard<std::mutex> lock(p->run);
    VkFFTLaunchParams lp = {};
    lp.commandQueue = &q;
    lp.buffer = buf.ptr();
    const int dir = kind == FftKind::Forward || kind == FftKind::RealForward ? -1 : 1;
    const VkFFTResult r = VkFFTAppend(&p->app, dir, &lp);
    if (r != VKFFT_SUCCESS) {
        throw Error(std::string("VkFFT: ") + getVkFFTErrorString(r));
    }
    // The plan's kernels are shared: finish before another thread sets
    // their arguments.
    check(clFinish(q), "clFinish");
}

void drop_fft_plans(const Context* ctx) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_plans.remove_if([ctx](const auto& e) { return std::get<0>(e.first) == ctx; });
}

} // namespace viproc::opencl
