#include "device/opencl/cl_policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace viproc::opencl {

OpClass op_class(Op op) {
    switch (op) {
    case Op::Matmul:
        return OpClass::Gemm;
    case Op::Det:
    case Op::Slogdet:
    case Op::Inv:
    case Op::Solve:
    case Op::Solve1:
    case Op::CholeskyLo:
    case Op::CholeskyUp:
        return OpClass::Factor;
    default:
        return OpClass::Fft;
    }
}

namespace {

int dtype_index(DType d) {
    switch (d) {
    case DType::Float32:
        return 0;
    case DType::Float64:
        return 1;
    case DType::Complex64:
        return 2;
    default:
        return 3;
    }
}

bool is_double(DType d) { return d == DType::Float64 || d == DType::Complex128; }
bool is_complex(DType d) { return d == DType::Complex64 || d == DType::Complex128; }

// Work units per ns of a class and precision for real dtypes; complex gemm
// and factorizations do 4 real multiply-adds per unit.
double for_dtype(const double rate[3][2], OpClass c, DType d) {
    double r = rate[static_cast<int>(c)][is_double(d) ? 1 : 0];
    if (is_complex(d) && c != OpClass::Fft) {
        r /= 4;
    }
    return r;
}

int work_class(std::int64_t work) {
    int c = 0;
    while (c < 63 && (std::int64_t{2} << c) <= work) {
        ++c;
    }
    return c;
}

double now_ns() {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}

constexpr std::uint32_t kExploreFirst = 8, kExploreMax = 1024;
constexpr std::uint32_t kRetryFirst = 16, kRetryMax = 1024;
// An exploration runs the other side only if its estimate is within this
// factor of the chosen one.
constexpr double kExploreWithin = 4;

} // namespace

void DevicePolicy::configure(Context& ctx, bool measure) {
    if (const char* env = std::getenv("VIPROC_OPENCL_RESIDENT"); env != nullptr && *env) {
        set_resident(std::strcmp(env, "0") != 0);
    }
    DeviceProfile p;
    // CPU, one core running NumPy's loop (OpenBLAS with one thread from O2
    // on, LAPACK, pocketfft), work units per ns: gemm ~16/32 GFLOP/s, LU
    // (n^3 units, n^3/3 multiply-adds) ~10/20 GFLOP/s, fft (4 n log2 n
    // units) ~2-3 GFLOP/s.
    p.cpu_rate[0][0] = 16, p.cpu_rate[0][1] = 8;
    p.cpu_rate[1][0] = 30, p.cpu_rate[1][1] = 15;
    p.cpu_rate[2][0] = 3, p.cpu_rate[2][1] = 2;
    // Device: peak multiply-adds per ns from its properties (GPU: 64 lanes
    // per compute unit, fp64 at 1/16; CPU device: 8 lanes per thread, fp64
    // at 1/2), times what each class reaches of it (CLBlast's gemm ~40%,
    // viproc's factorization kernels and VkFFT on the host-resident path
    // far less).
    const double ghz = ctx.clock_mhz() / 1000.0;
    const double lanes = ctx.is_gpu() ? 64 : 8;
    const double peak = ctx.compute_units() * ghz * lanes;
    const double dp = ctx.is_gpu() ? 1.0 / 16 : 0.5;
    const double eff[3] = {0.4, 0.05, 0.05};
    for (int c = 0; c < 3; ++c) {
        p.device_rate[c][0] = std::max(peak * eff[c], 1e-3);
        p.device_rate[c][1] = std::max(peak * eff[c] * dp, 1e-3);
    }
    if (measure) {
        try {
            cl_command_queue q = ctx.queue();
            // Round trip of a tiny transfer: the fixed cost of an op.
            Buffer tiny(ctx, 64);
            char b[64] = {};
            double best = 1e30;
            for (int i = 0; i < 5; ++i) {
                const double t0 = now_ns();
                write(q, tiny, b, sizeof b);
                read(q, tiny, b, sizeof b);
                best = std::min(best, now_ns() - t0);
            }
            p.latency_ns = std::max(best, 1000.0);
            // Bandwidths: 16 MB (at most a quarter of the largest
            // allocation), best of three.
            const std::size_t bytes = static_cast<std::size_t>(
                std::min<std::uint64_t>(std::uint64_t{16} << 20, ctx.max_allocation() / 4));
            if (bytes >= (std::size_t{1} << 20)) {
                std::vector<char> host(bytes, 1);
                Buffer big(ctx, bytes);
                double up = 1e30, down = 1e30;
                for (int i = 0; i < 3; ++i) {
                    double t0 = now_ns();
                    write(q, big, host.data(), bytes);
                    up = std::min(up, now_ns() - t0);
                    t0 = now_ns();
                    read(q, big, host.data(), bytes);
                    down = std::min(down, now_ns() - t0);
                }
                p.upload_bpns = static_cast<double>(bytes) / std::max(up, 1.0);
                p.download_bpns = static_cast<double>(bytes) / std::max(down, 1.0);
            }
            p.measured = true;
        } catch (const Error&) {
            // keep the defaults
        }
    }
    set_profile(p);
}

void DevicePolicy::set_profile(const DeviceProfile& p) {
    std::lock_guard<std::mutex> lock(mutex_);
    profile_ = p;
    latency_ns_.store(p.latency_ns, std::memory_order_relaxed);
    double fastest = 1e30;
    for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 2; ++d) {
            fastest = std::min(fastest, 1.0 / std::max(p.cpu_rate[c][d], 1e-9));
        }
    }
    // A rate measured later may be faster than the prior: half of it.
    fastest_cpu_ns_per_work_.store(fastest / 2, std::memory_order_relaxed);
    for (int i = 0; i < kOps * kDtypes * kClasses; ++i) {
        classes_[i] = Class{};
    }
    for (auto& per_op : failures_) {
        for (Failure& f : per_op) {
            f.bytes.store(UINT64_MAX, std::memory_order_relaxed);
            f.skips.store(0, std::memory_order_relaxed);
            f.retry_every.store(kRetryFirst, std::memory_order_relaxed);
        }
    }
    n_device_ = 0, n_cpu_ = 0, n_explored_ = 0, n_skipped_ = 0;
}

void DevicePolicy::set_min_work(std::int64_t w) {
    if (w < 0) {
        min_work_.store(-2, std::memory_order_relaxed); // back to the environment's
    } else {
        min_work_.store(w, std::memory_order_relaxed);
    }
}

std::int64_t DevicePolicy::min_work() const {
    std::int64_t w = min_work_.load(std::memory_order_relaxed);
    if (w == -2) {
        const char* env = std::getenv("VIPROC_OPENCL_MIN_WORK");
        w = env != nullptr && *env ? std::max<std::int64_t>(std::atoll(env), 0) : -1;
        min_work_.store(w, std::memory_order_relaxed);
    }
    return w;
}

std::int64_t DevicePolicy::threshold() const {
    const std::int64_t w = min_work();
    if (w >= 0) {
        return w;
    }
    return deterministic_.load(std::memory_order_relaxed) ? kDefaultMinWork : -1;
}

DeviceProfile DevicePolicy::profile() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return profile_;
}

PolicyCounts DevicePolicy::counts() const {
    return {n_device_.load(std::memory_order_relaxed), n_cpu_.load(std::memory_order_relaxed),
            n_explored_.load(std::memory_order_relaxed),
            n_skipped_.load(std::memory_order_relaxed)};
}

DevicePolicy::Class& DevicePolicy::cls(const OpRequest& r) {
    const int op = std::clamp(static_cast<int>(r.op), 0, kOps - 1);
    return classes_[(op * kDtypes + dtype_index(r.dtype)) * kClasses + work_class(r.work)];
}

double DevicePolicy::fixed_ns(const OpRequest& r) const {
    const std::uint64_t up = r.bytes_in - std::min(r.bytes_resident, r.bytes_in);
    const std::uint64_t down = r.output_resident ? 0 : r.bytes_out;
    return profile_.latency_ns + static_cast<double>(up) / profile_.upload_bpns +
           static_cast<double>(down) / profile_.download_bpns;
}

double DevicePolicy::cpu_fixed_ns(const OpRequest& r) const {
    return static_cast<double>(r.bytes_stale) / profile_.download_bpns;
}

double DevicePolicy::device_ns_per_work(const OpRequest& r, const Class& c) const {
    if (c.device.samples != 0) {
        return c.device.ns_per_work;
    }
    return 1.0 / for_dtype(profile_.device_rate, op_class(r.op), r.dtype);
}

double DevicePolicy::cpu_ns_per_work(const OpRequest& r, const Class& c) const {
    if (c.cpu.samples != 0) {
        return c.cpu.ns_per_work;
    }
    return 1.0 / for_dtype(profile_.cpu_rate, op_class(r.op), r.dtype);
}

void DevicePolicy::estimate(const OpRequest& r, double* device_ns, double* cpu_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    const Class& c = cls(r);
    const double w = static_cast<double>(r.work);
    const double compute = w * device_ns_per_work(r, c);
    *device_ns = fixed_ns(r) + compute + wait_ns(compute);
    *cpu_ns = cpu_fixed_ns(r) + w * cpu_ns_per_work(r, c);
}

double DevicePolicy::wait_ns(double compute_ns) const {
    const int n = in_flight_.load(std::memory_order_relaxed);
    if (n <= 0) {
        return 0;
    }
    const double backlog = std::max(backlog_ns_.load(std::memory_order_relaxed), 0.0);
    return std::min(backlog, n * compute_ns);
}

void DevicePolicy::begin(OpRequest& r) {
    if (r.compute_ns <= 0 && r.work > 0) {
        // threshold mode, a retry, below the cheap first test: not estimated yet
        std::lock_guard<std::mutex> lock(mutex_);
        r.compute_ns = static_cast<double>(r.work) * device_ns_per_work(r, cls(r));
    }
    r.wait_ns = wait_ns(r.compute_ns);
    r.sharing = in_flight_.fetch_add(1, std::memory_order_relaxed);
    backlog_ns_.fetch_add(r.compute_ns, std::memory_order_relaxed);
}

void DevicePolicy::end(const OpRequest& r) {
    backlog_ns_.fetch_sub(r.compute_ns, std::memory_order_relaxed);
    in_flight_.fetch_sub(1, std::memory_order_relaxed);
}

bool DevicePolicy::use_device(OpRequest& r, bool* explore, bool* timed) {
    if (explore != nullptr) {
        *explore = false;
    }
    if (timed != nullptr) {
        *timed = false;
    }
    // A failure at this size or below: the CPU, except a retry now and then.
    Failure& f = failures_[std::clamp(static_cast<int>(r.op), 0, kOps - 1)][dtype_index(r.dtype)];
    if (r.bytes_in + r.bytes_out >= f.bytes.load(std::memory_order_relaxed)) {
        const std::uint32_t s = f.skips.fetch_add(1, std::memory_order_relaxed) + 1;
        if (s % f.retry_every.load(std::memory_order_relaxed) != 0) {
            n_skipped_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        n_device_.fetch_add(1, std::memory_order_relaxed);
        return true; // the retry
    }
    if (const std::int64_t t = threshold(); t >= 0) {
        const bool dev = r.work >= t;
        (dev ? n_device_ : n_cpu_).fetch_add(1, std::memory_order_relaxed);
        return dev;
    }
    // Cheap first test, no lock: the CPU's fastest rate beats the device's
    // round trip alone (inputs on the device excepted: the CPU would
    // download them).
    if (r.bytes_stale == 0 &&
        static_cast<double>(r.work) * fastest_cpu_ns_per_work_.load(std::memory_order_relaxed) <
            latency_ns_.load(std::memory_order_relaxed)) {
        n_cpu_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    Class& c = cls(r);
    const double w = static_cast<double>(r.work);
    r.compute_ns = w * device_ns_per_work(r, c);
    const double dev_ns = fixed_ns(r) + r.compute_ns + wait_ns(r.compute_ns);
    const double cpu_ns = cpu_fixed_ns(r) + w * cpu_ns_per_work(r, c);
    bool dev = dev_ns < cpu_ns;
    if (timed != nullptr) {
        *timed = true;
    }
    // Explorations keep the side not chosen measured (rates change: other
    // load, a size class first seen): every explore_every-th decision of
    // the class, if the other side is within kExploreWithin; the period
    // doubles while the choice stays, and restarts when it flips.
    if (++c.decisions % c.explore_every == 0) {
        const double chosen = dev ? dev_ns : cpu_ns, other = dev ? cpu_ns : dev_ns;
        if (dev == c.last_device) {
            c.explore_every = std::min(c.explore_every * 2, kExploreMax);
        } else {
            c.explore_every = kExploreFirst;
        }
        c.last_device = dev;
        if (other < chosen * kExploreWithin) {
            dev = !dev;
            if (explore != nullptr) {
                *explore = true;
            }
            n_explored_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    (dev ? n_device_ : n_cpu_).fetch_add(1, std::memory_order_relaxed);
    return dev;
}

void DevicePolicy::record(const OpRequest& r, bool device, double ns) {
    if (r.work <= 0 || ns <= 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    Class& c = cls(r);
    const double w = static_cast<double>(r.work);
    double v = 0;
    if (device) {
        // The compute part, as if the device were idle: without the wait
        // for the ops in flight at the start (estimated), at least the fair
        // share of the measured time.
        const double compute = std::max(ns - fixed_ns(r), 0.1 * ns);
        v = std::max(compute - r.wait_ns, compute / (1 + r.sharing)) / w;
    } else {
        v = std::max(ns - cpu_fixed_ns(r), 0.1 * ns) / w;
        const double half = v / 2;
        if (half < fastest_cpu_ns_per_work_.load(std::memory_order_relaxed)) {
            fastest_cpu_ns_per_work_.store(half, std::memory_order_relaxed);
        }
    }
    Rate& rate = device ? c.device : c.cpu;
    rate.ns_per_work = rate.samples == 0 ? v : 0.75 * rate.ns_per_work + 0.25 * v;
    ++rate.samples;
}

void DevicePolicy::failed(const OpRequest& r) {
    Failure& f = failures_[std::clamp(static_cast<int>(r.op), 0, kOps - 1)][dtype_index(r.dtype)];
    const std::uint64_t bytes = r.bytes_in + r.bytes_out;
    std::uint64_t cur = f.bytes.load(std::memory_order_relaxed);
    if (bytes >= cur) {
        // a retry failed again: retry less often
        const std::uint32_t e = f.retry_every.load(std::memory_order_relaxed);
        f.retry_every.store(std::min(e * 2, kRetryMax), std::memory_order_relaxed);
        return;
    }
    while (bytes < cur && !f.bytes.compare_exchange_weak(cur, bytes, std::memory_order_relaxed)) {
    }
}

void DevicePolicy::succeeded(const OpRequest& r) {
    Failure& f = failures_[std::clamp(static_cast<int>(r.op), 0, kOps - 1)][dtype_index(r.dtype)];
    if (r.bytes_in + r.bytes_out >= f.bytes.load(std::memory_order_relaxed)) {
        // a retry worked: the failure was transient
        f.bytes.store(UINT64_MAX, std::memory_order_relaxed);
        f.retry_every.store(kRetryFirst, std::memory_order_relaxed);
    }
}

DevicePolicy& policy() {
    static DevicePolicy& p = *new DevicePolicy;
    return p;
}

} // namespace viproc::opencl
