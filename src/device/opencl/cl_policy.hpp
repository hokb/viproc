// Where a device op runs: on the OpenCL device or on the CPU (optimization
// level O4; docs/design/opencl.md, "Device or CPU").
//
// A device kernel asks the process-wide DevicePolicy (policy()) once per op,
// in its task, with the op's actual sizes (OpRequest). Two policies:
//
// - Threshold: the device for work >= min_work, and for smaller ops whose
//   operands' values are on the device only (a chain stays there;
//   deterministic mode, or a min_work set by the user:
//   viproc.init(device_min_work=...), env VIPROC_OPENCL_MIN_WORK).
// - Cost (default): compares the estimated time on the CPU with the
//   estimated time on the device: a fixed cost per op (the device's round
//   trip), the transfers of the operands not on the device yet and of the
//   results (bandwidth up and down), the compute time, and the wait for
//   the device ops in flight (they share the device; weighted by their
//   estimated compute times, see wait_ns()). Latency and bandwidths
//   are measured once per device (measure()); compute rates start from a
//   prior (device properties, a single CPU core) and follow the measured
//   run times per op, dtype and size class (record()), with rare
//   explorations of the side not chosen.
//
// Both remember device failures (failed()): an op of the same kind and at
// least the same size runs on the CPU right away, without another upload,
// except a retry now and then (the failure may have been transient: memory
// other ops held).
//
// Thread safety: any thread (kernels run on workers); a mutex guards the
// rate tables, taken only for ops above the cheap first test.
#pragma once

#include "device/opencl/cl_kernels.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

namespace viproc::opencl {

// Op classes with one compute rate each in the prior.
enum class OpClass { Gemm, Factor, Fft };
OpClass op_class(Op op);

// An op's request, made in its task.
struct OpRequest {
    Op op = Op::None;
    DType dtype = DType::Float64; // the device's computation dtype
    std::int64_t work = 0;        // the op's work estimate (multiply-adds, n log n, ...)
    std::uint64_t bytes_in = 0;   // operands' bytes
    std::uint64_t bytes_out = 0;  // results' bytes
    // Of bytes_in, already on the device (no upload).
    std::uint64_t bytes_resident = 0;
    // Of bytes_in, only on the device: the CPU kernel downloads them first.
    std::uint64_t bytes_stale = 0;
    // The results stay on the device (no download now; a host reader pays
    // it later, a device reader never).
    bool output_resident = false;
    // Set by the policy: the op's estimated compute time on an idle device
    // (use_device in cost mode, else begin()), and at begin() the wait for
    // the device ops in flight and their number.
    double compute_ns = 0;
    double wait_ns = 0;
    int sharing = 0;
};

// The device's costs (ns, bytes per ns, work units per ns).
struct DeviceProfile {
    double latency_ns = 20000;     // fixed cost of one op (enqueue, launch, sync)
    double upload_bpns = 4;        // host -> device, bytes per ns (4 GB/s)
    double download_bpns = 4;      // device -> host
    double device_rate[3][2] = {}; // [OpClass][double precision]: work per ns
    double cpu_rate[3][2] = {};    // the same on one CPU core (NumPy's loop)
    bool measured = false;         // latency and bandwidths measured on the device
};

// Counters of the decisions (offload_stats, device_info).
struct PolicyCounts {
    std::uint64_t device = 0, cpu = 0, explored = 0, skipped = 0;
};

class DevicePolicy {
  public:
    // The device the policy is for: priors from its properties, latency and
    // bandwidths measured on it (a few ms). Resets the rates and the failure
    // memory. Main thread (vp_opencl_select).
    void configure(Context& ctx, bool measure);
    // The user's threshold (viproc.init(device_min_work=...)): >= 0 selects
    // threshold mode, < 0 unsets it (then env VIPROC_OPENCL_MIN_WORK, if
    // set). Any thread.
    void set_min_work(std::int64_t min_work);
    // The user's threshold, or -1.
    std::int64_t min_work() const;
    // Deterministic mode (viproc.init(adaptive=False)): where an op runs
    // must not depend on timings, so without a threshold of the user's the
    // threshold is kDefaultMinWork.
    void set_deterministic(bool on) { deterministic_.store(on, std::memory_order_relaxed); }
    static constexpr std::int64_t kDefaultMinWork = std::int64_t{1} << 24;
    // The threshold in effect, or -1: the cost model decides.
    std::int64_t threshold() const;

    // True: run `r` on the device. *explore: the decision is an
    // exploration; *timed: record() the run's time (cost mode, above the
    // cheap first test).
    bool use_device(OpRequest& r, bool* explore = nullptr, bool* timed = nullptr);

    // A device run of `r` starts / ended (begin and end once each, also on
    // failure): the device's load, which later decisions wait for.
    void begin(OpRequest& r);
    void end(const OpRequest& r);
    // The wait of an op of `compute_ns` behind the device ops in flight: the
    // device shares its time between them, so each delays it by at most its
    // own remaining compute time, and by at most the op's own (fair share):
    // min(backlog, ops x compute_ns). A large matmul in flight delays a small
    // op by about the small op's own time, a small one in flight delays a
    // large op by its own time only.
    double wait_ns(double compute_ns) const;

    // Device ops leave their results on the device (default; env
    // VIPROC_OPENCL_RESIDENT=0 or set_resident(false): downloaded at once).
    bool resident() const { return resident_.load(std::memory_order_relaxed); }
    void set_resident(bool on) { resident_.store(on, std::memory_order_relaxed); }
    // A run took `ns` on the device (device = true) or on the CPU kernel.
    void record(const OpRequest& r, bool device, double ns);
    // The device failed on `r` (Error, bad_alloc).
    void failed(const OpRequest& r);
    // A device run of `r` succeeded (clears a failure the size reached).
    void succeeded(const OpRequest& r);

    // Device ops running now and their summed estimated compute times.
    int in_flight() const { return in_flight_.load(std::memory_order_relaxed); }
    double backlog_ns() const { return backlog_ns_.load(std::memory_order_relaxed); }

    DeviceProfile profile() const;
    // Tests: replace the profile (priors and measurements), clear the rates.
    void set_profile(const DeviceProfile& p);
    PolicyCounts counts() const;

    // Estimated ns of `r` on the device and on the CPU (with the current
    // rates and the device's load).
    void estimate(const OpRequest& r, double* device_ns, double* cpu_ns);

  private:
    static constexpr int kOps = 14, kDtypes = 4, kClasses = 64;
    struct Rate {
        double ns_per_work = 0; // 0: no sample
        std::uint32_t samples = 0;
    };
    struct Class {
        Rate device, cpu;
        std::uint32_t decisions = 0;
        std::uint32_t explore_every = 8; // doubles up to 1024 while explorations confirm
        bool last_device = false;
    };
    struct Failure {
        std::atomic<std::uint64_t> bytes{UINT64_MAX}; // smallest failing size
        std::atomic<std::uint32_t> skips{0};
        std::atomic<std::uint32_t> retry_every{16};
    };
    Class& cls(const OpRequest& r);
    double fixed_ns(const OpRequest& r) const;
    double cpu_fixed_ns(const OpRequest& r) const;
    double device_ns_per_work(const OpRequest& r, const Class& c) const;
    double cpu_ns_per_work(const OpRequest& r, const Class& c) const;

    mutable std::atomic<std::int64_t> min_work_{-2}; // -2: env not read yet
    std::atomic<bool> deterministic_{false};
    std::atomic<bool> resident_{true};
    mutable std::mutex mutex_;
    DeviceProfile profile_;
    // A lower bound of the device's time for any op, read without the lock.
    std::atomic<double> latency_ns_{20000};
    std::atomic<double> fastest_cpu_ns_per_work_{0};
    std::unique_ptr<Class[]> classes_ = std::make_unique<Class[]>(kOps * kDtypes * kClasses);
    Failure failures_[kOps][kDtypes];
    std::atomic<int> in_flight_{0};
    std::atomic<double> backlog_ns_{0};
    std::atomic<std::uint64_t> n_device_{0}, n_cpu_{0}, n_explored_{0}, n_skipped_{0};
};

// The process-wide policy. Never destroyed (device kernels may run until
// exit).
DevicePolicy& policy();

} // namespace viproc::opencl
