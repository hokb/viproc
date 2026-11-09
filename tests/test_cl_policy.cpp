// The device policy's cost model (O4, cl_policy.*): the device's load,
// weighted by the compute times of the ops in flight. No device needed:
// the profile is set by hand.
#include "device/opencl/cl_policy.hpp"

#include <cmath>
#include <cstdio>

using viproc::DType;
using viproc::opencl::DevicePolicy;
using viproc::opencl::DeviceProfile;
using viproc::opencl::Op;
using viproc::opencl::OpRequest;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    failures += ok ? 0 : 1;
}

bool near(double a, double b) { return std::fabs(a - b) <= 1e-6 * std::fabs(b) + 1e-9; }

OpRequest gemm(std::int64_t work) {
    OpRequest r;
    r.op = Op::Matmul;
    r.dtype = DType::Float64;
    r.work = work;
    r.output_resident = true;
    return r;
}

} // namespace

int main() {
    // Device: 1 work unit per ns (float64 gemm), the CPU half of that;
    // transfers free (all operands resident), a round trip of 1 µs.
    DeviceProfile p;
    p.latency_ns = 1000;
    p.upload_bpns = p.download_bpns = 1e9;
    for (int c = 0; c < 3; ++c) {
        p.device_rate[c][0] = p.device_rate[c][1] = 1;
        p.cpu_rate[c][0] = p.cpu_rate[c][1] = 0.5;
    }
    DevicePolicy pol;
    pol.set_profile(p);
    pol.set_min_work(-1);

    double dev = 0, cpu = 0;
    OpRequest big = gemm(1000000000);
    pol.estimate(big, &dev, &cpu);
    check(near(dev, 1000 + 1e9) && near(cpu, 2e9), "idle device: latency + compute");

    // A tiny op in flight delays a large one by its own compute time only
    // (counting ops made the large one twice as expensive: a tie with the CPU).
    OpRequest tiny = gemm(1000);
    pol.begin(tiny);
    check(near(tiny.compute_ns, 1000) && tiny.wait_ns == 0 && tiny.sharing == 0,
          "begin estimates an op the decision did not");
    check(pol.in_flight() == 1 && near(pol.backlog_ns(), 1000), "load: one op of 1 µs");
    pol.estimate(big, &dev, &cpu);
    check(near(dev, 1000 + 1e9 + 1000), "large op behind a tiny one waits 1 µs");
    check(pol.use_device(big), "large op behind a tiny one: the device");
    pol.end(tiny);
    check(pol.in_flight() == 0 && near(pol.backlog_ns(), 0), "end removes the load");

    // A large op in flight delays a small one by about its own time (fair
    // share), not by the large op's.
    OpRequest large = gemm(1000000000);
    pol.begin(large);
    OpRequest small = gemm(1000000);
    pol.estimate(small, &dev, &cpu);
    check(near(dev, 1000 + 1e6 + 1e6), "small op behind a large one waits its own time");
    check(!pol.use_device(small), "small op behind a large one: the CPU (2.001 ms vs. 2 ms)");

    // Several ops in flight: their summed compute time, at most the fair share.
    OpRequest mid = gemm(300000);
    pol.begin(mid);
    pol.estimate(small, &dev, &cpu);
    check(near(dev, 1000 + 1e6 + 2e6), "two ops in flight: a fair share each");
    pol.end(large);
    pol.estimate(small, &dev, &cpu);
    check(near(dev, 1000 + 1e6 + 3e5), "the remaining op's backlog: 0.3 ms");
    pol.end(mid);

    // A measured device run with a wait: normalized to an idle device.
    OpRequest run = gemm(1000000);
    run.compute_ns = 1e6;
    pol.begin(run); // idle: no wait
    pol.end(run);
    run.wait_ns = 1e6;
    run.sharing = 1;
    pol.record(run, true, 1000 + 2e6); // waited 1 ms behind another op
    pol.estimate(gemm(1000000), &dev, &cpu);
    check(near(dev, 1000 + 1e6), "a run's wait is not taken as its compute time");

    std::printf("%s\n", failures == 0 ? "all passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
