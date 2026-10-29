#include "cpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <vector>
#elif defined(__linux__)
#include <sched.h>

#include <cstdio>
#include <set>
#include <string>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace viproc {

namespace {

#if defined(_WIN32)

// One RelationProcessorCore record per physical core; counts the cores
// with at least one logical processor in this process's affinity (group 0
// mask; processes on more than one group count all cores of other groups).
std::size_t os_physical_cores() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) {
        return 0;
    }
    std::vector<char> buf(len);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) {
        return 0;
    }
    DWORD_PTR process_mask = 0;
    DWORD_PTR system_mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
        process_mask = ~DWORD_PTR{0};
    }
    std::size_t cores = 0;
    for (DWORD off = 0; off < len;) {
        auto* rec = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        if (rec->Relationship == RelationProcessorCore) {
            const GROUP_AFFINITY& g = rec->Processor.GroupMask[0];
            if (g.Group != 0 || (g.Mask & process_mask) != 0) {
                ++cores;
            }
        }
        off += rec->Size;
    }
    return cores;
}

#elif defined(__linux__)

// First number of a sysfs CPU list ("0-1", "0,10"): the lowest logical CPU
// of a core identifies the core.
long first_cpu_of(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
        return -1;
    }
    long cpu = -1;
    if (std::fscanf(f, "%ld", &cpu) != 1) {
        cpu = -1;
    }
    std::fclose(f);
    return cpu;
}

std::size_t os_physical_cores() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return 0;
    }
    std::set<long> cores;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &set)) {
            continue;
        }
        const std::string dir = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        long core = first_cpu_of(dir + "core_cpus_list"); // Linux >= 5.7
        if (core < 0) {
            core = first_cpu_of(dir + "thread_siblings_list");
        }
        if (core < 0) {
            return 0; // no topology (some containers, emulators)
        }
        cores.insert(core);
    }
    return cores.size();
}

#elif defined(__APPLE__)

std::size_t os_physical_cores() {
    int n = 0;
    std::size_t size = sizeof(n);
    if (sysctlbyname("hw.physicalcpu", &n, &size, nullptr, 0) != 0 || n <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(n);
}

#else

std::size_t os_physical_cores() { return 0; }

#endif

volatile std::uint32_t probe_sink = 0;

} // namespace

std::size_t physical_core_count() {
    const std::size_t logical = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t cores = os_physical_cores();
    return cores == 0 ? logical : std::min(cores, logical);
}

std::size_t default_worker_count() {
    const std::size_t cores = physical_core_count();
    return cores > 1 ? cores - 1 : 1;
}

double issue_probe_ns() {
    // Eight interleaved walks over a random cycle of 256 nodes (4 KB, L1),
    // each step a load of the next node and of its value, a multiply-add
    // and a store: the dependent loads, refcount updates and hashing of the
    // issue path, with enough independent work that a wider core finishes
    // it sooner (bound by load/store throughput and issue width, not by
    // one load's latency). No SIMD, no data-dependent branch.
    struct Node {
        Node* next;
        std::uintptr_t value;
    };
    static constexpr std::size_t kNodes = 256;
    static constexpr int kSteps = 2048;
    Node nodes[kNodes];
    std::size_t order[kNodes];
    std::uint32_t rng = 0x9e3779b9u;
    for (std::size_t i = 0; i < kNodes; ++i) {
        order[i] = i;
    }
    for (std::size_t i = kNodes - 1; i > 0; --i) {
        rng = rng * 1664525u + 1013904223u;
        std::swap(order[i], order[(rng >> 8) % (i + 1)]);
    }
    for (std::size_t i = 0; i < kNodes; ++i) {
        nodes[order[i]] = {&nodes[order[(i + 1) % kNodes]], i};
    }
    using clock = std::chrono::steady_clock;
    double best = 1e300;
    std::uintptr_t sink = 0;
    // The best run: until 16 runs in a row found nothing 2% faster (a
    // preempted run, a core still ramping up its clock), at most 128 runs.
    for (int run = 0, stale = 0; run < 128 && stale < 16; ++run) {
        Node* p0 = &nodes[order[0]];
        Node* p1 = &nodes[order[32]];
        Node* p2 = &nodes[order[64]];
        Node* p3 = &nodes[order[96]];
        Node* p4 = &nodes[order[128]];
        Node* p5 = &nodes[order[160]];
        Node* p6 = &nodes[order[192]];
        Node* p7 = &nodes[order[224]];
        std::uintptr_t h0 = 0, h1 = 1, h2 = 2, h3 = 3, h4 = 4, h5 = 5, h6 = 6, h7 = 7;
        const auto t0 = clock::now();
        for (int k = 0; k < kSteps; ++k) {
#define VIPROC_PROBE_STEP(p, h)                                                                    \
    p = p->next;                                                                                   \
    h = h * 0x9e3779b1u + p->value;                                                                \
    ++p->value;
            VIPROC_PROBE_STEP(p0, h0)
            VIPROC_PROBE_STEP(p1, h1)
            VIPROC_PROBE_STEP(p2, h2)
            VIPROC_PROBE_STEP(p3, h3)
            VIPROC_PROBE_STEP(p4, h4)
            VIPROC_PROBE_STEP(p5, h5)
            VIPROC_PROBE_STEP(p6, h6)
            VIPROC_PROBE_STEP(p7, h7)
#undef VIPROC_PROBE_STEP
        }
        const double ns = std::chrono::duration<double, std::nano>(clock::now() - t0).count();
        stale = ns < 0.98 * best ? 0 : stale + 1;
        best = std::min(best, ns);
        sink += h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7;
    }
    probe_sink = static_cast<std::uint32_t>(sink); // keeps the walks
    return best;
}

double issue_speed_scale() {
    static const double scale = std::clamp(issue_probe_ns() / kIssueProbeReferenceNs, 0.2, 5.0);
    return scale;
}

} // namespace viproc
