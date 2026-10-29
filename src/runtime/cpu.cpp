#include "cpu.hpp"

#include <algorithm>
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

} // namespace viproc
