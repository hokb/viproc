// Tests of the runtime core (no Python): ordering, parallelism, continuations,
// copy-on-write, error propagation and the look-ahead limit.

#include "runtime/runtime.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <thread>

using namespace viproc;
using namespace std::chrono_literals;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        ++g_failures;
    }
}

// Kernel from a lambda over float64 operands.
class FnKernel : public Kernel {
  public:
    using Fn = std::function<KernelResult(std::span<const Operand>, std::span<const Operand>)>;
    explicit FnKernel(Fn fn) : fn_(std::move(fn)) {}
    KernelResult run(std::span<const Operand> in, std::span<const Operand> out) override {
        return fn_(in, out);
    }

  private:
    Fn fn_;
};

RuntimeOptions opts(std::size_t workers, std::size_t max_active = kDefaultMaxActiveTasks) {
    RuntimeOptions o;
    o.workers = workers;
    o.max_active_tasks = max_active;
    return o;
}

double* f64(const Operand& op) { return reinterpret_cast<double*>(op.first()); }

Layout vec(std::int64_t n) { return Layout::contiguous(DType::Float64, {n}); }

ArrayPtr make_ready(Runtime& rt, std::int64_t n, double value) {
    Layout l = vec(n);
    auto b = std::make_shared<Buffer>(l.nbytes());
    b->ensure_allocated();
    std::fill_n(reinterpret_cast<double*>(b->data()), n, value);
    return rt.wrap(l, b);
}

// out = in0 + c (elementwise), optionally sleeping first.
std::shared_ptr<Kernel> add_const(double c, std::chrono::milliseconds delay = 0ms) {
    return std::make_shared<FnKernel>([c, delay](auto in, auto out) {
        std::this_thread::sleep_for(delay);
        const std::int64_t n = out[0].layout.size();
        for (std::int64_t i = 0; i < n; ++i) {
            f64(out[0])[i] = f64(in[0])[i] + c;
        }
        return KernelResult{};
    });
}

// out = sum of all inputs (elementwise).
std::shared_ptr<Kernel> add_all() {
    return std::make_shared<FnKernel>([](auto in, auto out) {
        const std::int64_t n = out[0].layout.size();
        for (std::int64_t i = 0; i < n; ++i) {
            double s = 0;
            for (const Operand& op : in) {
                s += f64(op)[i];
            }
            f64(out[0])[i] = s;
        }
        return KernelResult{};
    });
}

ArrayPtr issue1(Runtime& rt, std::shared_ptr<Kernel> k, std::initializer_list<ArrayPtr> in,
                std::int64_t n, const char* site = "test") {
    std::vector<ArrayPtr> ins(in);
    Layout l = vec(n);
    return rt.issue(std::move(k), ins, {}, std::span<const Layout>(&l, 1), site)[0];
}

double value(Runtime& rt, const ArrayPtr& a, std::int64_t i = 0) {
    rt.wait(*a);
    return reinterpret_cast<const double*>(a->buffer()->data() + a->layout().offset)[i];
}

void test_chain() {
    Runtime rt(opts(4));
    ArrayPtr x = make_ready(rt, 16, 0.0);
    for (int i = 0; i < 500; ++i) {
        x = issue1(rt, add_const(1.0), {x}, 16);
    }
    check(value(rt, x, 7) == 500.0, "chain of 500 dependent ops computes the right value");
}

void test_diamond() {
    Runtime rt(opts(4));
    ArrayPtr a = issue1(rt, add_const(1.0, 20ms), {make_ready(rt, 8, 1.0)}, 8); // 2
    ArrayPtr b = issue1(rt, add_const(10.0, 20ms), {a}, 8);                     // 12
    ArrayPtr c = issue1(rt, add_const(100.0), {a}, 8);                          // 102
    ArrayPtr d = issue1(rt, add_all(), {b, c}, 8);                              // 114
    check(value(rt, d, 3) == 114.0, "diamond a -> (b, c) -> d");
}

void test_main_thread_runs_ahead() {
    Runtime rt(opts(2));
    auto start = std::chrono::steady_clock::now();
    ArrayPtr x = make_ready(rt, 4, 0.0);
    for (int i = 0; i < 10; ++i) {
        x = issue1(rt, add_const(1.0, 20ms), {x}, 4);
    }
    auto issued = std::chrono::steady_clock::now() - start;
    check(issued < 50ms, "issuing 10 slow ops does not block the main thread");
    check(value(rt, x) == 10.0, "... and they all run");
}

void test_independent_ops_overlap() {
    Runtime rt(opts(4));
    auto start = std::chrono::steady_clock::now();
    std::vector<ArrayPtr> rs;
    for (int i = 0; i < 8; ++i) {
        rs.push_back(issue1(rt, add_const(i, 50ms), {make_ready(rt, 4, 0.0)}, 4));
    }
    rt.wait_all();
    auto elapsed = std::chrono::steady_clock::now() - start;
    // Serial: 400 ms. With 4 workers: ~100 ms.
    check(elapsed < 250ms, "8 independent 50 ms ops on 4 workers overlap");
}

void test_continuation_on_same_thread() {
    Runtime rt(opts(4));
    std::mutex m;
    std::vector<std::thread::id> ids;
    auto record = std::make_shared<FnKernel>([&](auto in, auto out) {
        {
            std::lock_guard lock(m);
            ids.push_back(std::this_thread::get_id());
        }
        std::this_thread::sleep_for(2ms);
        f64(out[0])[0] = f64(in[0])[0] + 1;
        return KernelResult{};
    });
    // Gate keeps the chain pending until all of it is issued.
    ArrayPtr x = issue1(rt, add_const(0.0, 30ms), {make_ready(rt, 1, 0.0)}, 1);
    for (int i = 0; i < 20; ++i) {
        x = issue1(rt, record, {x}, 1);
    }
    check(value(rt, x) == 20.0, "chain result");
    bool same = true;
    for (auto id : ids) {
        same = same && id == ids.front();
    }
    check(same, "dependent chain runs as continuations on one thread");
}

void test_deep_chain_no_stack_growth() {
    Runtime rt(opts(2));
    // Gate makes all 100000 tasks pending, so they run as continuations;
    // nested execution would overflow the stack.
    ArrayPtr x = issue1(rt, add_const(0.0, 20ms), {make_ready(rt, 1, 0.0)}, 1);
    for (int i = 0; i < 100000; ++i) {
        x = issue1(rt, add_const(1.0), {x}, 1);
    }
    check(value(rt, x) == 100000.0, "100000-long continuation chain (trampoline)");
}

void test_inout_in_place() {
    Runtime rt(opts(2));
    ArrayPtr a = make_ready(rt, 8, 1.0);
    ArrayPtr one = make_ready(rt, 8, 1.0);
    Buffer* before = a->buffer().get();
    std::vector<ArrayPtr> in{one};
    std::vector<ArrayPtr> io{a};
    rt.issue(add_all(), in, io, {}, "a += 1");
    check(value(rt, a, 5) == 2.0, "in/out a += 1 without readers");
    check(a->buffer().get() == before, "... writes in place (no copy)");
}

void test_copy_on_write() {
    Runtime rt(opts(2));
    ArrayPtr a = make_ready(rt, 8, 1.0);
    ArrayPtr one = make_ready(rt, 8, 1.0);
    // Pending reader of `a`: waits on a slow gate, so it reads `a` later.
    ArrayPtr gate = issue1(rt, add_const(0.0, 50ms), {make_ready(rt, 8, 0.0)}, 8);
    ArrayPtr reader = issue1(rt, add_all(), {gate, a}, 8);
    Buffer* before = a->buffer().get();
    check(before->async_reads.load() == 1, "a has one pending reader");
    std::vector<ArrayPtr> in{one};
    std::vector<ArrayPtr> io{a};
    rt.issue(add_all(), in, io, {}, "a += 1");
    check(a->buffer().get() != before, "in/out with pending reader switches to a new buffer");
    check(value(rt, a, 0) == 2.0, "a has the new value");
    check(value(rt, reader, 0) == 1.0, "pending reader saw the old value");
    rt.wait_all();
}

void test_error_propagation() {
    Runtime rt(opts(2));
    auto fail = std::make_shared<FnKernel>([](auto, auto) {
        std::this_thread::sleep_for(10ms);
        return KernelResult{1, "boom", 0};
    });
    ArrayPtr bad = issue1(rt, fail, {make_ready(rt, 4, 0.0)}, 4, "prog.py:10 f");
    ArrayPtr pending_child = issue1(rt, add_const(1.0), {bad}, 4, "prog.py:11 f");
    rt.wait_all();
    ArrayPtr late_child = issue1(rt, add_const(1.0), {bad}, 4, "prog.py:12 f");
    auto e1 = rt.wait(*pending_child);
    auto e2 = rt.wait(*late_child);
    check(e1 && e1->message == "boom" && e1->issue_site == "prog.py:10 f",
          "error reaches a consumer that was pending, with the original issue site");
    check(e2 && e2->issue_site == "prog.py:10 f",
          "error reaches a consumer issued after the failure");
}

void test_active_limit() {
    std::atomic<int> running{0};
    std::atomic<int> max_seen{0};
    Runtime rt(opts(4, 6));
    std::size_t max_active = 0;
    auto k = std::make_shared<FnKernel>([&](auto in, auto out) {
        int r = ++running;
        int m = max_seen.load();
        while (r > m && !max_seen.compare_exchange_weak(m, r)) {
        }
        std::this_thread::sleep_for(2ms);
        f64(out[0])[0] = f64(in[0])[0];
        --running;
        return KernelResult{};
    });
    ArrayPtr src = make_ready(rt, 1, 0.0);
    for (int i = 0; i < 100; ++i) {
        issue1(rt, k, {src}, 1);
        max_active = std::max(max_active, rt.active_tasks());
    }
    rt.wait_all();
    check(max_active <= 6, "active tasks never exceed the look-ahead limit");
    check(rt.active_tasks() == 0, "all tasks finished");
}

void test_random_programs() {
    // Random op sequences against a sequential reference evaluation.
    std::mt19937 rng(1);
    for (int round = 0; round < 20; ++round) {
        Runtime rt(opts(4));
        std::vector<ArrayPtr> arrays;
        std::vector<double> ref;
        for (int i = 0; i < 4; ++i) {
            arrays.push_back(make_ready(rt, 3, i));
            ref.push_back(i);
        }
        for (int step = 0; step < 2000; ++step) {
            const std::size_t i = rng() % arrays.size();
            const std::size_t j = rng() % arrays.size();
            switch (rng() % 3) {
            case 0: { // new array
                arrays.push_back(issue1(rt, add_all(), {arrays[i], arrays[j]}, 3));
                ref.push_back(ref[i] + ref[j]);
                break;
            }
            case 1: { // in/out: arrays[i] += arrays[j]
                std::vector<ArrayPtr> in{arrays[j]};
                std::vector<ArrayPtr> io{arrays[i]};
                rt.issue(add_all(), in, io, {}, "inout");
                ref[i] += ref[j];
                break;
            }
            default: { // new array = arrays[i] + 1
                arrays.push_back(issue1(rt, add_const(1.0), {arrays[i]}, 3));
                ref.push_back(ref[i] + 1);
                break;
            }
            }
            if (ref.size() > 64) { // keep values bounded
                for (std::size_t k = 0; k < ref.size(); ++k) {
                    ref[k] = std::fmod(ref[k], 1e6);
                }
                for (auto& a : arrays) {
                    rt.wait(*a);
                }
                for (std::size_t k = 0; k < arrays.size(); ++k) {
                    arrays[k] = make_ready(rt, 3, ref[k]);
                }
                arrays.resize(8);
                ref.resize(8);
            }
        }
        bool ok = true;
        for (std::size_t k = 0; k < arrays.size(); ++k) {
            ok = ok && value(rt, arrays[k], 2) == ref[k];
        }
        if (!ok) {
            check(false, "random program matches sequential reference");
            return;
        }
    }
    check(true, "20 random programs (2000 ops each, with in/out) match sequential reference");
}

} // namespace

int main() {
    test_chain();
    test_diamond();
    test_main_thread_runs_ahead();
    test_independent_ops_overlap();
    test_continuation_on_same_thread();
    test_deep_chain_no_stack_growth();
    test_inout_in_place();
    test_copy_on_write();
    test_error_propagation();
    test_active_limit();
    test_random_programs();
    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
