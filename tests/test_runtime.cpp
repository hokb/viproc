// Tests of the runtime core (no Python): ordering, parallelism, continuations,
// copy-on-write, error propagation and the look-ahead limit.

#include "runtime/runtime.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
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

// Plain asynchronous dispatch (no inline execution) for the scheduling tests.
RuntimeOptions opts(std::size_t workers, std::size_t max_active = kDefaultMaxActiveTasks) {
    RuntimeOptions o;
    o.workers = workers;
    o.max_active_tasks = max_active;
    o.offload.sync_below = 0;
    o.offload.adaptive = false;
    return o;
}

double* f64(const Operand& op) { return reinterpret_cast<double*>(op.first()); }

// Element i of a 1-d (possibly strided) float64 operand.
double& at(const Operand& op, std::int64_t i) {
    return *reinterpret_cast<double*>(op.first() + i * op.layout.strides[0]);
}

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
            at(out[0], i) = at(in[0], i) + c;
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
                s += at(op, i);
            }
            at(out[0], i) = s;
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
    const Layout& l = a->layout();
    const std::int64_t stride = l.strides.empty() ? 0 : l.strides[0];
    return *reinterpret_cast<const double*>(a->buffer()->data() + l.offset + i * stride);
}

// View of a 1-d float64 array: elements start, start+step, ... (n of them).
ArrayPtr slice(Runtime& rt, const ArrayPtr& base, std::int64_t start, std::int64_t step,
               std::int64_t n) {
    Layout l = base->layout();
    l.offset += start * l.strides[0];
    l.strides[0] *= step;
    l.shape[0] = n;
    return rt.view(base, l);
}

void inplace_add(Runtime& rt, const ArrayPtr& target, const ArrayPtr& other) {
    std::vector<ArrayPtr> in{other};
    std::vector<ArrayPtr> io{target};
    rt.issue(add_all(), in, io, {}, "target += other");
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

void test_view_write_visible_in_base() {
    Runtime rt(opts(2));
    ArrayPtr base = make_ready(rt, 8, 0.0);
    ArrayPtr even = slice(rt, base, 0, 2, 4); // base[0::2]
    Buffer* before = base->buffer().get();
    inplace_add(rt, even, make_ready(rt, 4, 1.0)); // base[0::2] += 1
    check(base->buffer().get() == before, "view write without readers is in place");
    check(!base->ready() || value(rt, base, 0) == 1.0,
          "base is pending until the view write completes");
    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        ok = ok && value(rt, base, i) == (i % 2 == 0 ? 1.0 : 0.0);
    }
    check(ok, "write through view base[0::2] is visible in base");
}

void test_view_write_waits_for_base_producer() {
    Runtime rt(opts(2));
    // base is pending (slow producer); a view of it is written in place.
    ArrayPtr base = issue1(rt, add_const(5.0, 30ms), {make_ready(rt, 8, 0.0)}, 8);
    ArrayPtr odd = slice(rt, base, 1, 2, 4);
    inplace_add(rt, odd, make_ready(rt, 4, 1.0)); // base[1::2] += 1
    check(value(rt, base, 0) == 5.0 && value(rt, base, 1) == 6.0,
          "view write orders after the pending producer of its base");
}

void test_view_copy_on_write() {
    Runtime rt(opts(2));
    ArrayPtr base = make_ready(rt, 8, 0.0);
    ArrayPtr even = slice(rt, base, 0, 2, 4);
    // Pending reader of the whole base.
    ArrayPtr gate = issue1(rt, add_const(0.0, 50ms), {make_ready(rt, 8, 0.0)}, 8);
    ArrayPtr reader = issue1(rt, add_all(), {gate, base}, 8);
    Buffer* before = base->buffer().get();
    inplace_add(rt, even, make_ready(rt, 4, 1.0)); // base[0::2] += 1
    check(base->buffer().get() != before && base->buffer() == even->buffer(),
          "view write with pending reader: base and view switch to a new buffer together");
    check(even->layout().strides[0] == 16, "view keeps its layout (no consolidation with views)");
    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        ok = ok && value(rt, base, i) == (i % 2 == 0 ? 1.0 : 0.0) && value(rt, reader, i) == 0.0;
    }
    check(ok, "base sees the view write, the pending reader sees the old data");
}

void test_consolidated_copy_on_write_of_strided_array() {
    Runtime rt(opts(2));
    // A reversed array (stride -8) that alone covers its whole storage.
    Layout rev = vec(4);
    rev.strides[0] = -8;
    rev.offset = 24;
    auto b = std::make_shared<Buffer>(rev.nbytes());
    b->ensure_allocated();
    for (int i = 0; i < 4; ++i) {
        reinterpret_cast<double*>(b->data())[i] = i; // logical a = [3, 2, 1, 0]
    }
    ArrayPtr a = rt.wrap(rev, b);
    ArrayPtr gate = issue1(rt, add_const(0.0, 30ms), {make_ready(rt, 4, 0.0)}, 4);
    ArrayPtr reader = issue1(rt, add_all(), {gate, a}, 4);
    inplace_add(rt, a, make_ready(rt, 4, 10.0));
    check(a->layout().strides[0] == 8 && a->layout().offset == 0 && a->buffer().get() != b.get(),
          "consolidated copy-on-write: new contiguous buffer, decided at issue time");
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        ok = ok && value(rt, a, i) == 13.0 - i && value(rt, reader, i) == 3.0 - i;
    }
    check(ok, "consolidated copy-on-write: logical values kept, reader sees old data");
}

void test_overlapping_inplace() {
    Runtime rt(opts(2));
    // a += a[::-1] must read all of `a` before writing (NumPy semantics).
    Layout l = vec(6);
    auto b = std::make_shared<Buffer>(l.nbytes());
    b->ensure_allocated();
    for (int i = 0; i < 6; ++i) {
        reinterpret_cast<double*>(b->data())[i] = i;
    }
    ArrayPtr a = rt.wrap(l, b);
    ArrayPtr rev = slice(rt, a, 5, -1, 6);
    inplace_add(rt, a, rev);
    bool ok = true;
    for (int i = 0; i < 6; ++i) {
        ok = ok && value(rt, a, i) == 5.0; // i + (5 - i)
    }
    check(ok, "a += a[::-1] reads the old data (overlap forces copy-on-write)");
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

void test_random_view_programs() {
    // Random in-place writes through views (incl. overlapping and reversed
    // ones) and new arrays, against a sequential reference with NumPy
    // semantics (all inputs are read before the output is written).
    std::mt19937 rng(2);
    constexpr std::int64_t n = 6;
    struct Slice {
        std::int64_t start, step, len;
    };
    const Slice slices[] = {{0, 1, 6}, {0, 2, 3}, {1, 2, 3}, {5, -1, 6}, {4, -2, 3}, {2, 1, 3}};
    for (int round = 0; round < 20; ++round) {
        Runtime rt(opts(4));
        std::vector<ArrayPtr> bases;
        std::vector<std::vector<double>> ref;
        for (int i = 0; i < 4; ++i) {
            bases.push_back(make_ready(rt, n, i));
            ref.push_back(std::vector<double>(n, i));
        }
        for (int step = 0; step < 1500; ++step) {
            const std::size_t i = rng() % bases.size();
            const std::size_t j = rng() % bases.size();
            if (rng() % 4 == 0) { // new base = bases[i] + bases[j]
                bases.push_back(issue1(rt, add_all(), {bases[i], bases[j]}, n));
                std::vector<double> r(n);
                for (std::int64_t k = 0; k < n; ++k) {
                    r[k] = ref[i][k] + ref[j][k];
                }
                ref.push_back(r);
                continue;
            }
            // bases[i][si] += bases[j][sj] with matching lengths.
            const Slice& si = slices[rng() % std::size(slices)];
            Slice sj = slices[rng() % std::size(slices)];
            while (sj.len != si.len) {
                sj = slices[rng() % std::size(slices)];
            }
            inplace_add(rt, slice(rt, bases[i], si.start, si.step, si.len),
                        slice(rt, bases[j], sj.start, sj.step, sj.len));
            std::vector<double> src(si.len);
            for (std::int64_t k = 0; k < si.len; ++k) {
                src[k] = ref[j][sj.start + k * sj.step];
            }
            for (std::int64_t k = 0; k < si.len; ++k) {
                ref[i][si.start + k * si.step] += src[k];
            }
            if (bases.size() > 12) {
                bases.resize(4);
                ref.resize(4);
            }
        }
        bool ok = true;
        for (std::size_t b = 0; b < bases.size(); ++b) {
            for (std::int64_t k = 0; k < n; ++k) {
                ok = ok && value(rt, bases[b], k) == ref[b][k];
            }
        }
        if (!ok) {
            check(false, "random view program matches sequential reference");
            return;
        }
    }
    check(true, "20 random view programs (1500 ops each, overlapping views) match reference");
}

std::atomic<bool> on_main_thread_last{false};

// out = in0 + 1, recording whether it ran on the calling (main) thread.
std::shared_ptr<Kernel> add_one_where(std::thread::id main) {
    return std::make_shared<FnKernel>([main](auto in, auto out) {
        on_main_thread_last = std::this_thread::get_id() == main;
        const std::int64_t n = out[0].layout.size();
        for (std::int64_t i = 0; i < n; ++i) {
            at(out[0], i) = at(in[0], i) + 1;
        }
        return KernelResult{};
    });
}

void test_offload_small_ops_run_inline() {
    RuntimeOptions o = opts(2);
    o.offload.sync_below = 1024;
    Runtime rt(o);
    const auto main = std::this_thread::get_id();
    ArrayPtr small = issue1(rt, add_one_where(main), {make_ready(rt, 100, 1.0)}, 100);
    check(small->ready() && on_main_thread_last,
          "op below sync_below with ready inputs runs inline");
    ArrayPtr big = issue1(rt, add_one_where(main), {make_ready(rt, 4096, 1.0)}, 4096);
    rt.wait(*big);
    check(!on_main_thread_last, "op above sync_below goes to a worker");
    // Pending input: always asynchronous, whatever the size.
    ArrayPtr gate = issue1(rt, add_const(0.0, 20ms), {make_ready(rt, 4096, 0.0)}, 4096);
    ArrayPtr dep = issue1(rt, add_one_where(main), {gate}, 4096);
    check(!dep->ready(), "op with a pending input is issued asynchronously");
    rt.wait(*dep);
    check(rt.offload_policy().inline_count() == 1, "only the small ready op ran inline");
}

void test_offload_adaptive_per_site() {
    RuntimeOptions o = opts(2);
    o.offload.sync_below = 0;
    o.offload.adaptive = true;
    Runtime rt(o);
    const auto main = std::this_thread::get_id();
    ArrayPtr x = make_ready(rt, 2048, 1.0);
    // Site A: issue, then immediately wait (nothing to overlap).
    int inline_runs = 0;
    for (int i = 0; i < 64; ++i) {
        ArrayPtr r = issue1(rt, add_one_where(main), {x}, 2048, "prog.py:1 a");
        rt.wait(*r);
        inline_runs += on_main_thread_last ? 1 : 0;
    }
    check(inline_runs > 40,
          "a site whose result is always waited on right away switches to inline");
    // Site B: results are not waited on right away.
    const std::uint64_t before = rt.offload_policy().inline_count();
    std::vector<ArrayPtr> keep;
    for (int i = 0; i < 64; ++i) {
        keep.push_back(issue1(rt, add_const(1.0), {x}, 2048, "prog.py:2 b"));
        for (int j = 0; j < 10; ++j) { // other work in between
            keep.push_back(issue1(rt, add_const(1.0), {x}, 2048, "prog.py:3 c"));
        }
    }
    rt.wait_all();
    check(rt.offload_policy().inline_count() == before,
          "sites whose results are not waited on right away stay asynchronous");
    // Site A again, but now with overlap: it switches back (exploration).
    for (int i = 0; i < 400; ++i) {
        keep.push_back(issue1(rt, add_one_where(main), {x}, 2048, "prog.py:1 a"));
        for (int j = 0; j < 10; ++j) {
            keep.push_back(issue1(rt, add_const(1.0), {x}, 2048, "prog.py:3 c"));
        }
        if (keep.size() > 200) {
            rt.wait_all();
            keep.clear();
        }
    }
    rt.wait_all();
    int async_now = 0;
    for (int i = 0; i < 20; ++i) {
        ArrayPtr r = issue1(rt, add_one_where(main), {x}, 2048, "prog.py:1 a");
        for (int j = 0; j < 10; ++j) {
            keep.push_back(issue1(rt, add_const(1.0), {x}, 2048, "prog.py:3 c"));
        }
        rt.wait(*r);
        async_now += on_main_thread_last ? 0 : 1;
    }
    rt.wait_all();
    check(async_now > 10, "a site switched to inline goes back to async once its results overlap");
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
    test_view_write_visible_in_base();
    test_view_write_waits_for_base_producer();
    test_view_copy_on_write();
    test_consolidated_copy_on_write_of_strided_array();
    test_overlapping_inplace();
    test_error_propagation();
    test_active_limit();
    test_random_programs();
    test_offload_small_ops_run_inline();
    test_offload_adaptive_per_site();
    test_random_view_programs();
    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
