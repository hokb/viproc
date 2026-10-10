// End-to-end: array programs issued through the runtime and executed with
// NumPy's ufunc loops on worker threads, compared bit-for-bit with eager NumPy.

#include "kernels/numpy_ufunc.hpp"

#include "embed_python.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace viproc;
namespace vnp = viproc::numpy;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) {
        ++g_failures;
    }
}

PyObject* g_globals = nullptr;

PyObject* eval(const char* expr) {
    PyObject* r = PyRun_String(expr, Py_eval_input, g_globals, g_globals);
    if (r == nullptr) {
        PyErr_Print();
    }
    return r;
}

void set(const char* name, PyObject* value) { PyDict_SetItemString(g_globals, name, value); }

// Issues ufunc(args...) through the runtime; output dtype given explicitly.
class Program {
  public:
    explicit Program(Runtime& rt) : rt_(rt) {}

    ArrayPtr input(const char* expr) {
        PyObject* obj = eval(expr);
        keep_.push_back(obj); // alive until the program ends
        std::string err;
        ArrayPtr a = vnp::wrap_ndarray(rt_, obj, err);
        if (!a) {
            check(false, "wrap " + std::string(expr) + ": " + err);
        }
        return a;
    }

    ArrayPtr call(const char* ufunc_name, std::vector<ArrayPtr> args, DType out_dtype) {
        std::vector<DType> dtypes;
        Shape shape;
        for (const ArrayPtr& a : args) {
            dtypes.push_back(a->layout().dtype);
            broadcast_shapes(shape, a->layout().shape, shape);
        }
        dtypes.push_back(out_dtype);
        const vnp::UfuncLoop* loop = loop_for(ufunc_name, dtypes);
        if (loop == nullptr) {
            std::exit(1);
        }
        Layout out = Layout::contiguous(out_dtype, shape);
        return rt_.issue(vnp::make_elementwise_kernel(loop), args, {},
                         std::span<const Layout>(&out, 1), ufunc_name)[0];
    }

    // target = ufunc(target, other), in place (in/out semantics).
    void call_inplace(const char* ufunc_name, const ArrayPtr& target, const ArrayPtr& other) {
        DType t = target->layout().dtype;
        std::vector<DType> dtypes{t, other->layout().dtype, t};
        const vnp::UfuncLoop* loop = loop_for(ufunc_name, dtypes);
        if (loop == nullptr) {
            std::exit(1);
        }
        std::vector<ArrayPtr> in{other};
        std::vector<ArrayPtr> io{target};
        // Kernel inputs are [other, target]; the ufunc expects (target, other).
        rt_.issue(std::make_shared<Swap2>(vnp::make_elementwise_kernel(loop)), in, io, {},
                  ufunc_name);
    }

    PyObject* result(const ArrayPtr& a) {
        Py_BEGIN_ALLOW_THREADS;
        rt_.wait(*a);
        Py_END_ALLOW_THREADS;
        return vnp::to_ndarray(*a);
    }

    ~Program() {
        Py_BEGIN_ALLOW_THREADS;
        rt_.wait_all();
        Py_END_ALLOW_THREADS;
        for (PyObject* o : keep_) {
            Py_DECREF(o);
        }
    }

  private:
    struct Swap2 : Kernel {
        explicit Swap2(std::shared_ptr<Kernel> k) : k_(std::move(k)) {}
        KernelResult run(std::span<const Operand> in, std::span<const Operand> out) override {
            Operand swapped[2] = {in[1], in[0]};
            return k_->run(swapped, out);
        }
        std::shared_ptr<Kernel> k_;
    };

    const vnp::UfuncLoop* loop_for(const char* name, const std::vector<DType>& dtypes) {
        PyObject* uf = eval((std::string("np.") + name).c_str());
        std::string err;
        const vnp::UfuncLoop* loop = cache_.get(uf, dtypes, err);
        Py_DECREF(uf);
        if (loop == nullptr) {
            check(false, std::string("resolve ") + name + ": " + err);
        }
        return loop;
    }

    Runtime& rt_;
    vnp::LoopCache cache_;
    std::vector<PyObject*> keep_;
};

bool same(PyObject* got, const char* expected_expr) {
    set("_got", got);
    PyObject* eq = eval(("bool(np.array_equal(_got, " + std::string(expected_expr) +
                         ", equal_nan=True) and _got.dtype == (" + expected_expr + ").dtype)")
                            .c_str());
    bool ok = eq == Py_True;
    Py_XDECREF(eq);
    Py_XDECREF(got);
    return ok;
}

void test_broadcast_expression(Runtime& rt) {
    Program p(rt);
    ArrayPtr a = p.input("A");
    ArrayPtr b = p.input("B");
    ArrayPtr s = p.input("S");
    // r = sin(a * b + s) - a
    ArrayPtr r = p.call(
        "subtract",
        {p.call("sin",
                {p.call("add", {p.call("multiply", {a, b}, DType::Float64), s}, DType::Float64)},
                DType::Float64),
         a},
        DType::Float64);
    check(same(p.result(r), "np.sin(A * B + S) - A"),
          "sin(a*b + s) - a with (64,33)x(33,) broadcast and 0-d scalar: bit-identical");
}

void test_dtypes(Runtime& rt) {
    Program p(rt);
    struct Case {
        const char* make;
        DType dtype;
    };
    const Case cases[] = {{"np.arange(-50, 50, dtype=np.int32)", DType::Int32},
                          {"np.arange(-50, 50, dtype=np.int64) * 3**30", DType::Int64},
                          {"np.linspace(-3, 3, 100, dtype=np.float32)", DType::Float32},
                          {"np.linspace(-3, 3, 100) + 0.5j", DType::Complex128},
                          {"np.linspace(-3, 3, 100).astype(np.complex64)", DType::Complex64}};
    for (const Case& c : cases) {
        set("_x", eval(c.make));
        ArrayPtr x = p.input("_x");
        ArrayPtr sq = p.call("multiply", {x, x}, c.dtype);
        check(same(p.result(sq), "_x * _x"), std::string("multiply ") + c.make);
    }
    ArrayPtr f = p.input("np.linspace(-1, 1, 101)");
    ArrayPtr g = p.input("np.linspace(1, -1, 101)");
    ArrayPtr less = p.call("less", {f, g}, DType::Bool);
    ArrayPtr both =
        p.call("logical_and", {less, p.call("isfinite", {f}, DType::Bool)}, DType::Bool);
    check(same(p.result(both), "np.logical_and(np.linspace(-1,1,101) < np.linspace(1,-1,101), "
                               "np.isfinite(np.linspace(-1,1,101)))"),
          "less / isfinite / logical_and -> bool");
}

void test_inplace_copy_on_write(Runtime& rt) {
    Program p(rt);
    ArrayPtr a = p.input("np.arange(1000.0)");
    ArrayPtr b = p.input("np.full(1000, 2.0)");
    // Slow-ish chain gates the reader so it still pends when `a` is written.
    ArrayPtr gate = p.input("np.ones(1000)");
    for (int i = 0; i < 200; ++i) {
        gate = p.call("sqrt", {gate}, DType::Float64);
    }
    ArrayPtr reader = p.call("add", {gate, a}, DType::Float64); // reads old a
    p.call_inplace("multiply", a, b);                           // a *= 2
    p.call_inplace("add", a, b);                                // a += 2
    check(same(p.result(a), "np.arange(1000.0) * 2 + 2"), "a *= 2; a += 2 (in/out)");
    check(same(p.result(reader), "np.ones(1000) + np.arange(1000.0)"),
          "pending reader of a sees the old data (copy-on-write)");
}

void test_shared_loop_concurrently(Runtime& rt) {
    // Many independent tasks use the same resolved loop at the same time.
    Program p(rt);
    ArrayPtr x = p.input("X");
    std::vector<ArrayPtr> rs;
    for (int i = 0; i < 64; ++i) {
        rs.push_back(p.call("exp", {p.call("multiply", {x, x}, DType::Float64)}, DType::Float64));
    }
    bool ok = true;
    for (const ArrayPtr& r : rs) {
        ok = same(p.result(r), "np.exp(X * X)") && ok;
    }
    check(ok, "64 concurrent tasks sharing one loop: all bit-identical");
}

void test_long_program(Runtime& rt) {
    // Several independent chains interleaved, like a sequential program.
    Program p(rt);
    std::vector<ArrayPtr> v;
    for (int i = 0; i < 4; ++i) {
        v.push_back(p.input("Y"));
    }
    for (int step = 0; step < 100; ++step) {
        for (auto& x : v) {
            x = p.call("cos", {p.call("add", {x, x}, DType::Float64)}, DType::Float64);
        }
    }
    PyObject* expected = eval("(lambda y: [y := np.cos(y + y) for _ in range(100)][-1])(Y)");
    set("_expected", expected);
    Py_DECREF(expected);
    bool ok = true;
    for (auto& x : v) {
        ok = same(p.result(x), "_expected") && ok;
    }
    check(ok, "4 interleaved chains of 200 ops: bit-identical");
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (!init_embedded_python()) {
        return 1;
    }
    if (!vnp::init()) {
        PyErr_Print();
        return 1;
    }
    g_globals = PyDict_New();
    PyDict_SetItemString(g_globals, "__builtins__", PyEval_GetBuiltins());
    PyObject* np = PyImport_ImportModule("numpy");
    set("np", np);
    Py_DECREF(np);
    PyRun_String("rng = np.random.default_rng(7)\n"
                 "A = rng.standard_normal((64, 33))\n"
                 "B = rng.standard_normal(33)\n"
                 "S = np.array(0.25)\n"
                 "X = rng.standard_normal(10000)\n"
                 "Y = rng.standard_normal((50, 50))\n",
                 Py_file_input, g_globals, g_globals);

    {
        RuntimeOptions o;
        o.workers = 4;
        Runtime rt(o);
        test_broadcast_expression(rt);
        test_dtypes(rt);
        test_inplace_copy_on_write(rt);
        test_shared_loop_concurrently(rt);
        test_long_program(rt);
    }

    Py_DECREF(g_globals);
    if (Py_FinalizeEx() < 0) {
        return 1;
    }
    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
