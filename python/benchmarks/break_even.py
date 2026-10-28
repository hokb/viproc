"""Break-even of the task path for small data: viproc vs. NumPy per sync_below.

Ready ops below `sync_below` elements take the direct path (no task, run on
the main thread); above it they become tasks, and a chain of dependent ops
costs one hand-off to the pool. This measures, per element count and
sync_below, the speedup over NumPy of a few patterns:

  chains_sin   4 independent chains of sin/cos ops (expensive per element)
  chains_add   4 independent chains of x * 0.5 + 1.0 (cheap per element)
  one_chain    one chain of x * 0.5 + 1.0 (nothing to overlap: pure overhead)
  read_now     (a * b).sum() read right away (nothing to overlap either)

    PYTHONPATH=build/python python3 python/benchmarks/break_even.py
"""

import argparse
import time

import numpy

import viproc


def chains_sin(np, xs, length):
    xs = list(xs)
    for _ in range(length):
        for i, x in enumerate(xs):
            xs[i] = np.sin(x) * 0.5 + np.cos(x) * 0.25
    return xs


def chains_add(np, xs, length):
    xs = list(xs)
    for _ in range(length):
        for i, x in enumerate(xs):
            xs[i] = x * 0.5 + 1.0
    return xs


def one_chain(np, xs, length):
    x = xs[0]
    for _ in range(length * 4):
        x = x * 0.5 + 1.0
    return [x]


def read_now(np, xs, length):
    a, b = xs[0], xs[1]
    s = 0.0
    for _ in range(length * 4):
        s += float((a * b).sum())
    return [s]


PATTERNS = {"chains_sin": chains_sin, "chains_add": chains_add, "one_chain": one_chain,
            "read_now": read_now}


def best(fn, repeat):
    fn()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sizes", default="100,200,300,500,1000,2000")
    p.add_argument("--thresholds", default="200,500,1000,2048")
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--repeat", type=int, default=7)
    p.add_argument("--deterministic", action="store_true", help="disable the adaptive policy")
    args = p.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    thresholds = [int(s) for s in args.thresholds.split(",")]
    rng = numpy.random.default_rng(0)
    data = {n: [rng.standard_normal(n) for _ in range(4)] for n in sizes}

    t_np = {(name, n): best(lambda: f(numpy, data[n], args.length), args.repeat)
            for name, f in PATTERNS.items() for n in sizes}
    rows = {}
    for th in thresholds:
        viproc.init(sync_below=th, adaptive=False if args.deterministic else True)
        for name, f in PATTERNS.items():
            for n in sizes:
                xs = [viproc.asarray(x) for x in data[n]]

                def run():
                    r = f(viproc, xs, args.length)
                    viproc.wait_all()
                    return r

                ref = f(numpy, data[n], args.length)
                got = run()
                assert all(numpy.array_equal(numpy.asarray(g), r) for g, r in zip(got, ref))
                rows[name, n, th] = t_np[name, n] / best(run, args.repeat)
    mode = "deterministic" if args.deterministic else "adaptive"
    print(f"speedup vs. NumPy ({mode}, length {args.length}, best of {args.repeat})")
    for name in PATTERNS:
        print(f"\n{name:10s} " + " ".join(f"{'sb=' + str(t):>8s}" for t in thresholds)
              + f" {'NumPy us/op':>12s}")
        ops = {"chains_sin": 20, "chains_add": 8, "one_chain": 8, "read_now": 8}[name]
        for n in sizes:
            print(f"{n:>10d} " + " ".join(f"{rows[name, n, t]:8.2f}" for t in thresholds)
                  + f" {t_np[name, n] / (args.length * ops) * 1e6:12.2f}")


if __name__ == "__main__":
    main()
