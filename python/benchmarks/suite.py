"""viproc benchmark suite: many array programs, NumPy vs. viproc, one score.

Runs every case of suite_cases.py in several sizes and layouts, plain NumPy
against viproc with 1 worker and with one worker per core, checks that the
results are identical, and condenses the speedups into the *parallel
efficiency score* (0..100, 100 = ideal), normalized to the machine:

- **Cores.** A case of parallel width P (independent instruction streams)
  ideally runs min(cores, P) times faster than NumPy on medium and large
  data. Tiny data cannot amortize a hand-off, so there the ideal is parity
  with NumPy (1x). Efficiency = speedup / ideal; this divides out the core
  count, so the score is comparable between a laptop and a server.
- **RAM.** The `large` size class is scaled to the memory per core
  (RAM / (512 x max(cores, 8)) elements, clamped to [500k, 10M]): large
  for the machine, never swapping.

Score = 100 x geometric mean over categories of the geometric mean of the
efficiencies in that category (each category weighs the same).

Usage:
    PYTHONPATH=build/python python3 python/benchmarks/suite.py [--quick] [--out DIR]
    python3 python/benchmarks/suite.py --merge a/results.json b/results.json --out DIR
    docker run --rm -v "$PWD:/work" viproc:linux benchmark suite

Writes DIR/results.json, DIR/report.md and DIR/report.html.
"""

import argparse
import ctypes
import datetime
import json
import math
import os
import platform
import re
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import suite_cases  # noqa: E402
import suite_report  # noqa: E402


# --- machine ------------------------------------------------------------------


def usable_cores():
    """CPUs this process may use: affinity and a cgroup CPU quota count."""
    if hasattr(os, "process_cpu_count"):
        n = os.process_cpu_count() or 1
    elif hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    else:
        n = os.cpu_count() or 1
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


def physical_memory():
    """Bytes of RAM this process may use (a cgroup limit counts)."""
    total = 0
    if sys.platform == "win32":

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = MemoryStatusEx()
        m.dwLength = ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            total = m.ullTotalPhys
    else:
        try:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            total = 0
        try:
            limit = open("/sys/fs/cgroup/memory.max").read().strip()
            if limit != "max":
                total = min(total, int(limit)) if total else int(limit)
        except (OSError, ValueError):
            pass
    return total or 8 << 30


def cpu_model():
    if sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def machine_info(cores, ram):
    import viproc

    return {
        "label": os.environ.get("VIPROC_PLATFORM") or f"{platform.system().lower()}",
        "os": platform.platform(),
        "cpu": cpu_model(),
        "cores": cores,
        "logical_cpus": os.cpu_count(),
        "ram_bytes": ram,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "viproc": viproc.__version__,
        "build_type": os.environ.get("VIPROC_BUILD_TYPE", ""),
        "elision": viproc._viproc.elision_mode(),
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "default"),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def size_elements(size, cores, ram):
    if size == "large":
        n = ram // (512 * max(cores, 8))
        return int(min(10_000_000, max(500_000, n)))
    return suite_cases.SIZE_ELEMENTS[size]


# --- data ---------------------------------------------------------------------


def tree_map(x, fn):
    if isinstance(x, list):
        return [tree_map(i, fn) for i in x]
    if isinstance(x, tuple):
        return tuple(tree_map(i, fn) for i in x)
    return fn(x)


def numpy_inputs(case, size, layout, dtype, cores, ram, seed):
    """Base arrays from case.make, then the layout's view of each array of
    the layout's base shape."""
    lay = suite_cases.Layout(layout)
    base_shape = lay.base_shape(size_elements(size, cores, ram))
    width = cores if case.width == "cores" else case.width
    rng = np.random.default_rng(seed)
    extra = [dtype] if dtype else []  # typed variants (category "dtypes")
    bases = tree_map(case.make(rng, base_shape, width, *extra),
                     lambda a: np.asarray(a) if np.isscalar(a) else a)
    return bases, lay, base_shape


def default_dtype(bases):
    """dtype name of an untyped case's inputs, index arrays aside: the one
    float/complex dtype, or 'mixed'."""
    kinds = set()
    tree_map(bases, lambda a: kinds.add(a.dtype))
    data = {d for d in kinds if d.kind not in "iub"} or kinds
    return str(data.pop()) if len(data) == 1 else "mixed"


def apply_layout(bases, lay, base_shape, convert):
    def one(a):
        a = convert(a)
        return lay.view(a) if tuple(a.shape) == tuple(base_shape) else a

    return tree_map(bases, one)


def materialize(result):
    """NumPy copy of a result (sync point for viproc results)."""
    def one(x):
        if hasattr(x, "shape"):
            return np.array(np.asarray(x))
        return np.asarray(x)

    return tree_map(result, one)


def identical(a, b):
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))


# --- measurement ----------------------------------------------------------------


def best_of(fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), times


def repeats_for(first_run, repeat):
    """Fewer timed runs for slow measurements (large data, or much slower than
    NumPy): their noise is relatively small, their cost is not."""
    return repeat if first_run < 0.3 else min(repeat, 2)


def calibrate_reps(case, data, target):
    """reps so that one NumPy run takes about `target` seconds."""
    reps = 1
    while True:
        t0 = time.perf_counter()
        case.run(data, reps)
        t = time.perf_counter() - t0
        if t >= target / 4 or reps >= 1 << 16:
            break
        reps *= 4 if t < target / 32 else 2
    return max(1, min(1 << 16, int(round(reps * target / max(t, 1e-9)))))


def ideal_speedup(size, width, cores):
    if size in ("scalar", "tiny", "small"):
        return 1.0
    return float(min(cores, width))


def geomean(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def score(results, cores):
    """Per-category and overall efficiency scores of finished results."""
    by_cat = {}
    for r in results:
        if r.get("efficiency") is not None:
            by_cat.setdefault(r["category"], []).append(r["efficiency"])
    cats = {c: geomean(v) for c, v in by_cat.items()}
    overall = 100.0 * geomean(list(cats.values()))
    by_size = {}
    for r in results:
        if r.get("efficiency") is not None:
            by_size.setdefault(r["size"], []).append(r["efficiency"])
    return {
        "score": overall,
        "categories": {c: 100.0 * v for c, v in cats.items()},
        "sizes": {s: 100.0 * geomean(v) for s, v in by_size.items()},
        "speedup_geomean": geomean([r["speedup"] for r in results if r.get("speedup")]),
        "speedup_w1_geomean": geomean([r["speedup_w1"] for r in results
                                       if r.get("speedup_w1")]),
        "identical": sum(1 for r in results if r.get("identical")),
        "configs": len(results),
    }


def measure_viproc(case, bases, lay, base_shape, reps, repeat, expected, workers):
    """Best time of `case` on viproc with `workers` threads, whether its result
    equals NumPy's, and the eager fallbacks of one run. Every viproc array
    is gone when this returns (before the next init)."""
    import viproc

    viproc.init(workers=workers)
    unsupported = []

    def convert(a):
        # A dtype viproc does not support (int16, float16) stays a NumPy
        # array, as in a viproc program: those ops run in plain NumPy.
        try:
            return viproc.asarray(a)
        except TypeError:
            unsupported.append(str(a.dtype))
            return a

    vdata = apply_layout(bases, lay, base_shape, convert)
    viproc.wait_all()

    def run():
        out = case.run(vdata, reps)
        viproc.wait_all()
        return out

    f0 = viproc._ndarray.fallback_calls
    t0 = time.perf_counter()
    same = identical(materialize(run()), expected)  # also the warm-up
    warm = time.perf_counter() - t0
    fallbacks = viproc._ndarray.fallback_calls - f0
    t, _ = best_of(run, repeats_for(warm, repeat))
    return t, same, fallbacks, sorted(set(unsupported))


def run_suite(args):
    import viproc

    cores = args.cores or usable_cores()
    ram = physical_memory()
    workers = sorted({int(w) for w in args.workers.replace("max", str(cores)).split(",")})
    wmax = max(workers)
    target = 0.02 if args.quick else 0.04
    repeat = args.repeat or (3 if args.quick else 5)
    info = machine_info(cores, ram)
    info.update({"workers": workers, "repeat": repeat, "target_s": target,
                 "large_elements": size_elements("large", cores, ram)})

    configs = []
    for case in suite_cases.CASES:
        for variant in case.variants:
            size, layout = variant[:2]
            dtype = variant[2] if len(variant) > 2 else None
            if args.quick and size == "large" and (layout != "vec" or dtype not in
                                                   (None, "float64", "float32")):
                continue
            name = f"{case.name}/{size}/{layout}/{dtype or 'float64'}"
            if args.filter and not re.search(args.filter, name):
                continue
            configs.append((case, size, layout, dtype))

    print(f"viproc benchmark suite: {len(configs)} configurations, {cores} cores, "
          f"{ram / 2**30:.1f} GiB RAM, workers {workers}, large = "
          f"{info['large_elements']:,} elements", flush=True)

    results = []
    for i, (case, size, layout, dtype) in enumerate(configs):
        width = cores if case.width == "cores" else case.width
        bases, lay, base_shape = numpy_inputs(case, size, layout, dtype, cores, ram, seed=i)
        data = apply_layout(bases, lay, base_shape, lambda a: a)
        reps = calibrate_reps(case, data, target)
        t0 = time.perf_counter()
        expected = materialize(case.run(data, reps))
        warm = time.perf_counter() - t0
        t_np, _ = best_of(lambda: case.run(data, reps), repeats_for(warm, repeat))
        r = {"case": case.name, "category": case.category, "description": case.description,
             "size": size, "layout": layout, "dtype": dtype or default_dtype(bases),
             "width": width, "reps": reps,
             "elements": int(np.prod(base_shape)) if base_shape else 1,
             "numpy_s": t_np, "viproc_s": {}, "fallbacks": None, "identical": True}
        for w in workers:
            t, same, fallbacks, unsupported = measure_viproc(
                case, bases, lay, base_shape, reps, repeat, expected, w)
            r["viproc_s"][str(w)] = t
            r["identical"] = r["identical"] and same
            if w == wmax:
                r["fallbacks"] = fallbacks
                r["not_in_viproc"] = unsupported
        viproc.shutdown()
        r["speedup"] = t_np / r["viproc_s"][str(wmax)]
        r["speedup_w1"] = t_np / r["viproc_s"]["1"] if "1" in r["viproc_s"] else None
        r["ideal"] = ideal_speedup(size, width, wmax)
        r["efficiency"] = r["speedup"] / r["ideal"]
        results.append(r)
        mark = "" if r["identical"] else "  RESULT DIFFERS"
        print(f"[{i + 1:3}/{len(configs)}] {case.name:18} {size:6} {layout:7} "
              f"{r['dtype']:18} P={width:<3} "
              f"numpy {t_np * 1e3:8.2f} ms  viproc {r['viproc_s'][str(wmax)] * 1e3:8.2f} ms  "
              f"speedup {r['speedup']:5.2f}x  eff {r['efficiency']:4.2f}  "
              f"fallbacks {r['fallbacks']}"
              + (f"  stays NumPy: {','.join(r['not_in_viproc'])}" if r["not_in_viproc"] else "")
              + mark, flush=True)

    return {"machine": info, "results": results, "summary": score(results, wmax)}


def main():
    # Some programs overflow or divide by zero on purpose (float16, int
    # wrap-around); NumPy and viproc then agree on inf/nan, the warnings are
    # noise here.
    warnings.simplefilter("ignore", RuntimeWarning)
    np.seterr(all="ignore")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="reports", help="output directory (default: ./reports)")
    p.add_argument("--quick", action="store_true", help="shorter runs, fewer large cases")
    p.add_argument("--filter", help="regular expression on case/size/layout")
    p.add_argument("--workers", default="1,max",
                   help="viproc worker counts, 'max' = cores (default 1,max)")
    p.add_argument("--cores", type=int, help="override the detected core count")
    p.add_argument("--repeat", type=int, help="timed runs per measurement (best of)")
    p.add_argument("--merge", nargs="+", metavar="RESULTS_JSON",
                   help="combine results of several machines/platforms into one report")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.merge:
        runs = [json.load(open(f)) for f in args.merge]
        suite_report.write_comparison(runs, args.out)
        print(f"comparison of {len(runs)} runs written to {args.out}/report.md and report.html")
        return

    run = run_suite(args)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(run, f, indent=1)
    suite_report.write_reports(run, args.out)
    s = run["summary"]
    print()
    print(suite_report.summary_markdown(run))
    print(f"report: {os.path.join(args.out, 'report.html')} (also report.md, results.json)")
    if s["identical"] != s["configs"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
