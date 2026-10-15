"""ILNumerics Accelerator as a reference for the viproc shape map.

Runs the programs of shape_map_cases.py, written in C# for ILNumerics
(python/benchmarks/ilnumerics, built in Release: the ILNumerics Accelerator
compiles it), at exactly the grid points and step counts (reps) of a viproc
shape map run on the same machine, and reports

- **ILNumerics vs. NumPy**: the same map as shape_map.py (median and
  interquartile range over the cases of NumPy time / ILNumerics time), with
  the NumPy times of the viproc run, and
- **viproc vs. ILNumerics**: the difference map, per case ILNumerics time /
  viproc time (> 1: viproc faster), condensed the same way.

Inputs are generated, not drawn from NumPy's random generator, so that both
sides see the same values (element i of input k: 0.5 + u, u in [0, 1) from
splitmix64 of the point's seed, k and i). Every point is checked against
NumPy: the C# program runs each program for `--check-reps` steps and reports
shape, sum and sampled values of its outputs, which must match NumPy on the
same inputs within a relative tolerance (ILNumerics computes with its own
kernels, so results are not bit-identical).

Usage (one machine, all three steps):
    PYTHONPATH=build/python python3 python/benchmarks/shape_map.py --out DIR
    python3 python/benchmarks/shape_map_ilnumerics.py --viproc DIR/shape_map.json --out DIR
    python3 python/benchmarks/shape_map_ilnumerics.py --viproc DIR/shape_map.json \\
        --iln DIR/ilnumerics_raw.json --out DIR      # only redo the reports

Writes DIR/ilnumerics_plan.json (the C# program's input), DIR/ilnumerics_raw.json
(its output), DIR/shape_map_ilnumerics.{json,md,png} and _groups.png
(ILNumerics vs. NumPy), DIR/shape_map_vs_ilnumerics.{json,md,png} and
_groups.png (viproc vs. ILNumerics).

The ILNumerics timings need a licensed build: an unlicensed build runs, but
throttled. Build it once on a licensed seat; the build writes
python/benchmarks/ilnumerics/ILNumerics_deploy/net8.0/ilnumerics.lic, which
lets other machines rebuild the unchanged sources (README.md there).
"""

import argparse
import copy
import json
import os
import queue
import subprocess
import sys
import threading
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import shape_map  # noqa: E402
import shape_map_cases  # noqa: E402

PROJECT = os.path.join(HERE, "ilnumerics")
# The committed runtime license (ilnumerics/ILNumerics_deploy) was made for
# Release|x64 of exactly these sources: build that configuration, and any
# change to the C# sources needs a new license from a licensed seat.
PLATFORM = "x64"
DLL = os.path.join(PROJECT, "bin", PLATFORM, "Release", "net8.0", "ShapeMapILNumerics.dll")
ILN = "ILNumerics Accelerator"


# --- inputs shared with the C# program ------------------------------------------------


def splitmix_uniform(key):
    """u in [0, 1) per uint64 key (Program.cs, Uniform)."""
    with np.errstate(over="ignore"):
        z = key + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def generated(seed, k, shape):
    n = int(np.prod(shape)) if shape else 1
    key = np.uint64((seed << 40) | (k << 32)) | np.arange(n, dtype=np.uint64)
    return splitmix_uniform(key).reshape(shape)


class GeneratedRng:
    """Stands in for the NumPy generator a case's `make` gets: input k of the
    point is `generated(seed, k, shape)`, scaled like rng.uniform. Only
    uniform(0.5, 1.5) inputs exist (shape_map_cases.uniform), so C# sees
    0.5 + u as well."""

    def __init__(self, seed):
        self.seed, self.k = seed, 0

    def uniform(self, lo, hi, size):
        assert (lo, hi) == (0.5, 1.5), "Program.cs generates 0.5 + u only"
        u = generated(self.seed, self.k, tuple(size))
        self.k += 1
        return lo + (hi - lo) * u


def case_inputs(case, shape, width, seed):
    data = case.make(GeneratedRng(seed), shape, width)
    return data, [[list(a.shape), a.dtype.name] for a in data]


# --- plan, run, check -----------------------------------------------------------------


def make_plan(vrun, args):
    cases = {c.name: c for c in shape_map_cases.CASES}
    points = []
    for i, r in enumerate(vrun["results"]):
        case = cases[r["case"]]
        _, inputs = case_inputs(case, tuple(r["shape"]), r["width"], i)
        points.append({"case": r["case"], "shape": r["shape"], "width": r["width"],
                       "reps": r["reps"], "seed": i, "inputs": inputs})
    return {"repeat": args.repeat or vrun["machine"].get("repeat", 3),
            "check_reps": args.check_reps, "samples": 16, "points": points}


def run_csharp(plan, plan_path, raw_path, threads, timeout):
    """Runs the plan. The program prints one progress line per point and
    saves its results every 50 points; a point without a progress line after
    `timeout` seconds is recorded as an error, the program is stopped and
    restarted with the points after it (and those it had not saved yet), so
    a hung point does not stop the map."""
    env = dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_NOLOGO="1")
    print("building", PROJECT, "(Release|x64: ILNumerics Accelerator)", flush=True)
    subprocess.run(["dotnet", "build", "-c", "Release", f"-p:Platform={PLATFORM}", "-nologo",
                    "-v", "q"], cwd=PROJECT, env=env, check=True)
    points = plan["points"]
    results, machine, todo, part = [None] * len(points), None, list(range(len(points))), 0
    while todo:
        sub_plan, out = f"{plan_path}.part{part}", f"{raw_path}.part{part}"
        with open(sub_plan, "w", encoding="utf-8") as f:
            json.dump(dict(plan, points=[points[i] for i in todo]), f)
        cmd = ["dotnet", DLL, sub_plan, out] + (["--threads", str(threads)] if threads else [])
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        lines = queue.Queue()
        threading.Thread(target=lambda: [lines.put(x) for x in proc.stdout], daemon=True).start()
        done, last, hung = 0, time.monotonic(), False
        while proc.poll() is None or not lines.empty():
            try:
                line = lines.get(timeout=0.5)
                print(line, end="", flush=True)
                if line.startswith("["):
                    done, last = done + 1, time.monotonic()
            except queue.Empty:
                if proc.poll() is None and time.monotonic() - last > timeout:
                    proc.kill()
                    proc.wait()
                    hung = True
        saved = []
        if os.path.exists(out):
            raw = json.load(open(out, encoding="utf-8"))
            machine, saved = machine or raw["machine"], raw["results"]
            os.remove(out)
        os.remove(sub_plan)
        for i, r in zip(todo, saved):
            results[i] = r
        if len(saved) == len(todo):
            break
        bad = todo[min(done, len(todo) - 1)]
        p = points[bad]
        reason = (f"timeout: no result after {timeout:g} s" if hung
                  else f"program exited with code {proc.returncode}")
        print(f"{p['case']} {p['shape']}: {reason}; continuing after it", flush=True)
        results[bad] = {"case": p["case"], "shape": p["shape"], "reps": p["reps"],
                        "error": reason}
        todo = [i for i in todo[len(saved):] if i != bad]
        part += 1
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({"machine": machine, "results": results}, f, indent=1)


def as2d(shape):
    """ILNumerics shapes are 2-d at least: () -> (1, 1)."""
    shape = tuple(shape)
    return shape if len(shape) >= 2 else (1,) * (2 - len(shape)) + shape


def check(case, point, res, check_reps):
    """'' if the ILNumerics outputs of a point match NumPy, else the reason."""
    if "error" in res:
        return res["error"]
    data, _ = case_inputs(case, tuple(point["shape"]), point["width"], point["seed"])
    expected = case.run(data, check_reps)
    expected = [np.asarray(e) for e in (expected if isinstance(expected, list) else [expected])]
    outs = res["check"]["outputs"]
    if len(outs) != len(expected):
        return f"{len(outs)} outputs, NumPy {len(expected)}"
    for j, (o, e) in enumerate(zip(outs, expected)):
        if tuple(o["shape"]) != as2d(e.shape):
            return f"output {j}: shape {o['shape']}, NumPy {list(e.shape)}"
        f32 = e.dtype == np.float32
        rtol = 2e-5 if f32 else 1e-9
        flat = e.astype(np.float64).ravel()
        got = np.asarray(o["values"])
        if not np.allclose(got, flat[o["index"]], rtol=rtol, atol=rtol):
            bad = int(np.argmax(np.abs(got - flat[o["index"]])))
            return (f"output {j}: element {o['index'][bad]} is {got[bad]!r}, "
                    f"NumPy {flat[o['index'][bad]]!r}")
        if not np.isclose(o["sum"], flat.sum(), rtol=1e-4 if f32 else 1e-8):
            return f"output {j}: sum {o['sum']!r}, NumPy {flat.sum()!r}"
    return ""


# --- reports ------------------------------------------------------------------------


def derived_run(vrun, raw, results, labels):
    run = {"machine": copy.deepcopy(vrun["machine"]), "results": results,
           "skipped": vrun.get("skipped", []), "labels": labels}
    run["machine"]["ilnumerics"] = raw["machine"]
    run["points"] = shape_map.summarize(results)
    return run


def build_runs(vrun, raw, plan, check_reps):
    cases = {c.name: c for c in shape_map_cases.CASES}
    iln_results, diff_results, failed = [], [], []
    for v, point, res in zip(vrun["results"], plan["points"], raw["results"]):
        assert (v["case"], v["shape"]) == (res["case"], res["shape"]), "plan and results differ"
        reason = check(cases[v["case"]], point, res, check_reps)
        if reason:
            failed.append(f"{v['case']} {v['shape_key']}: {reason}")
        if "iln_s" not in res:
            continue
        base = {k: v[k] for k in ("case", "group", "width", "reps", "shape", "shape_key",
                                  "elements")}
        t_iln = res["iln_s"]
        iln_results.append(dict(base, numpy_s=v["numpy_s"], ilnumerics_s=t_iln,
                                speedup=v["numpy_s"] / t_iln, identical=not reason,
                                fallbacks=0))
        diff_results.append(dict(base, viproc_s=v["viproc_s"], ilnumerics_s=t_iln,
                                 speedup=t_iln / v["viproc_s"],
                                 identical=bool(v["identical"]) and not reason,
                                 fallbacks=v["fallbacks"]))
    im = raw["machine"]
    m = vrun["machine"]
    iln_tag = (f"ILNumerics {im.get('ilnumerics')}, accelerator "
               f"{'on' if im.get('accelerator') else 'OFF'}, {im.get('ilnumerics_threads')} threads")
    note = ("ILNumerics has no 0-d arrays: the 0-d point runs on [1, 1] arrays. "
            "Same programs, reps and grid as the viproc run; inputs generated identically "
            "on both sides.")
    errors = [f"{r['case']} {'x'.join(map(str, r['shape'])) or '0-d'} ({r['error']})"
              for r in raw["results"] if "error" in r]
    if errors:
        note += (f" Not measured, so not in the medians ({len(errors)}): "
                 + "; ".join(errors) + ".")
    iln = derived_run(vrun, raw, iln_results, {
        "subject": "ILNumerics", "baseline": "NumPy", "title": f"{ILN} shape map",
        "prefix": "shape_map_ilnumerics",
        "check": f"Matches NumPy (rel. tol., {check_reps} steps)",
        "header": f"{m['cpu']}, {m['cores']} cores, {iln_tag}, NumPy {m['numpy']}",
        "note": note})
    diff = derived_run(vrun, raw, diff_results, {
        "subject": "viproc", "baseline": "ILNumerics", "title": f"viproc vs. {ILN}",
        "prefix": "shape_map_vs_ilnumerics",
        "check": "viproc bit-identical to NumPy and ILNumerics matching NumPy",
        "header": (f"{m['cpu']}, {m['cores']} cores, viproc {m['workers']} workers, "
                   f"build {m['build']}; {iln_tag}"),
        "note": note})
    return iln, diff, failed


def report(run, out_dir):
    prefix = run["labels"]["prefix"]
    with open(os.path.join(out_dir, f"{prefix}.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1)
    shape_map.write_markdown(run, os.path.join(out_dir, f"{prefix}.md"))
    return [os.path.join(out_dir, f"{prefix}.md")] + shape_map.plot(run, out_dir)


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    np.seterr(all="ignore")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--viproc", required=True, metavar="SHAPE_MAP_JSON",
                   help="viproc shape map run of this machine (grid, reps, NumPy times)")
    p.add_argument("--out", default="reports/shape_map", help="output directory")
    p.add_argument("--iln", metavar="RAW_JSON", help="use these ILNumerics results, do not run")
    p.add_argument("--filter", help="regular expression on the case names")
    p.add_argument("--threads", type=int, help="ILNumerics threads (default: its own)")
    p.add_argument("--repeat", type=int, help="timed runs per measurement (default: as viproc)")
    p.add_argument("--check-reps", type=int, default=3, help="steps of the result check")
    p.add_argument("--timeout", type=float, default=60,
                   help="seconds without a new result before a point counts as hung")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    vrun = json.load(open(args.viproc, encoding="utf-8"))
    if args.filter:
        import re

        vrun["results"] = [r for r in vrun["results"] if re.search(args.filter, r["case"])]
        vrun["machine"]["cases"] = len({r["case"] for r in vrun["results"]})
    plan = make_plan(vrun, args)
    plan_path = os.path.join(args.out, "ilnumerics_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f)
    raw_path = args.iln or os.path.join(args.out, "ilnumerics_raw.json")
    if not args.iln:
        run_csharp(plan, plan_path, raw_path, args.threads, args.timeout)
    raw = json.load(open(raw_path, encoding="utf-8"))
    if not raw["machine"].get("accelerator"):
        print("WARNING: the ILNumerics build did not run through the Accelerator")

    iln, diff, failed = build_runs(vrun, raw, plan, args.check_reps)
    files = report(iln, args.out) + report(diff, args.out)
    print("written: " + ", ".join(files))
    for line in failed:
        print("CHECK FAILED:", line)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
