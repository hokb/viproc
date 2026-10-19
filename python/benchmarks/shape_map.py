"""viproc shape map: speedup vs. NumPy over the 2-d problem size (rows x cols).

Runs every case of shape_map_cases.py at every point of a log-spaced grid
of shapes (rows, cols), as plain NumPy and on viproc (one worker per core),
checks that the results are identical and condenses the per-case speedups
(NumPy time / viproc time) of each grid point into

- the **median** over the cases (the representative value of the point:
  robust against single outliers; on ratios, the median of the logs is
  the log of the median), and
- the **interquartile range** (25th..75th percentile) as its spread, given
  as the factor q75 / q25.

Grid: rows and columns 1 .. 1,000 in half decades (1, 3, 10, 32, 100, 316,
1,000); beyond 1,000 per dimension the data leaves the caches and the map
shows nothing new (--max-rows / --max-cols extend it). `--quick` uses whole
decades. A 0-d point (true NumPy scalars) is measured as well and reported
next to the grid.

Usage:
    PYTHONPATH=build/python python3 python/benchmarks/shape_map.py [--quick] [--out DIR]
    python3 python/benchmarks/shape_map.py --plot DIR/shape_map.json [--out DIR] [--groups G,..]

Writes DIR/shape_map.json, DIR/shape_map.md and, with matplotlib installed,
DIR/shape_map.png (median and spread) and DIR/shape_map_groups.png (median
per case group).
"""

import argparse
import json
import math
import os
import re
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shape_map_cases  # noqa: E402
from suite import (best_of, calibrate_reps, check_out_dir, identical, machine_info,  # noqa: E402
                   materialize, physical_memory, repeats_for, usable_cores)

MEM_BUDGET = 0.5  # fraction of RAM a single measurement may use


def grid_axis(top, quick):
    """Log-spaced extents 1 .. top: half decades (whole decades with quick);
    a top that is no grid value becomes the last point."""
    step = 1.0 if quick else 0.5
    values, k = [], 0.0
    while round(10 ** k) <= top:
        values.append(int(round(10 ** k)))
        k += step
    if values[-1] != top:
        values.append(top)
    return values


def fits(case, elements, ram, itemsize=8):
    return elements * itemsize * case.mem <= MEM_BUDGET * ram


def measure(case, shape, cores, workers, target, repeat, seed, module=False):
    """NumPy and viproc time of one case at one shape (best of `repeat`).
    module: the viproc run calls the cases' `np.*` through the viproc module,
    as `python -m viproc` / `import viproc as np` do (ufunc proxies, where)."""
    import viproc

    width = cores if case.width == "cores" else case.width
    rng = np.random.default_rng(seed)
    data = case.make(rng, shape, width)
    reps = calibrate_reps(case, data, target)
    t0 = time.perf_counter()
    expected = materialize(case.run(data, reps))
    warm = time.perf_counter() - t0
    t_np, _ = best_of(lambda: case.run(data, reps), repeats_for(warm, repeat))

    viproc.init(workers=workers)
    vdata = [viproc.asarray(a) for a in data]
    viproc.wait_all()

    def run():
        if module:
            shape_map_cases.np = viproc
        try:
            out = case.run(vdata, reps)
        finally:
            shape_map_cases.np = np
        viproc.wait_all()
        return out

    f0 = viproc._ndarray.fallback_calls
    t0 = time.perf_counter()
    same = identical(materialize(run()), expected)  # also the warm-up
    warm = time.perf_counter() - t0
    fallbacks = viproc._ndarray.fallback_calls - f0
    t_vp, _ = best_of(run, repeats_for(warm, repeat))
    del vdata, expected
    viproc.shutdown()
    return {"case": case.name, "group": case.group, "width": width, "reps": reps,
            "numpy_s": t_np, "viproc_s": t_vp, "speedup": t_np / t_vp,
            "identical": bool(same), "fallbacks": fallbacks}


def quartiles(values):
    """(q25, median, q75) of the speedups, interpolated on the log scale."""
    logs = np.log([v for v in values if v > 0])
    if not len(logs):
        return (float("nan"),) * 3
    return tuple(float(np.exp(np.percentile(logs, q))) for q in (25, 50, 75))


def run_groups(run):
    """The case groups a run measured, in the order of shape_map_cases.GROUPS."""
    have = {r["group"] for r in run["results"]}
    return [g for g in shape_map_cases.GROUPS if g in have]


def select_groups(run, groups):
    """A run restricted to the cases of `groups` (points summarized again)."""
    results = [r for r in run["results"] if r["group"] in groups]
    machine = dict(run["machine"], cases=len({r["case"] for r in results}))
    return dict(run, machine=machine, results=results,
                skipped=[s for s in run["skipped"]
                         if any(c.name == s["case"] and c.group in groups
                                for c in shape_map_cases.CASES)],
                points=summarize(results))


def summarize(results):
    """Per grid point: median and quartiles over all cases and per group."""
    points = {}
    for r in results:
        points.setdefault(r["shape_key"], []).append(r)
    out = []
    for key, rs in points.items():
        q25, med, q75 = quartiles([r["speedup"] for r in rs])
        groups = {}
        for g in shape_map_cases.GROUPS:
            gs = [r["speedup"] for r in rs if r["group"] == g]
            if gs:
                groups[g] = quartiles(gs)[1]
        out.append({"shape": rs[0]["shape"], "elements": rs[0]["elements"],
                    "cases": len(rs), "median": med, "q25": q25, "q75": q75,
                    "spread": q75 / q25, "groups": groups,
                    "identical": sum(r["identical"] for r in rs),
                    "fallbacks": sum(r["fallbacks"] for r in rs)})
    return out


def run_map(args):
    cores = args.cores or usable_cores()
    ram = physical_memory()
    workers = args.workers or cores
    target = 0.02 if args.quick else 0.04
    repeat = args.repeat or 3
    cols = grid_axis(args.max_cols, args.quick)
    rows = grid_axis(args.max_rows, args.quick)
    shapes = [()] + [(r, c) for r in rows for c in cols]
    cases = [c for c in shape_map_cases.CASES
             if (not args.filter or re.search(args.filter, c.name))
             and (not args.groups or c.group in args.groups)]
    info = machine_info(cores, ram)
    info.update({"workers": workers, "repeat": repeat, "target_s": target,
                 "rows": rows, "cols": cols, "cases": len(cases), "module": args.module})
    print(f"viproc shape map: {len(rows)} x {len(cols)} shapes + 0-d, {len(cases)} cases, "
          f"{cores} cores, {workers} workers, build {info['build']}", flush=True)

    results, skipped = [], []
    for si, shape in enumerate(shapes):
        elements = int(np.prod(shape)) if shape else 1
        t0 = time.perf_counter()
        point = []
        for ci, case in enumerate(cases):
            if not shape and not case.scalar_ok:
                continue
            if not fits(case, elements, ram):
                skipped.append({"case": case.name, "shape": list(shape)})
                continue
            r = measure(case, shape, cores, workers, target, repeat, seed=si * 100 + ci,
                        module=args.module)
            r.update({"shape": list(shape), "shape_key": "x".join(map(str, shape)) or "0-d",
                      "elements": elements})
            point.append(r)
        results += point
        q25, med, q75 = quartiles([r["speedup"] for r in point])
        bad = [r["case"] for r in point if not r["identical"]]
        label = "x".join(map(str, shape)) or "0-d"
        print(f"[{si + 1:3}/{len(shapes)}] {label:>12} {elements:>11,} el.  "
              f"median {med:6.2f}x  IQR {q25:5.2f}..{q75:5.2f}x  "
              f"({len(point)} cases, {time.perf_counter() - t0:5.1f} s)"
              + (f"  RESULT DIFFERS: {','.join(bad)}" if bad else ""), flush=True)
    return {"machine": info, "results": results, "skipped": skipped,
            "points": summarize(results)}


# --- reports -----------------------------------------------------------------------------


def labels(run):
    """What a run compares: `subject` vs. `baseline`, speedup = baseline time /
    subject time. Runs of this script compare viproc with NumPy; other
    comparisons (shape_map_ilnumerics.py) store their own labels in the run."""
    out = {"subject": "viproc", "baseline": "NumPy", "title": "viproc shape map",
           "prefix": "shape_map", "check": "Bit-identical to NumPy", "header": None,
           "note": None}
    out.update(run.get("labels") or {})
    if out["header"] is None:
        m = run["machine"]
        out["header"] = (f"{m['cpu']}, {m['cores']} cores, {m['workers']} workers, "
                         f"NumPy {m['numpy']}, build {m['build']}")
    return out


def write_markdown(run, path):
    m = run["machine"]
    lb = labels(run)
    sub_, base = lb["subject"], lb["baseline"]
    pts = {p["shape"] and tuple(p["shape"]) or (): p for p in run["points"]}
    rows, cols = m["rows"], m["cols"]
    lines = [
        f"# {lb['title']}",
        "",
        f"{lb['header']}, Python {m['python']}, {m['date']}.",
        "",
        f"Median speedup of {sub_} vs. {base} over all cases ({base} time / {sub_} time; "
        f"> 1: {sub_} faster), and in brackets the interquartile range "
        f"(25th..75th percentile) over the {m['cases']} cases. Rows down, columns across.",
        "",
    ] + ([lb["note"], ""] if lb["note"] else []) + [
        "| rows \\ cols | " + " | ".join(f"{c:,}" for c in cols) + " |",
        "|---:|" + "---:|" * len(cols),
    ]
    for r in rows:
        cells = []
        for c in cols:
            p = pts.get((r, c))
            cells.append("–" if p is None else
                         f"**{p['median']:.2f}** ({p['q25']:.2f}–{p['q75']:.2f})")
        lines.append(f"| **{r:,}** | " + " | ".join(cells) + " |")
    s = pts.get(())
    if s:
        lines += ["", f"0-d scalars: median **{s['median']:.2f}x** "
                      f"({s['q25']:.2f}–{s['q75']:.2f}, {s['cases']} cases)."]
    total = len(run["results"])
    same = sum(r["identical"] for r in run["results"])
    lines += ["", f"{lb['check']}: {same} / {total} measurements."]
    if run["skipped"]:
        lines.append(f"Skipped (would not fit into RAM): {len(run['skipped'])} measurements.")
    lines += ["", "## Median per case group", ""]
    for g in run_groups(run):
        lines += [f"### {g}", "", "| rows \\ cols | " + " | ".join(f"{c:,}" for c in cols)
                  + " |", "|---:|" + "---:|" * len(cols)]
        for r in rows:
            cells = []
            for c in cols:
                v = pts.get((r, c), {}).get("groups", {}).get(g)
                cells.append("–" if v is None else f"{v:.2f}")
            lines.append(f"| **{r:,}** | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Cases", "", "| case | group | P | program |", "|---|---|---:|---|"]
    for c in shape_map_cases.CASES:
        if c.group not in run_groups(run):
            continue
        lines.append(f"| {c.name} | {c.group} | {c.width} | {c.description} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# Diverging map around parity (1x): red = slower than NumPy, blue = faster,
# neutral gray at 1x. Steps from the dataviz reference palette.
SLOW, MID, FAST = "#b8332f", "#f0efec", "#1f5fae"
INK, MUTED = "#2b2b29", "#6b6b66"


def _edges(values):
    """Cell edges on a log axis: geometric midpoints, half a step outside."""
    v = np.log10(np.asarray(values, dtype=float))
    if len(v) == 1:
        return 10 ** np.array([v[0] - 0.25, v[0] + 0.25])
    mids = (v[1:] + v[:-1]) / 2
    first = v[0] - (mids[0] - v[0])
    last = v[-1] + (v[-1] - mids[-1])
    return 10 ** np.concatenate([[first], mids, [last]])


def _grid(run, value):
    m = run["machine"]
    pts = {tuple(p["shape"]): p for p in run["points"] if p["shape"]}
    z = np.full((len(m["rows"]), len(m["cols"])), np.nan)
    for i, r in enumerate(m["rows"]):
        for j, c in enumerate(m["cols"]):
            p = pts.get((r, c))
            if p is not None:
                z[i, j] = value(p)
    return z


def _axes(ax, run, title):
    m = run["machine"]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(m["cols"])
    ax.set_xticklabels([f"{c:,}" for c in m["cols"]], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(m["rows"])
    ax.set_yticklabels([f"{r:,}" for r in m["rows"]], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("columns", color=INK)
    ax.set_ylabel("rows", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)


def _annotate(ax, run, z, fmt, cmap, norm, fontsize=7):
    m = run["machine"]
    for i, r in enumerate(m["rows"]):
        for j, c in enumerate(m["cols"]):
            if math.isnan(z[i, j]):
                continue
            rgba = cmap(norm(z[i, j]))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(c, r, fmt(i, j), ha="center", va="center", fontsize=fontsize,
                    color="white" if lum < 0.5 else INK)


def plot(run, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    except ImportError:
        print("matplotlib not installed: no images (pip install matplotlib)")
        return []
    m = run["machine"]
    xe, ye = _edges(m["cols"]), _edges(m["rows"])
    cmap = LinearSegmentedColormap.from_list("speedup", [SLOW, MID, FAST])
    cmap.set_bad("#ffffff")
    top = max(math.log10(2 * m["workers"]), 0.5)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=top)  # log10 of the speedup
    med = _grid(run, lambda p: p["median"])
    q25, q75 = _grid(run, lambda p: p["q25"]), _grid(run, lambda p: p["q75"])
    spread = q75 / q25
    scalar = next((p for p in run["points"] if not p["shape"]), None)
    lb = labels(run)
    sub_, base = lb["subject"], lb["baseline"]
    header = f"{sub_} vs. {base}: {lb['header']}"

    files = []
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.8), facecolor="white",
                            gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    mesh = ax.pcolormesh(xe, ye, np.log10(med), cmap=cmap, norm=norm)
    _axes(ax, run, f"Median speedup over {m['cases']} cases ({base} time / {sub_} time), "
                   "IQR below")
    _annotate(ax, run, np.log10(med),
              lambda i, j: f"{med[i, j]:.2f}x\n{q25[i, j]:.1f}–{q75[i, j]:.1f}", cmap, norm,
              fontsize=6.5)
    ticks = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
    ticks = [t for t in ticks if -1 <= math.log10(t) <= top]
    cb = fig.colorbar(mesh, ax=ax, ticks=[math.log10(t) for t in ticks], shrink=0.85)
    cb.ax.set_yticklabels([f"{t:g}x" for t in ticks], fontsize=8)
    cb.outline.set_visible(False)
    cb.set_label(f"{sub_} faster than {base}  ↑   ·   ↓  slower", color=MUTED, fontsize=8)

    ax = axs[1]
    seq = LinearSegmentedColormap.from_list("spread", ["#f4f3f0", "#5b5b57"])
    seq.set_bad("#ffffff")
    from matplotlib.colors import LogNorm

    snorm = LogNorm(vmin=1.0, vmax=max(2.0, float(np.nanmax(spread))))
    mesh = ax.pcolormesh(xe, ye, spread, cmap=seq, norm=snorm)
    _axes(ax, run, "Spread: interquartile range as factor q75 / q25")
    _annotate(ax, run, spread, lambda i, j: f"{spread[i, j]:.1f}", seq, snorm)
    sticks = [t for t in (1, 1.25, 1.5, 2, 3, 4, 6, 8, 12, 16) if t <= snorm.vmax]
    cb = fig.colorbar(mesh, ax=ax, shrink=0.85, ticks=sticks)
    cb.ax.minorticks_off()
    cb.ax.set_yticklabels([f"{t:g}x" for t in sticks], fontsize=8)
    cb.outline.set_visible(False)
    foot = header
    if scalar:
        foot += (f"   ·   0-d scalars: median {scalar['median']:.2f}x "
                 f"(IQR {scalar['q25']:.2f}–{scalar['q75']:.2f})")
    fig.text(0.01, 0.01, foot, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = os.path.join(out_dir, f"{lb['prefix']}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    files.append(path)

    groups = run_groups(run)
    ncols = 3 if len(groups) <= 5 else 4 if len(groups) <= 7 else 3
    nrows = -(-(len(groups) + 1) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.3 * ncols, 6 * nrows), facecolor="white",
                            layout="constrained")
    axs = axs.ravel()
    names = {g: [c.name for c in shape_map_cases.CASES if c.group == g] for g in groups}
    for k, g in enumerate(groups):
        ax = axs[k]
        z = _grid(run, lambda p, g=g: p["groups"].get(g, float("nan")))
        mesh = ax.pcolormesh(xe, ye, np.log10(z), cmap=cmap, norm=norm)
        _axes(ax, run, f"{g} ({len(names[g])} cases): median speedup")
        _annotate(ax, run, np.log10(z), lambda i, j, z=z: f"{z[i, j]:.2f}", cmap, norm)
    for ax in axs[len(groups):]:
        ax.axis("off")
    ax = axs[len(groups)]
    text = "\n".join(f"{g}: " + ", ".join(names[g]) for g in groups)
    ax.text(0, 0.62, "Cases per group\n\n" + text.replace(", ", ",\n    ")
            + "\n\n" + header.replace(", NumPy", ",\nNumPy").replace(": ", ":\n", 1)
            + (f"\n\n{lb['note']}" if lb["note"] else ""),
            va="top", fontsize=8, color=INK)
    cax = ax.inset_axes([0.0, 0.8, 0.9, 0.05])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal",
                      ticks=[math.log10(t) for t in ticks])
    cb.ax.set_xticklabels([f"{t:g}x" for t in ticks], fontsize=8)
    cb.outline.set_visible(False)
    cb.set_label(f"median speedup of {sub_} vs. {base} (red: slower, blue: faster)", color=MUTED,
                 fontsize=8)
    path = os.path.join(out_dir, f"{lb['prefix']}_groups.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    files.append(path)
    return files


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    np.seterr(all="ignore")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="reports/shape_map", help="output directory")
    p.add_argument("--quick", action="store_true", help="whole decades, shorter runs")
    p.add_argument("--filter", help="regular expression on the case names")
    p.add_argument("--workers", type=int, help="viproc workers (default: cores)")
    p.add_argument("--cores", type=int, help="override the detected core count")
    p.add_argument("--repeat", type=int, help="timed runs per measurement (best of, default 3)")
    p.add_argument("--max-cols", type=int, default=1_000)
    p.add_argument("--max-rows", type=int, default=1_000)
    p.add_argument("--groups", type=lambda s: s.split(","),
                   help="comma-separated case groups (default: all); with --plot: report "
                        "only these groups of the run")
    p.add_argument("--module", action="store_true",
                   help="viproc runs call np.* through the viproc module (python -m viproc)")
    p.add_argument("--plot", metavar="SHAPE_MAP_JSON", help="only redraw the reports of a run")
    p.add_argument("--reference", action="store_true",
                   help="allow --out to replace results committed to git (reference runs)")
    args = p.parse_args()
    check_out_dir(args.out, args.reference)
    os.makedirs(args.out, exist_ok=True)
    if args.plot:
        run = json.load(open(args.plot, encoding="utf-8"))
        if args.groups:
            run = select_groups(run, args.groups)
    else:
        run = run_map(args)
        with open(os.path.join(args.out, "shape_map.json"), "w", encoding="utf-8") as f:
            json.dump(run, f, indent=1)
    write_markdown(run, os.path.join(args.out, "shape_map.md"))
    files = plot(run, args.out)
    print(f"written: {args.out}/shape_map.md" + "".join(f", {f}" for f in files))
    if not all(r["identical"] for r in run["results"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
