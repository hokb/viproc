"""Builds the README's benchmark figures from measured results.

All results were bit-identical to NumPy. Steps:

1. `--import` copies the raw results of a run into data/ as small CSV files:
       python make_charts.py --import --shape-map <shape_map.json>
           --loop <array_pipelining output> --kmeans <Part 3 values.csv>
2. The figures of measurements are drawn with ILNumerics (plots/, .NET 8):
       dotnet run -c Release --project plots -- data .
   shape_map, loop, kmeans as .svg and .bmp (GDI)
3. This script turns the .bmp files into the .png files the README shows
   (Pillow) and draws programs.png (matplotlib) from numbers documented in
   docs/benchmarks.md and CLAUDE.md, listed below.
       python make_charts.py
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def import_shape_map(path):
    """shape_map.json -> data/shape_map.csv: median speedup per grid point,
    over all cases and over the 'single' class (one op per step)."""
    with open(path) as f:
        d = json.load(f)
    with open(os.path.join(DATA, "shape_map.csv"), "w") as f:
        f.write("rows,cols,elements,all,single\n")
        for p in d["points"]:
            r, c = (p["shape"] + [1])[:2] if p["shape"] else (0, 0)
            f.write(f"{r},{c},{p['elements']},{p['median']:.4f},{p['classes']['single'][1]:.4f}\n")
    m = d["machine"]
    return f"{m['cpu']}, {m['cores']} cores, {m['workers']} workers, NumPy {m['numpy']}, build {m['build']}"


def import_loop(path, workers):
    """array_pipelining.py output -> data/loop.csv: speedups vs. NumPy per n
    (NumPy thread pool for `loop`, viproc `loop`, `loop_dep`, `vectorized`)."""
    text = open(path).read()
    rows = []
    for block in re.split(r"\n(?=A: )", text)[1:]:
        n = int(re.search(r"\[(\d+), \d+\]", block).group(1))
        header = next(l for l in block.splitlines() if l.startswith("| program"))
        cols = [c.strip() for c in header.strip("|").split("|")]
        vals = {}
        for line in block.splitlines():
            m = re.match(r"\| `(\w+)` \|", line)
            if m:
                cells = [c.strip() for c in line.strip("|").split("|")]
                vals[m.group(1)] = dict(zip(cols, cells))

        def speedup(prog, col):
            cell = vals[prog].get(col, "–")
            m = re.search(r"\(([\d.]+)x\)", cell)
            return float(m.group(1)) if m else float("nan")

        thr = next(c for c in cols if c.startswith("NumPy") and "threads" in c)
        vp = f"viproc w={workers}"
        rows.append((n, speedup("loop", thr), speedup("loop", vp), speedup("loop_dep", vp),
                     speedup("vectorized", vp)))
    with open(os.path.join(DATA, "loop.csv"), "w") as f:
        f.write("n,numpy_threads,loop,loop_dep,vectorized\n")
        for r in rows:
            f.write(",".join(f"{v:g}" for v in r) + "\n")


def import_kmeans(path):
    """Part 3's values.csv (id, elapsed ms, ms per run) -> data/kmeans.csv."""
    with open(path) as f, open(os.path.join(DATA, "kmeans.csv"), "w") as out:
        out.write("id,elapsed_ms,run_ms\n")
        for line in f:
            if line.strip():
                out.write(",".join(v.strip() for v in line.split(",")) + "\n")


def programs_chart():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({
        "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    # 4-core Xeon VM, bit-identical. Sources: CLAUDE.md / docs/benchmarks.md
    # (strong_scaling.py, FFT) and "Real-world programs" in docs/benchmarks.md
    # (python/benchmarks/realworld.py).
    rows = [
        ("8 independent sin/cos chains, 1e7", 7.5, None),
        ("FFT: 100x abs(ifft(fft(A))), n = 200+", 4.4, "4.4-5.7x"),
        ("Black-Scholes greeks, 1e6 options", 2.8, None),
        ("Heat equation stencil, 2048 x 2048", 2.2, None),
        ("Feature engineering, 200,000 x 20", 1.8, None),
        ("Logistic regression, 100,000 x 50", 0.94, None),
    ]
    rows.reverse()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ys = range(len(rows))
    vals = [r[1] for r in rows]
    bars = ax.barh(list(ys), vals, 0.62, color=["#2a78d6" if v >= 1 else "#a8a7a0" for v in vals])
    for b, (_, v, text) in zip(bars, rows):
        ax.annotate(text or (f"{v:.2f}x" if v < 1 else f"{v:.1f}x"),
                    (v, b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=9)
    ax.axvline(1.0, color=INK2, lw=1, ls="--")
    ax.set_xlabel("speedup vs. NumPy, 4 cores")
    ax.set_xlim(0, 8.6)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("Unchanged NumPy programs, bit-identical results", loc="left", fontsize=12)
    fig.savefig(os.path.join(HERE, "programs.png"), dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--import", dest="do_import", action="store_true")
    p.add_argument("--shape-map")
    p.add_argument("--loop")
    p.add_argument("--loop-workers", type=int, default=4)
    p.add_argument("--kmeans")
    a = p.parse_args()
    if a.do_import:
        os.makedirs(DATA, exist_ok=True)
        if a.shape_map:
            print(import_shape_map(a.shape_map))
        if a.loop:
            import_loop(a.loop, a.loop_workers)
        if a.kmeans:
            import_kmeans(a.kmeans)
    else:
        from PIL import Image

        for name in ("shape_map", "loop", "kmeans"):
            bmp = os.path.join(HERE, name + ".bmp")
            if os.path.exists(bmp):
                Image.open(bmp).save(os.path.join(HERE, name + ".png"))
                os.remove(bmp)
        programs_chart()
