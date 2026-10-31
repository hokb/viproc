"""Builds the README's benchmark figures from measured results.

All results were bit-identical to NumPy. Steps:

1. `--import` copies the raw results of a run into data/ as small CSV files:
       python make_charts.py --import --shape-map <shape_map.json>
           --loop <array_pipelining output> --kmeans <Part 3 values.csv>
   data/programs.tsv (bar chart) is written by hand from numbers documented
   in docs/benchmarks.md ("Real-world programs", strong_scaling.py, FFT).
   data/loop.csv's numpy_ft_threads column: the loop's thread pool on a
   free-threaded CPython 3.13t, measured with
   python/benchmarks/free_threaded_loop.py (speedup vs. NumPy there).
2. The figures are drawn with ILNumerics (plots/, .NET 8; ILNumerics.Drawing
   and ILNumerics.Toolboxes.Drawing2 for the bars):
       dotnet run -c Release --project plots -- data .
   shape_map, loop, kmeans, programs as .svg and .bmp (GDI)
3. This script turns the .bmp files into the .png files the README shows
   (Pillow):
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

        for name in ("shape_map", "loop", "kmeans", "programs"):
            bmp = os.path.join(HERE, name + ".bmp")
            if os.path.exists(bmp):
                Image.open(bmp).save(os.path.join(HERE, name + ".png"))
                os.remove(bmp)
