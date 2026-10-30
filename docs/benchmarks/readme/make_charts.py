"""Draws the README's benchmark charts from measured results.

Every number below is a measurement documented in docs/benchmarks.md (or
CLAUDE.md); all results were bit-identical to NumPy. Run:
    python docs/benchmarks/readme/make_charts.py
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NUMPY = "#a8a7a0"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def label_bars(ax, bars, fmt="{:.1f}x"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK)


def loop_chart():
    # docs/benchmarks.md, "Array pipelining", "Measured costs, chains":
    # i7-12700K, Windows, 1500 x 1500, build b7ee06d, speedup vs. NumPy at
    # 1/4/11 workers; NumPy with 12 threads (Parallel.For over the columns):
    # 0.43-0.74x (best shown); loop_dep gives wrong results with threads.
    workers = ["NumPy +\n12 threads", "viproc\n1 worker", "viproc\n4 workers", "viproc\n11 workers"]
    loop = [0.74, 1.23, 4.21, 3.11]
    dep = [None, 1.17, 4.17, 3.09]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = range(len(workers))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], loop, w * 0.92, color=[NUMPY] + [SERIES[0]] * 3,
                label="loop: C[i] = sum(abs(sin(A[:, i])))")
    b2 = ax.bar([i + w / 2 for i in x[1:]], dep[1:], w * 0.92, color=SERIES[1],
                label="loop_dep: C[:i+1] = ... (order matters)")
    label_bars(ax, b1)
    label_bars(ax, b2)
    ax.annotate("wrong\nresults", (0 + w / 2, 0.05), ha="center", va="bottom",
                fontsize=8, color=INK2)
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.set_xticks(list(x), workers)
    ax.set_ylabel("speedup vs. NumPy")
    ax.set_ylim(0, 6)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], label=b1.get_label()),
                       Patch(color=SERIES[1], label=b2.get_label()),
                       Patch(color=NUMPY, label="NumPy + thread pool (loop)")],
              frameon=False, fontsize=9, loc="upper left")
    ax.set_title("A plain Python loop over 1,500 columns (i7-12700K)", loc="left", fontsize=12)
    save(fig, "loop.png")


def programs_chart():
    # 4-core Xeon VM, bit-identical. Sources: CLAUDE.md / docs/benchmarks.md
    # (strong_scaling.py, k-means, FFT) and "Real-world programs" in
    # docs/benchmarks.md (python/benchmarks/realworld.py).
    rows = [
        ("8 independent sin/cos chains, 1e7", 7.5, None),
        ("FFT: 100x abs(ifft(fft(A))), n = 200+", 4.4, "4.4-5.7x"),
        ("k-means (ILNumerics), 1203 x 1203, k = 50", 3.9, None),
        ("Black-Scholes greeks, 1e6 options", 2.8, None),
        ("Heat equation stencil, 2048 x 2048", 2.2, None),
        ("Feature engineering, 200,000 x 20", 1.8, None),
        ("Logistic regression, 100,000 x 50", 0.94, None),
    ]
    rows.reverse()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ys = range(len(rows))
    vals = [r[1] for r in rows]
    colors = [SERIES[0] if v >= 1 else NUMPY for v in vals]
    bars = ax.barh(list(ys), vals, 0.62, color=colors)
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
    save(fig, "programs.png")


def size_chart():
    # docs/benchmarks.md, "Shape map", "After D11": 4-core Xeon VM, 4 workers,
    # 2026-10-15, build #38e22687; median speedup per case group and grid
    # point, aggregated (geomean) per half decade of elements.
    with open(os.path.join(HERE, "shape_map_d11_points.json")) as f:
        points = json.load(f)
    groups = ["independent", "mixed", "reductions", "dependent", "sync"]
    names = {"independent": "independent chains", "mixed": "mixed expressions",
             "reductions": "reductions", "dependent": "dependent chains",
             "sync": "sync after every step"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for g, color in zip(groups, SERIES):
        bins = {}
        for p in points:
            if p["elements"] < 1 or g not in p["groups"]:
                continue
            k = round(2 * math.log10(p["elements"]))
            bins.setdefault(k, []).append(math.log(p["groups"][g]))
        ks = sorted(bins)
        xs = [10 ** (k / 2) for k in ks]
        ys = [math.exp(sum(bins[k]) / len(bins[k])) for k in ks]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=4, label=names[g])
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.set_yticks([0.5, 1, 2, 4], ["0.5x", "1x", "2x", "4x"])
    ax.set_xlabel("elements per array (rows x columns)")
    ax.set_ylabel("speedup vs. NumPy")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Where it pays off: speedup over data size (4 cores)", loc="left", fontsize=12)
    save(fig, "sizes.png")


if __name__ == "__main__":
    loop_chart()
    programs_chart()
    size_chart()
