"""Reports of the viproc benchmark suite (suite.py): Markdown and a
self-contained HTML page (no scripts, no external resources), for one run
or a comparison of several runs (machines, platforms)."""

import html
import math
import os

CATEGORY_ORDER = ["chains", "indexing", "reductions", "algorithms", "dtypes", "scalars"]
SIZE_ORDER = ["scalar", "tiny", "small", "medium", "large"]

SCORE_EXPLAINED = """\
**How the score works.** Each configuration runs as plain NumPy and on viproc
(one worker per core). *Speedup* = NumPy time / viproc time. Its *ideal* is
min(cores, P) for medium and large data, where P is the program's parallel
width (independent instruction streams), and 1x (parity with NumPy) for
scalar, tiny and small data, where no hand-off can pay off. *Efficiency* =
speedup / ideal: dividing by the cores makes it comparable across machines.
The `large` size class scales with RAM per core. **Score** = 100 x geometric
mean over the categories of the geometric mean of their efficiencies
(100 = ideal everywhere; above 100 is possible, e.g. through temporary
reuse). *Fallbacks* count operations viproc ran eagerly in NumPy (each one a
sync point); they show what still keeps a case from scaling. "NumPy only"
marks inputs of a dtype viproc does not support yet (int16, float16): they
stay NumPy arrays, and their ops run in plain NumPy."""


def _fallbacks(r):
    """Fallback count, or which input dtypes viproc left to plain NumPy."""
    if r.get("not_in_viproc"):
        return "NumPy only (" + ", ".join(r["not_in_viproc"]) + ")"
    return str(r["fallbacks"])


def _fmt_ms(s):
    ms = s * 1e3
    return f"{ms:.3f}" if ms < 1 else f"{ms:.2f}" if ms < 100 else f"{ms:.0f}"


def _fmt_x(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.2f}x"


def _fmt_score(v):
    return "–" if v is None or math.isnan(v) else f"{v:.0f}"


def _gib(b):
    return f"{b / 2**30:.1f} GiB"


def _ordered(d, order):
    return [k for k in order if k in d] + sorted(k for k in d if k not in order)


# --- Markdown -----------------------------------------------------------------


def summary_markdown(run):
    m, s = run["machine"], run["summary"]
    wmax = max(m["workers"])
    lines = [
        f"## Parallel efficiency score: **{_fmt_score(s['score'])}** / 100",
        "",
        f"{m['label']} · {m['cpu']} · {m['cores']} cores · {_gib(m['ram_bytes'])} RAM · "
        f"Python {m['python']} · NumPy {m['numpy']} · viproc {m['viproc']}"
        f"{' · build ' + m['build'] if m.get('build') else ''}",
        "",
        "| | |",
        "|---|---|",
        f"| speedup vs NumPy, {wmax} workers (geometric mean) | {_fmt_x(s['speedup_geomean'])} |",
        f"| speedup vs NumPy, 1 worker (overhead) | {_fmt_x(s['speedup_w1_geomean'])} |",
        f"| results identical to NumPy | {s['identical']} / {s['configs']} |",
        "",
        "| category | score | | size class | score |",
        "|---|---:|---|---|---:|",
    ]
    cats = _ordered(s["categories"], CATEGORY_ORDER)
    sizes = _ordered(s["sizes"], SIZE_ORDER)
    for i in range(max(len(cats), len(sizes))):
        c = cats[i] if i < len(cats) else ""
        z = sizes[i] if i < len(sizes) else ""
        lines.append(f"| {c} | {_fmt_score(s['categories'][c]) if c else ''} | "
                     f"| {z} | {_fmt_score(s['sizes'][z]) if z else ''} |")
    return "\n".join(lines)


def report_markdown(run):
    m = run["machine"]
    wmax = max(m["workers"])
    out = ["# viproc benchmark suite", "", summary_markdown(run), "", "## Machine", "",
           "| | |", "|---|---|"]
    for k, label in [("os", "OS"), ("cpu", "CPU"), ("cores", "cores used"),
                     ("logical_cpus", "logical CPUs"), ("python", "Python"),
                     ("numpy", "NumPy"), ("viproc", "viproc"), ("build_type", "build type"),
                     ("elision", "temporary elision"), ("blas_threads", "BLAS threads"),
                     ("date", "date")]:
        out.append(f"| {label} | {m.get(k, '')} |")
    out += [f"| RAM | {_gib(m['ram_bytes'])} |",
            f"| large size class | {m['large_elements']:,} elements |",
            f"| timing | best of {m['repeat']}, ~{m['target_s'] * 1e3:.0f} ms per NumPy run |",
            "", "## Results", ""]
    workers = m["workers"]
    head = ["case", "size", "layout", "dtype", "P", "NumPy ms"]
    head += [f"viproc w={w} ms" for w in workers]
    head += ["speedup", "ideal", "efficiency", "fallbacks", "identical"]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---"] * 4 + ["---:"] * (len(head) - 5) + [":-:"]) + "|")
    for cat in _ordered({r["category"] for r in run["results"]}, CATEGORY_ORDER):
        out.append(f"| **{cat}** |" + " |" * (len(head) - 1))
        for r in run["results"]:
            if r["category"] != cat:
                continue
            row = [r["case"], r["size"], r["layout"], r.get("dtype", ""), str(r["width"]),
                   _fmt_ms(r["numpy_s"])]
            row += [_fmt_ms(r["viproc_s"][str(w)]) for w in workers]
            row += [_fmt_x(r["speedup"]), _fmt_x(r["ideal"]), f"{r['efficiency']:.2f}",
                    _fallbacks(r), "yes" if r["identical"] else "**NO**"]
            out.append("| " + " | ".join(row) + " |")
    out += ["", SCORE_EXPLAINED, "", "Case descriptions:", ""]
    seen = set()
    for r in run["results"]:
        if r["case"] not in seen:
            seen.add(r["case"])
            out.append(f"- `{r['case']}` ({r['category']}): {r['description']}")
    out.append("")
    out.append(f"viproc ran with {wmax} workers for the score; speedups use that column.")
    return "\n".join(out) + "\n"


# --- HTML ---------------------------------------------------------------------

CSS = """
:root {
  color-scheme: light;
  --surface-0: #f4f3f0; --surface-1: #fcfcfb; --line: #dddcd7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #6e6d68;
  --faster: #2a78d6; --slower: #e34948; --ideal: #0b0b0b;
  --good: #0ca30c; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --line: #383835;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a9990;
    --faster: #3987e5; --slower: #e66767; --ideal: #f0efec;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --line: #383835;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a9990;
  --faster: #3987e5; --slower: #e66767; --ideal: #f0efec;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 10px; }
.sub { color: var(--text-secondary); margin: 0 0 20px; }
.hero { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
.hero.wide { grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); }
.tile { background: var(--surface-1); border: 1px solid var(--line); border-radius: 8px;
  padding: 14px 16px; min-width: 0; }
.tile .label { color: var(--text-secondary); font-size: 13px; }
.tile .value { font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile.main .value { font-size: 52px; line-height: 1.1; }
.tile .note { color: var(--text-muted); font-size: 12px; }
.panel { background: var(--surface-1); border: 1px solid var(--line); border-radius: 8px;
  padding: 4px 0; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 5px 10px; text-align: left; white-space: nowrap; }
th { color: var(--text-secondary); font-weight: 500; font-size: 12px;
  border-bottom: 1px solid var(--line); }
td.num, th.num { text-align: right; }
tr.cat td { font-weight: 600; padding-top: 14px; border-bottom: 1px solid var(--line); }
tbody tr:hover td { background: color-mix(in srgb, var(--line) 35%, transparent); }
.bar { position: relative; width: 260px; height: 18px; }
.bar .axis { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--text-muted); }
.bar .fill { position: absolute; top: 4px; height: 10px; }
.bar .fill.faster { background: var(--faster); border-radius: 0 4px 4px 0; }
.bar .fill.slower { background: var(--slower); border-radius: 4px 0 0 4px; }
.bar .ideal { position: absolute; top: 1px; height: 16px; width: 2px; background: var(--ideal); }
.bar .tip { display: none; position: absolute; left: 50%; bottom: 22px; transform: translateX(-50%);
  background: var(--surface-1); color: var(--text-primary); border: 1px solid var(--line);
  border-radius: 6px; padding: 6px 8px; font-size: 12px; white-space: nowrap; z-index: 2;
  box-shadow: 0 2px 8px rgba(0,0,0,.15); }
.bar:hover .tip { display: block; }
.ok { color: var(--good); } .bad { color: var(--critical); font-weight: 600; }
.legend { display: flex; gap: 18px; color: var(--text-secondary); font-size: 12px;
  margin: 8px 2px; flex-wrap: wrap; }
.sw { display: inline-block; width: 12px; height: 10px; border-radius: 2px;
  vertical-align: -1px; margin-right: 5px; }
.explain { color: var(--text-secondary); max-width: 860px; }
code { font-size: 12px; }
"""

# log2 scale of the speedup bars: 1/16x .. 16x, 1x in the middle
_BAR_W, _BAR_SPAN = 260, 4.0


def _bar_x(v):
    v = min(max(v, 2 ** -_BAR_SPAN), 2 ** _BAR_SPAN)
    return _BAR_W / 2 * (1 + math.log2(v) / _BAR_SPAN)


def _bar(speedup, ideal, tip):
    mid = _BAR_W / 2
    x = _bar_x(speedup)
    cls = "faster" if speedup >= 1 else "slower"
    left, width = (mid, max(1.0, x - mid)) if speedup >= 1 else (x, max(1.0, mid - x))
    parts = [f'<div class="bar"><span class="axis" style="left:{mid:.0f}px"></span>',
             f'<span class="fill {cls}" style="left:{left:.1f}px;width:{width:.1f}px"></span>']
    if ideal:
        parts.append(f'<span class="ideal" style="left:{_bar_x(ideal) - 1:.1f}px"></span>')
    parts.append(f'<span class="tip">{tip}</span></div>')
    return "".join(parts)


def _legend():
    return ('<div class="legend"><span><span class="sw" style="background:var(--faster)">'
            '</span>faster than NumPy</span><span><span class="sw" '
            'style="background:var(--slower)"></span>slower than NumPy</span><span>'
            '<span class="sw" style="background:var(--ideal);width:2px"></span>ideal '
            'speedup</span><span>log scale, 1/16x … 16x, center line = NumPy</span></div>')


def _tiles(run):
    m, s = run["machine"], run["summary"]
    wmax = max(m["workers"])
    ok = s["identical"] == s["configs"]
    t = [
        ('main', "Parallel efficiency score", _fmt_score(s["score"]),
         "100 = ideal scaling on this machine"),
        ('', f"Speedup vs NumPy ({wmax} workers)", _fmt_x(s["speedup_geomean"]),
         "geometric mean over all cases"),
        ('', "Speedup vs NumPy (1 worker)", _fmt_x(s["speedup_w1_geomean"]),
         "overhead of the async runtime"),
        ('', "Identical to NumPy",
         f'<span class="{"ok" if ok else "bad"}">{"✓" if ok else "✗"}</span> '
         f'{s["identical"]}/{s["configs"]}', "bit-identical results"),
    ]
    return '<div class="hero">' + "".join(
        f'<div class="tile {c}"><div class="label">{html.escape(a)}</div>'
        f'<div class="value">{b}</div><div class="note">{html.escape(n)}</div></div>'
        for c, a, b, n in t) + "</div>"


def _score_tables(s):
    rows = []
    for title, d, order in [("category", s["categories"], CATEGORY_ORDER),
                            ("size class", s["sizes"], SIZE_ORDER)]:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td class='num'>{_fmt_score(d[k])}</td>"
                       f"<td>{_score_bar(d[k])}</td></tr>" for k in _ordered(d, order))
        rows.append(f'<div class="panel"><table><thead><tr>'
                    f'<th>{title}</th><th class="num">score</th><th></th></tr></thead>'
                    f'<tbody>{body}</tbody></table></div>')
    return '<div class="hero wide">' + "".join(rows) + "</div>"


def _score_bar(v):
    """Bar of a 0..100 score (wider than 100 is clipped), ideal tick at 100."""
    if v is None or math.isnan(v):
        return ""
    w = min(150.0, max(1.0, v * 1.5))
    return (f'<div class="bar" style="width:152px"><span class="fill faster" '
            f'style="left:0;width:{w:.0f}px"></span><span class="ideal" '
            f'style="left:149px"></span></div>')


def report_html(run):
    m = run["machine"]
    workers = m["workers"]
    wmax = max(workers)
    head = ("<tr><th>case</th><th>size</th><th>layout</th><th>dtype</th>"
            "<th class='num'>P</th>"
            "<th class='num'>NumPy ms</th>"
            + "".join(f"<th class='num'>viproc w={w}</th>" for w in workers)
            + "<th>speedup (w=" + str(wmax) + ")</th><th class='num'>speedup</th>"
            "<th class='num'>efficiency</th><th class='num'>fallbacks</th><th>identical</th></tr>")
    body = []
    for cat in _ordered({r["category"] for r in run["results"]}, CATEGORY_ORDER):
        body.append(f"<tr class='cat'><td colspan='{11 + len(workers)}'>{html.escape(cat)}"
                    f" — score {_fmt_score(run['summary']['categories'].get(cat))}</td></tr>")
        for r in run["results"]:
            if r["category"] != cat:
                continue
            tip = (f"{html.escape(r['case'])} · {r['size']} ({r['elements']:,} el.) · "
                   f"{r['layout']} · {r.get('dtype', '')}<br>"
                   f"speedup {_fmt_x(r['speedup'])}, ideal {_fmt_x(r['ideal'])}, efficiency {r['efficiency']:.2f}<br>"
                   f"{html.escape(r['description'])}")
            body.append(
                f"<tr><td title='{html.escape(r['description'])}'>{html.escape(r['case'])}</td>"
                f"<td>{r['size']}</td><td>{r['layout']}</td><td>{r.get('dtype', '')}</td>"
                f"<td class='num'>{r['width']}</td>"
                f"<td class='num'>{_fmt_ms(r['numpy_s'])}</td>"
                + "".join(f"<td class='num'>{_fmt_ms(r['viproc_s'][str(w)])}</td>"
                          for w in workers)
                + f"<td>{_bar(r['speedup'], r['ideal'], tip)}</td>"
                f"<td class='num'>{_fmt_x(r['speedup'])}</td>"
                f"<td class='num'>{r['efficiency']:.2f}</td>"
                f"<td class='num'>{html.escape(_fallbacks(r))}</td>"
                + ("<td class='ok'>✓ yes</td>" if r["identical"]
                   else "<td class='bad'>✗ differs</td>") + "</tr>")
    machine = "".join(
        f"<tr><td>{html.escape(a)}</td><td>{html.escape(str(b))}</td></tr>" for a, b in [
            ("platform", m["label"]), ("OS", m["os"]), ("CPU", m["cpu"]),
            ("cores used", m["cores"]), ("logical CPUs", m["logical_cpus"]),
            ("RAM", _gib(m["ram_bytes"])), ("Python", m["python"]), ("NumPy", m["numpy"]),
            ("viproc", m["viproc"]), ("build type", m.get("build_type") or "–"),
            ("temporary elision", m["elision"]), ("BLAS threads", m["blas_threads"]),
            ("large size class", f"{m['large_elements']:,} elements"),
            ("timing", f"best of {m['repeat']}, ~{m['target_s'] * 1e3:.0f} ms per NumPy run"),
            ("date", m["date"])])
    explain = html.escape(SCORE_EXPLAINED).replace("**", "")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>viproc benchmark</title><style>{CSS}</style></head>
<body><main>
<h1>viproc benchmark suite</h1>
<p class="sub">{html.escape(m['label'])} · {html.escape(m['cpu'])} · {m['cores']} cores ·
{_gib(m['ram_bytes'])} RAM · NumPy {m['numpy']} · {m['date']}{' · build ' + html.escape(m['build']) if m.get('build') else ''}</p>
{_tiles(run)}
<h2>Scores</h2>
{_score_tables(run['summary'])}
<h2>Results</h2>
{_legend()}
<div class="panel"><table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>
<h2>How the score works</h2>
<p class="explain">{explain}</p>
<h2>Machine</h2>
<div class="panel" style="max-width:640px"><table><tbody>{machine}</tbody></table></div>
</main></body></html>
"""


def write_reports(run, out_dir):
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_markdown(run))
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(report_html(run))


# --- comparison of several runs -------------------------------------------------


def _key(x):
    """Identity of a configuration across runs."""
    return (x["case"], x["size"], x["layout"], x.get("dtype", ""))


def _run_name(run, i):
    m = run["machine"]
    return f"{m['label']} ({m['cores']} cores)" if m.get("label") else f"run {i + 1}"


def comparison_markdown(runs):
    names = [_run_name(r, i) for i, r in enumerate(runs)]
    out = ["# viproc benchmark suite: comparison", "",
           "| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(runs)]

    def row(label, fn):
        out.append(f"| {label} | " + " | ".join(fn(r) for r in runs) + " |")

    row("**score**", lambda r: f"**{_fmt_score(r['summary']['score'])}**")
    row("speedup vs NumPy", lambda r: _fmt_x(r["summary"]["speedup_geomean"]))
    row("speedup, 1 worker", lambda r: _fmt_x(r["summary"]["speedup_w1_geomean"]))
    row("identical", lambda r: f"{r['summary']['identical']}/{r['summary']['configs']}")
    row("CPU", lambda r: r["machine"]["cpu"])
    row("RAM", lambda r: _gib(r["machine"]["ram_bytes"]))
    row("Python / NumPy", lambda r: f"{r['machine']['python']} / {r['machine']['numpy']}")
    cats = []
    for r in runs:
        cats += [c for c in r["summary"]["categories"] if c not in cats]
    for c in _ordered(dict.fromkeys(cats), CATEGORY_ORDER):
        row(c, lambda r, c=c: _fmt_score(r["summary"]["categories"].get(c, float("nan"))))
    out += ["", "## Speedup vs NumPy per configuration", "",
            "| configuration | " + " | ".join(names) + " |", "|---|" + "---:|" * len(runs)]
    keys = []
    for r in runs:
        keys += [_key(x) for x in r["results"]
                 if _key(x) not in keys]
    for k in keys:
        cells = []
        for r in runs:
            hit = [x for x in r["results"] if _key(x) == k]
            cells.append(_fmt_x(hit[0]["speedup"]) if hit else "–")
        out.append(f"| {'/'.join(k)} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n", names, keys


def write_comparison(runs, out_dir):
    md, names, keys = comparison_markdown(runs)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    tiles = '<div class="hero">' + "".join(
        f'<div class="tile main"><div class="label">{html.escape(n)}</div>'
        f'<div class="value">{_fmt_score(r["summary"]["score"])}</div>'
        f'<div class="note">{html.escape(r["machine"]["cpu"])} · '
        f'speedup {_fmt_x(r["summary"]["speedup_geomean"])}</div></div>'
        for n, r in zip(names, runs)) + "</div>"
    head = "<tr><th>configuration</th>" + "".join(
        f"<th>{html.escape(n)}</th><th class='num'></th>" for n in names) + "</tr>"
    body = []
    for k in keys:
        cells = []
        for r in runs:
            hit = [x for x in r["results"] if _key(x) == k]
            if hit:
                x = hit[0]
                tip = f"{html.escape('/'.join(k))}<br>speedup {_fmt_x(x['speedup'])}"
                cells.append(f"<td>{_bar(x['speedup'], x['ideal'], tip)}</td>"
                             f"<td class='num'>{_fmt_x(x['speedup'])}</td>")
            else:
                cells.append("<td></td><td class='num'>–</td>")
        body.append(f"<tr><td>{html.escape('/'.join(k))}</td>{''.join(cells)}</tr>")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>viproc benchmark comparison</title><style>{CSS}</style></head>
<body><main>
<h1>viproc benchmark suite: comparison</h1>
<p class="sub">Parallel efficiency score per run (100 = ideal scaling on that machine).</p>
{tiles}
<h2>Speedup vs NumPy per configuration</h2>
{_legend()}
<div class="panel"><table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>
<h2>How the score works</h2>
<p class="explain">{html.escape(SCORE_EXPLAINED).replace('**', '')}</p>
</main></body></html>
"""
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(page)
