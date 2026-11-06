"""Benchmark history: every recorded suite run, machine-readable, per change.

Each significant change is accompanied by a full suite run that is committed
with it (`suite.py --record "what changed"`). A record is one JSON file in
docs/benchmarks/runs/ (schema `viproc-benchmark-run/1`):

    {
      "schema": "viproc-benchmark-run/1",
      "id": "20261106-1342_7f568bd_O1_m3f2a91c0",
      "recorded": "2026-11-06T13:42:00",
      "change": {"commit": "7f568bd...", "subject": "...", "branch": "...",
                 "dirty": false, "description": "what changed", "pr": 63,
                 "role": "change", "baseline": null},
      "machine": {"id": "m3f2a91c0", "cpu": ..., "cores": ..., "build": ...},
      "options": {"quick": false, "workers": "1,max", "filter": null,
                  "repeat": null, "opt_level": 1},
      "summary": {"score": ..., "categories": {...}, "sizes": {...}, ...},
      "results": [{"case": ..., "size": ..., "layout": ..., "dtype": ...,
                   "numpy_s": ..., "viproc_s": {"1": ..., "4": ...},
                   "speedup": ..., "efficiency": ..., "identical": ...}, ...]
    }

`change.commit` is the code that ran (the record is committed after it).
`change.role` is "change" for the run of a change and "baseline" for a run
of the state before it, measured on the same machine for comparison. Runs
compare only on the same machine id (CPU model, cores, RAM, OS), with the
same options and optimization level: `impact` pairs every change with the
`baseline` record it names, else with the previous comparable run of
another commit and the same number of configurations.
docs/benchmarks/runs/progress-2026-10.json holds the summary rows of the
runs before records existed (no per-case results).

Usage:
    python3 python/benchmarks/history.py list [--machine ID] [--json]
    python3 python/benchmarks/history.py impact [--machine ID] [--json]
    python3 python/benchmarks/history.py compare A B [--top N] [--json]
    python3 python/benchmarks/history.py show ID [--json]
    python3 python/benchmarks/history.py record RESULTS_JSON --change TEXT
        [--commit SHA] [--pr N] [--baseline ID] [--opt N]

A and B are record ids, unique id prefixes or commit prefixes.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys

SCHEMA = "viproc-benchmark-run/1"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(ROOT, "docs", "benchmarks", "runs")
OPTIONS = ("quick", "workers", "filter", "repeat", "opt_level")


def machine_id(m):
    """Short id of a machine: runs compare only within one id."""
    ram_gib = round((m.get("ram_bytes") or 0) / 2**30)
    key = "|".join(str(x) for x in (m.get("label"), m.get("cpu"), m.get("cores"),
                                    m.get("logical_cpus"), ram_gib))
    return "m" + hashlib.sha256(key.encode()).hexdigest()[:8]


def _git(*args):
    try:
        out = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True,
                             timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_change(commit=None):
    """The measured code: commit, subject, branch, uncommitted changes to
    the sources of the build."""
    rev = commit or "HEAD"
    sha = _git("rev-parse", rev)
    # dirty: uncommitted changes to what the build is made of
    status = _git("status", "--porcelain", "--untracked-files=no", "--", "src", "include",
                  "python/viproc", "CMakeLists.txt", "pyproject.toml") if commit is None else ""
    return {
        "commit": sha,
        "subject": _git("log", "-1", "--format=%s", rev),
        "commit_date": _git("log", "-1", "--format=%cI", rev),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") if commit is None else None,
        "dirty": bool(status) if status is not None else None,
    }


def make_record(run, description, change=None, pr=None, baseline=None, options=None,
                role="change"):
    """A record of a suite run (`run`: what suite.py writes to results.json)."""
    machine = dict(run["machine"])
    machine["id"] = machine_id(machine)
    opts = {k: None for k in OPTIONS}
    opts.update(run.get("options") or {})
    opts.update(options or {})
    change = dict(change or git_change())
    change["description"] = description
    change["pr"] = pr
    change["baseline"] = baseline
    change["role"] = role
    when = run["machine"].get("date") or datetime.datetime.now().isoformat(timespec="seconds")
    stamp = datetime.datetime.fromisoformat(when).strftime("%Y%m%d-%H%M")
    sha = (change.get("commit") or "nocommit")[:7] + ("+" if change.get("dirty") else "")
    level = opts.get("opt_level")
    rid = f"{stamp}_{sha}_O{level if level is not None else '?'}_{machine['id']}"
    results = [{k: v for k, v in r.items() if k != "description"} for r in run["results"]]
    return {"schema": SCHEMA, "id": rid, "recorded": when, "change": change,
            "machine": machine, "options": opts, "summary": run["summary"],
            "results": results}


def write_record(record, runs_dir=RUNS_DIR):
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, record["id"].replace("+", "-dirty") + ".json")
    with open(path, "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    return path


def load(runs_dir=RUNS_DIR):
    """All records, oldest first. A file holds one record or {"runs": [...]}."""
    runs = []
    for name in sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []:
        if not name.endswith(".json"):
            continue
        with open(os.path.join(runs_dir, name)) as f:
            data = json.load(f)
        for r in data.get("runs", [data]):
            if r.get("schema") == SCHEMA:
                runs.append(r)
    return sorted(runs, key=lambda r: (r["recorded"], r["id"]))


def find(runs, ref):
    hits = [r for r in runs if r["id"] == ref] or \
           [r for r in runs if r["id"].startswith(ref)] or \
           [r for r in runs if (r["change"].get("commit") or "").startswith(ref)]
    if not hits:
        sys.exit(f"no record matches {ref!r}")
    if len(hits) > 1:
        sys.exit(f"{ref!r} matches {len(hits)} records: " + ", ".join(r["id"] for r in hits))
    return hits[0]


def comparable(a, b):
    return (a["machine"]["id"] == b["machine"]["id"]
            and all(a["options"].get(k) == b["options"].get(k) for k in OPTIONS))


def baseline_of(runs, run):
    """The run a record is judged against: the baseline it names, else the
    latest earlier comparable run of another commit with the same number of
    configurations (a suite that gained cases scores differently)."""
    named = run["change"].get("baseline")
    if named:
        return find(runs, named)
    earlier = [r for r in runs if r["recorded"] < run["recorded"] and comparable(r, run)
               and r["change"].get("commit") != run["change"].get("commit")
               and r["summary"].get("configs") == run["summary"].get("configs")]
    return earlier[-1] if earlier else None


def _config(r):
    return f"{r['case']}/{r['size']}/{r['layout']}/{r['dtype']}"


def compare(a, b):
    """Score and per-configuration differences from run a to run b."""
    sa, sb = a["summary"], b["summary"]
    out = {"from": a["id"], "to": b["id"], "comparable": comparable(a, b),
           "score": [sa["score"], sb["score"]],
           "categories": {c: [sa["categories"].get(c), sb["categories"].get(c)]
                          for c in sorted(set(sa["categories"]) | set(sb["categories"]))},
           "sizes": {s: [sa["sizes"].get(s), sb["sizes"].get(s)]
                     for s in sorted(set(sa["sizes"]) | set(sb["sizes"]))},
           "configs": []}
    ra = {_config(r): r for r in a.get("results") or []}
    for r in b.get("results") or []:
        o = ra.get(_config(r))
        if o is None or not o.get("speedup") or not r.get("speedup"):
            continue
        out["configs"].append({"config": _config(r), "speedup": [o["speedup"], r["speedup"]],
                               "ratio": r["speedup"] / o["speedup"],
                               "fallbacks": [o.get("fallbacks"), r.get("fallbacks")],
                               "identical": r.get("identical")})
    out["configs"].sort(key=lambda c: -abs(math.log(c["ratio"])))
    return out


def _f(v, fmt="{:.0f}"):
    return "" if v is None else fmt.format(v)


def _row(run):
    s, c = run["summary"], run["change"]
    return {"id": run["id"], "date": run["recorded"][:10], "commit": (c.get("commit") or "")[:7],
            "pr": c.get("pr"), "opt": run["options"].get("opt_level"),
            "machine": run["machine"]["id"], "role": c.get("role"), "score": s["score"],
            "w_max": s.get("speedup_geomean"), "w1": s.get("speedup_w1_geomean"),
            "change": c.get("description")}


def cmd_list(runs, args):
    rows = [_row(r) for r in runs if not args.machine or r["machine"]["id"] == args.machine]
    if args.json:
        print(json.dumps(rows, indent=1))
        return
    print(f"{'date':10} {'commit':7} {'O':1} {'machine':9} {'score':>5} {'w=max':>6} "
          f"{'w=1':>5}  change")
    for r in rows:
        print(f"{r['date']:10} {r['commit']:7} {_f(r['opt'], '{}'):1} {r['machine']:9} "
              f"{_f(r['score']):>5} {_f(r['w_max'], '{:.2f}x'):>6} {_f(r['w1'], '{:.2f}x'):>5}"
              f"  {'[baseline] ' if r['role'] == 'baseline' else ''}{r['change'] or ''}")


def cmd_impact(runs, args):
    rows = []
    for run in runs:
        if args.machine and run["machine"]["id"] != args.machine \
                or run["change"].get("role") == "baseline":
            continue
        base = baseline_of(runs, run)
        if base is None:
            continue
        d = compare(base, run)
        rows.append({"id": run["id"], "baseline": base["id"],
                     "commit": (run["change"].get("commit") or "")[:7],
                     "change": run["change"].get("description"),
                     "score": d["score"], "delta": d["score"][1] - d["score"][0],
                     "categories": d["categories"],
                     "top": d["configs"][:args.top]})
    if args.json:
        print(json.dumps(rows, indent=1))
        return
    for r in rows:
        print(f"{r['commit']:7} {r['score'][0]:5.1f} -> {r['score'][1]:5.1f} "
              f"({r['delta']:+5.1f})  {r['change'] or ''}")
        cats = ", ".join(f"{c} {_f(v[0])}->{_f(v[1])}" for c, v in r["categories"].items()
                         if v[0] is not None and v[1] is not None
                         and abs(v[1] - v[0]) >= args.min_points)
        if cats:
            print(f"        categories: {cats}")
        for c in r["top"]:
            if abs(math.log(c["ratio"])) >= math.log(1 + args.min_ratio):
                print(f"        {c['config']:40} {c['speedup'][0]:6.2f}x -> "
                      f"{c['speedup'][1]:6.2f}x")


def cmd_compare(runs, args):
    d = compare(find(runs, args.a), find(runs, args.b))
    if args.json:
        print(json.dumps(d, indent=1))
        return
    if not d["comparable"]:
        print("warning: different machines or options, the numbers do not compare")
    print(f"score {d['score'][0]:.1f} -> {d['score'][1]:.1f}")
    for kind in ("categories", "sizes"):
        print(f"{kind}: " + ", ".join(f"{k} {_f(v[0])}->{_f(v[1])}"
                                     for k, v in d[kind].items()))
    for c in d["configs"][:args.top]:
        print(f"  {c['config']:40} {c['speedup'][0]:6.2f}x -> {c['speedup'][1]:6.2f}x "
              f"({c['ratio']:.2f})")


def cmd_show(runs, args):
    run = find(runs, args.id)
    if args.json:
        print(json.dumps(run, indent=1))
        return
    for k, v in _row(run).items():
        print(f"{k:8} {v}")
    print(f"options  {run['options']}")
    print(f"machine  {run['machine'].get('cpu')}, {run['machine'].get('cores')} cores, "
          f"build {run['machine'].get('build')}")


def cmd_record(runs, args):
    with open(args.results) as f:
        run = json.load(f)
    options = {} if args.opt is None else {"opt_level": args.opt}
    change = git_change(args.commit) if args.commit else None
    rec = make_record(run, args.change, change=change, pr=args.pr, baseline=args.baseline,
                      options=options, role=args.role)
    print(write_record(rec))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "impact"):
        s = sub.add_parser(name)
        s.add_argument("--machine")
        s.add_argument("--json", action="store_true")
        s.add_argument("--top", type=int, default=5)
        s.add_argument("--min-points", type=float, default=3.0,
                       help="impact: category changes shown from this many score points")
        s.add_argument("--min-ratio", type=float, default=0.25,
                       help="impact: configurations shown from this relative change")
    s = sub.add_parser("compare")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("id")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("record")
    s.add_argument("results")
    s.add_argument("--change", required=True, help="what changed (one line)")
    s.add_argument("--commit", help="measured commit, if not HEAD")
    s.add_argument("--pr", type=int)
    s.add_argument("--baseline", help="record id this run is judged against")
    s.add_argument("--opt", type=int, help="optimization level, if results.json lacks it")
    s.add_argument("--role", choices=("change", "baseline"), default="change",
                   help="baseline: the state before a change, measured for comparison")
    args = p.parse_args()
    globals()["cmd_" + args.cmd](load(), args)


if __name__ == "__main__":
    main()
