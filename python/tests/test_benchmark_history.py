"""The benchmark history (python/benchmarks/history.py): records of suite
runs, machine-readable, compared per change."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import history  # noqa: E402

MACHINE = {"label": "linux", "cpu": "Test CPU", "cores": 4, "logical_cpus": 8,
           "ram_bytes": 16 << 30, "build": "x", "date": None}


def _run(date, speedups, level=1):
    results = [{"case": c, "category": "chains", "description": "d", "size": "medium",
                "layout": "vec", "dtype": "float64", "speedup": s, "efficiency": s / 4,
                "identical": True, "fallbacks": 0} for c, s in speedups.items()]
    score = 100.0 * sum(r["efficiency"] for r in results) / len(results)
    return {"machine": dict(MACHINE, date=date),
            "options": {"quick": False, "workers": "1,max", "filter": None, "repeat": None,
                        "opt_level": level},
            "results": results,
            "summary": {"score": score, "categories": {"chains": score},
                        "sizes": {"medium": score}, "configs": len(results)}}


def _change(sha):
    return {"commit": sha * 40, "subject": "s", "commit_date": None, "branch": "b",
            "dirty": False}


def test_records_round_trip_and_impact(tmp_path):
    base = history.make_record(_run("2026-11-01T10:00:00", {"a": 2.0, "b": 1.0}),
                               "before", change=_change("1"), role="baseline")
    other = history.make_record(_run("2026-11-01T11:00:00", {"a": 9.0, "b": 9.0}, level=2),
                                "other level", change=_change("3"))
    new = history.make_record(_run("2026-11-01T12:00:00", {"a": 3.0, "b": 1.0}),
                              "faster a", change=_change("2"), pr=7)
    for r in (base, other, new):
        history.write_record(r, str(tmp_path))
    runs = history.load(str(tmp_path))
    assert [r["id"] for r in runs] == [base["id"], other["id"], new["id"]]
    assert new["id"].startswith("20261101-1200_2222222_O1_m")
    assert "description" not in runs[2]["results"][0]
    assert runs[2]["change"]["pr"] == 7 and runs[2]["machine"]["id"] == base["machine"]["id"]
    # The O2 run does not compare with O1 runs; the baseline does.
    assert history.baseline_of(runs, runs[2])["id"] == base["id"]
    assert history.baseline_of(runs, runs[1]) is None
    d = history.compare(runs[0], runs[2])
    assert d["comparable"] and d["configs"][0]["config"] == "a/medium/vec/float64"
    assert d["configs"][0]["ratio"] == 1.5
    assert history.find(runs, "2222")["id"] == new["id"]


def test_committed_history_loads():
    runs = history.load()
    assert runs, "docs/benchmarks/runs holds no records"
    for r in runs:
        assert r["schema"] == history.SCHEMA
        assert {"commit", "description", "role"} <= set(r["change"])
        assert r["machine"]["id"] and "score" in r["summary"]
