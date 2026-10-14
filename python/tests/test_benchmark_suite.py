"""Smoke test of the benchmark suite (python/benchmarks/suite.py): a few
configurations run end to end, results match NumPy, reports are written."""

import json
import os
import subprocess
import sys

SUITE = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "suite.py")


def test_suite_runs_and_writes_reports(tmp_path):
    out = tmp_path / "bench"
    proc = subprocess.run(
        [sys.executable, SUITE, "--quick", "--repeat", "1", "--workers", "1,2",
         "--filter", r"chain_serial/tiny|dtype_chains/small/vec/(int16|complex128)|"
                     r"gather/small|scalar_newton",
         "--out", str(out)],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    run = json.loads((out / "results.json").read_text())
    assert run["summary"]["configs"] == 5
    assert run["summary"]["identical"] == 5
    assert 0 < run["summary"]["score"]
    by = {(r["case"], r["dtype"]): r for r in run["results"]}
    assert by[("dtype_chains", "int16")]["not_in_viproc"] == ["int16"]
    assert by[("gather", "float64")]["fallbacks"] == 0  # gathers run as tasks
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "Parallel efficiency score" in html and "gather" in html
    assert "Parallel efficiency score" in (out / "report.md").read_text(encoding="utf-8")

    merged = tmp_path / "merged"
    proc = subprocess.run(
        [sys.executable, SUITE, "--merge", str(out / "results.json"),
         str(out / "results.json"), "--out", str(merged)],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "comparison" in (merged / "report.md").read_text(encoding="utf-8")
