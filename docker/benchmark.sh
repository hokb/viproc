#!/bin/bash
# Runs the viproc benchmark suite (python/benchmarks/suite.py) in the Linux
# image and in the Windows build (Wine image), then merges both into one
# comparison report. Builds missing images first.
#
#   docker/benchmark.sh [--quick] [SUITE_ARGS...]
#
# Results: viproc-benchmark/<platform>/report.{html,md}, results.json and the
# comparison in viproc-benchmark/report.{html,md} (relative to the current
# directory). Images: VIPROC_IMAGES="linux wine" (default).
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
out=${VIPROC_BENCH_OUT:-viproc-benchmark}
images=${VIPROC_IMAGES:-linux wine}
mkdir -p "$out"

results=()
for img in $images; do
    if ! docker image inspect "viproc:$img" >/dev/null 2>&1; then
        echo "== building viproc:$img"
        docker build -f "$repo/docker/$img/Dockerfile" -t "viproc:$img" "$repo"
    fi
    echo "== benchmark suite on viproc:$img"
    docker run --rm -v "$(cd "$out" && pwd):/work" "viproc:$img" \
        benchmark suite --out "/work/$img" "$@"
    results+=("/work/$img/results.json")
done

if [[ ${#results[@]} -gt 1 ]]; then
    docker run --rm -v "$(cd "$out" && pwd):/work" viproc:linux \
        python /src/python/benchmarks/suite.py --merge "${results[@]}" --out /work
fi
echo "== reports in $out/ (report.html: comparison; <platform>/report.html: details)"
