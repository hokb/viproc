#!/bin/bash
# Runs the viproc benchmark suite (python/benchmarks/suite.py) in the Linux
# image and in the Windows build (Wine image), then merges both into one
# comparison report. Builds the images from this checkout first (Docker's
# cache makes that quick when nothing changed; VIPROC_NO_BUILD=1 skips it).
#
#   docker/benchmark.sh [--quick] [SUITE_ARGS...]
#
# The reports are copied out of the containers (docker cp, no bind mount,
# so this also works with a remote Docker daemon) into ./reports, relative
# to the current directory:
#   reports/report.{html,md}               comparison of the platforms
#   reports/<image>/report.{html,md}       details, and results.json
# Output directory: VIPROC_BENCH_OUT (default reports). Images:
# VIPROC_IMAGES="linux wine" (default).
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
out=${VIPROC_BENCH_OUT:-reports}
images=${VIPROC_IMAGES:-linux wine}
mkdir -p "$out"

containers=()
cleanup() {
    for c in "${containers[@]}"; do
        docker rm -f "$c" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT

# Runs IMAGE ARGS... in a new container, then copies its /work/reports/SUB
# to $out/SUB. Extra inputs: COPY_IN (a host directory copied to /work/in).
run_and_copy() {
    local sub=$1 img=$2
    shift 2
    local cid
    cid=$(docker create -e VIPROC_REPORTS_COPIED=1 "$img" "$@")
    containers+=("$cid")
    if [[ -n ${COPY_IN:-} ]]; then
        docker cp "$COPY_IN/." "$cid:/work/in"
    fi
    docker start -a "$cid"
    mkdir -p "$out/$sub"
    # No stale report may survive a run that does not write one.
    rm -f "$out/$sub/report.html" "$out/$sub/report.md" "$out/$sub/results.json"
    docker cp "$cid:/work/reports/$sub/." "$out/$sub"
}

results=()
for img in $images; do
    if [[ -z ${VIPROC_NO_BUILD:-} ]]; then
        echo "== building viproc:$img (cached when unchanged)"
        docker build -f "$repo/docker/$img/Dockerfile" -t "viproc:$img" "$repo"
    fi
    echo "== benchmark suite on viproc:$img"
    run_and_copy "$img" "viproc:$img" benchmark suite --out "/work/reports/$img" "$@"
    results+=("/work/in/$img/results.json")
done

if [[ ${#results[@]} -gt 1 ]]; then
    echo "== comparison"
    COPY_IN=$out run_and_copy . viproc:linux \
        python /src/python/benchmarks/suite.py --merge "${results[@]}" --out /work/reports
fi
echo "== reports in $(cd "$out" && pwd)/ (report.html: comparison; <image>/report.html: details)"
