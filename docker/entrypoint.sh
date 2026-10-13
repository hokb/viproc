#!/bin/bash
# Command line of the viproc Linux and Wine images (docker/README.md).
#
# The image holds the sources in $VIPROC_SRC and a configured build in
# $VIPROC_BUILD. Every command first brings the build up to date, so a
# checkout mounted over $VIPROC_SRC (docker run -v "$PWD:/src" ...) is
# rebuilt incrementally before it runs.
set -euo pipefail

usage() {
    cat <<EOF
usage: docker run [--rm -it] [-v DIR:/work -w /work] IMAGE [COMMAND [ARGS...]]

commands:
  test [CTEST_ARGS...]     build, then run all test suites with ctest (default)
  pytest [PYTEST_ARGS...]  build, then run the Python bridge tests
  python [ARGS...]         Python with viproc importable (interactive with -it)
  benchmark [NAME [ARGS]]  run python/benchmarks/NAME.py (default: suite, the full
                           benchmark suite; reports go to ./viproc-benchmark)
  build                    bring the build up to date
  info                     platform, Python, NumPy and viproc configuration
  help                     this text
  anything else            executed as a command (e.g. bash)

image: $VIPROC_PLATFORM, build type $(build_type), sources $VIPROC_SRC, build $VIPROC_BUILD
EOF
}

build_type() {
    sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' "$VIPROC_BUILD/CMakeCache.txt" 2>/dev/null || true
}

build() {
    local log
    log=$(mktemp)
    if ! cmake --build "$VIPROC_BUILD" >"$log" 2>&1; then
        cat "$log" >&2
        rm -f "$log"
        exit 1
    fi
    rm -f "$log"
}

cmd=${1:-test}
[[ $# -gt 0 ]] && shift

case "$cmd" in
test)
    build
    exec ctest --test-dir "$VIPROC_BUILD" --output-on-failure "$@"
    ;;
pytest)
    build
    exec python -m pytest "$VIPROC_SRC/python/tests" "$@"
    ;;
python)
    build
    exec python "$@"
    ;;
benchmark)
    build
    name=suite # "benchmark --quick" runs the suite with --quick
    if [[ $# -gt 0 && $1 != -* ]]; then
        name=$1
        shift
    fi
    exec python "$VIPROC_SRC/python/benchmarks/$name.py" "$@"
    ;;
build)
    build
    ;;
info)
    build
    echo "image: $VIPROC_PLATFORM, build type $(build_type)"
    exec python -c '
import platform, sys
import numpy, viproc
print("platform:", platform.platform())
print("python:", sys.version.split()[0], "numpy:", numpy.__version__)
print("temporary elision:", viproc._viproc.elision_mode())'
    ;;
help | -h | --help)
    usage
    ;;
*)
    exec "$cmd" "$@"
    ;;
esac
