#!/usr/bin/env bash
# One-click development loop (Linux/macOS): pull, build and install viproc into
# .venv (pip + scikit-build-core, incremental build in build/pip-<tag>), test.
# Usage: ./dev.sh [--no-pull] [--no-test]; PYTHON=python3.13 ./dev.sh picks the
# interpreter for a new .venv. Windows: dev.cmd.
set -euo pipefail
cd "$(dirname "$0")"
pull=1 test=1
for arg in "$@"; do
    case "$arg" in
        --no-pull) pull=0 ;;
        --no-test) test=0 ;;
        *) echo "usage: $0 [--no-pull] [--no-test]" >&2; exit 2 ;;
    esac
done
step() { printf '\n== %s\n' "$1"; }

if [ "$pull" = 1 ]; then step 'git pull'; git pull --ff-only; fi
if [ ! -x .venv/bin/python ]; then
    step 'create .venv'
    "${PYTHON:-python3}" -c 'import sys, struct; sys.exit(0 if sys.version_info >= (3, 12) and struct.calcsize("P") == 8 else 1)' \
        || { echo "${PYTHON:-python3} is not a 64-bit Python >= 3.12; set PYTHON=..." >&2; exit 1; }
    "${PYTHON:-python3}" -m venv .venv
fi
py=.venv/bin/python
step 'install build requirements'
"$py" -m pip install -q --upgrade pip scikit-build-core 'numpy>=2.0' pytest 'cmake>=3.24' ninja
step 'build and install viproc (Release)'
"$py" -m pip install --no-build-isolation --no-deps .
"$py" -c 'import viproc, numpy; print("viproc", viproc.__version__, "numpy", numpy.__version__, viproc.__file__)'
if [ "$test" = 1 ]; then step 'pytest'; "$py" -m pytest -q python/tests; fi
printf '\nDone. Run programs with: .venv/bin/python prog.py\n'
