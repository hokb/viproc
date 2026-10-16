# Docker images

Three images build viproc from this checkout, run all test suites by
default, and double as a command line for viproc programs.

| Image | Dockerfile | Host | Build |
|-------|------------|------|-------|
| `viproc:linux` | `docker/linux/Dockerfile` | Linux (or Docker Desktop, Linux containers) | GCC, CPython from `python:<ver>-bookworm` |
| `viproc:windows` | `docker/windows/Dockerfile` | Windows with Windows containers | MSVC (Visual Studio Build Tools), Windows CPython (NuGet) |
| `viproc:wine` | `docker/wine/Dockerfile` | Linux | Windows x64 build (llvm-mingw, UCRT) run under Wine 10 with the Windows CPython (NuGet) and NumPy's Windows wheels |

The Wine image runs the Windows binaries on a Linux host, so the Windows
code paths (stack capture with `RtlCaptureStackBackTrace`, `VirtualAlloc`,
the Windows CPython and NumPy) are tested in Linux CI as well. It uses
the Universal CRT from the Windows SDK, because Wine's own `ucrtbase.dll`
lacks functions NumPy needs. The native Windows image is the reference for
MSVC.

All suites (`ctest`: `version`, `numpy_loops`, `runtime`,
`numpy_runtime`, `python`) pass in every image with the same results.
Temporary elision (design doc D5) is active everywhere: mode `backtrace`
on CPython 3.12/3.13, `exact` on 3.14.

## Build

From the repository root (the build context is the checkout):

```bash
docker build -f docker/linux/Dockerfile -t viproc:linux .
docker build -f docker/wine/Dockerfile  -t viproc:wine  .
docker build -f docker/windows/Dockerfile -t viproc:windows .   # on Windows
```

Build arguments:

| Argument | Images | Default |
|----------|--------|---------|
| `PYTHON_VERSION` | all | `3.13` (Linux), `3.13.16` (Windows, Wine: a NuGet version) |
| `NUMPY_VERSION` | all | `2.5.4` |
| `BUILD_TYPE` | all | `Release` |
| `WINDOWS_VERSION` | windows | `ltsc2022`; must suit the host (`ltsc2025` for Windows Server 2025) |
| `VS_VERSION` | windows | `17` (Visual Studio 2022 Build Tools) |
| `LLVM_MINGW_VERSION`, `WINDOWS_SDK_VERSION` | wine | pinned releases |

Behind a TLS-inspecting proxy, pass its CA certificate to the Linux and
Wine builds as a BuildKit secret (used for downloads only, not stored in
the image): `--secret id=ca,src=/path/to/ca.pem`.

## Benchmark suite

One call runs the whole suite (python/benchmarks/suite.py) on Linux and on
the Windows build (Wine image), first (re)builds both images from this
checkout (Docker's cache makes that quick when nothing changed;
`VIPROC_NO_BUILD=1` skips it), and writes a comparison. Old report files
in the output directory are deleted before the new ones are copied, so a
report there is always from this run; its header names the build
(`build <time> #<fingerprint>`). `docker run IMAGE` without a command
runs the test suites only, not the benchmark.

```bash
docker/benchmark.sh            # Linux, macOS, WSL; add --quick for a short run
docker\benchmark.ps1           # Windows, Docker Desktop (Linux containers)
```

The reports are copied out of the containers (`docker cp`, no bind mount
needed, also with a remote Docker daemon) into `reports/` in the current
directory: `reports/report.html` (comparison of the platforms) and
`reports/<image>/report.html` (details; also `report.md` and
`results.json`). `VIPROC_BENCH_OUT` picks another directory.

A single image writes to `/work/reports/<platform>` (`linux`,
`windows-wine`, `windows`). Mount the current directory on `/work` to get
them into `./reports/<platform>`; the syntax for the current directory
depends on the shell:

```
docker run --rm -v "$PWD:/work" viproc:linux benchmark      # bash, zsh
docker run --rm -v "${PWD}:/work" viproc:linux benchmark    # PowerShell
docker run --rm -v "%cd%:/work" viproc:linux benchmark      # cmd.exe
```

Without a mount the reports stay inside the container; the command then
prints the `docker cp` line with this container's ID. That works only
while the container exists: `--rm` deletes it when the command ends, so
run it without `--rm` (or with `--name bench`, then `docker cp
bench:/work/reports/. ./reports`, and `docker rm bench`). Mind the `/.`:
without it, an existing `./reports` gets the copy one level deeper
(`./reports/reports/...`) and the old report stays where it was. Each
report shows the build it measured (`build` in its header: build time of
the extension module and a fingerprint of the package). Native Windows
image: `docker run --rm -v "${PWD}:C:\work" viproc:windows benchmark`
(writes `.\reports\windows`), then merge its `results.json` with others:
`suite.py --merge A/results.json B/results.json --out DIR`.

The suite runs ~100 configurations (element-wise chains, basic and advanced
indexing, reductions, algorithms such as k-means, Black-Scholes, Mandelbrot,
n-body; scalar to large data; vector, row/column-major, strided, (n,1)/(1,n)
layouts; float64, float32, int32, int16, complex128, float16 and mixed
types) as plain NumPy and on viproc (1 worker, one worker per core), checks
that every result is bit-identical, and reports the **parallel efficiency
score** (0-100, normalized to the machine's physical cores and RAM; how it is
computed is in the report). Full run: ~4 minutes per platform on 4 cores
(wider machines get wider programs, so somewhat longer); `--quick`: about
half. `--filter REGEX` selects configurations, `--workers 1,2,4,max` adds
scaling columns (`max` = physical cores, `threads` = hardware threads).

## Command line

```
docker run [--rm -it] [-v DIR:/work] IMAGE [COMMAND [ARGS...]]
```

| Command | Effect |
|---------|--------|
| `test [CTEST_ARGS...]` | all test suites with `ctest` (default) |
| `pytest [PYTEST_ARGS...]` | the Python bridge tests |
| `python [ARGS...]` | Python with viproc importable; interactive with `-it` |
| `benchmark [NAME] [ARGS]` | `python/benchmarks/NAME.py`; default `suite`, the benchmark suite (below), reports to `/work/reports/<platform>` |
| `build` | bring the build up to date |
| `info` | platform, Python, NumPy, build type, temporary elision mode |
| `help` | usage |
| anything else | executed as a command (`bash`; `cmd`/`powershell` on Windows) |

The working directory is `/work` (`C:\work` on Windows): mount a directory
there to run your own programs.

```bash
docker run --rm viproc:linux                                  # all tests
docker run --rm viproc:wine pytest -q -k elision              # some bridge tests
docker run --rm -it viproc:linux python                       # REPL with viproc
docker run --rm -v "$PWD:/work" viproc:wine python prog.py    # a program, on Windows
docker run --rm viproc:linux benchmark strong_scaling --workers 1,2,4,8
```

```powershell
docker run --rm viproc:windows
docker run --rm -v "${PWD}:C:\work" viproc:windows python prog.py
```

**Testing a checkout without rebuilding the image.** The sources live in
`/src` (`C:\src`) and the configured build in `/build` (`C:\build`). Every
command first brings the build up to date, so mounting a checkout over the
sources rebuilds only what changed:

```bash
docker run --rm -v "$PWD:/src" viproc:linux
docker run --rm -v "$PWD:/src" viproc:wine
```

Notes:

- Windows checkouts: `.gitattributes` keeps shell scripts, Dockerfiles
  and CMake files at LF line endings even with `core.autocrlf=true`, and
  the images install their command line with CRLF stripped. A checkout
  made before that fix still builds working images; to normalize it
  anyway: `git rm --cached -r . && git reset --hard` (discards local
  changes) or a fresh clone.

- Bytecode caches go to `PYTHONPYCACHEPREFIX` (`/tmp/pycache`), not into
  `__pycache__` of the sources: caches a mounted checkout brings from the
  host name the host's paths.
- The Wine image runs as root (the Wine prefix belongs to root). Wine
  prints nothing extra (`WINEDEBUG=-all`); the first start of a container
  takes a few seconds for `wineserver`.
- Thread counts: `OPENBLAS_NUM_THREADS=1` is set, so BLAS does not compete
  with viproc's workers (CLAUDE.md, "Linalg").
- The Windows image cannot be built or run on a Linux host: Windows
  containers need a Windows kernel. Use the Wine image there.
