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

## Command line

```
docker run [--rm -it] [-v DIR:/work] IMAGE [COMMAND [ARGS...]]
```

| Command | Effect |
|---------|--------|
| `test [CTEST_ARGS...]` | all test suites with `ctest` (default) |
| `pytest [PYTEST_ARGS...]` | the Python bridge tests |
| `python [ARGS...]` | Python with viproc importable; interactive with `-it` |
| `benchmark [NAME [ARGS]]` | `python/benchmarks/NAME.py` (default `strong_scaling`) |
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
docker run --rm viproc:linux benchmark strong_scaling
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
