"""Thread count of the BLAS library NumPy uses (OpenBLAS in NumPy's wheels).

viproc runs every BLAS call on one thread, at every optimization level
(docs/design/goals.md): viproc's workers parallelize across ops, and
OpenBLAS's own threads on top of them oversubscribe the cores (8 concurrent
300 x 300 `inv` took 20 ms to 1.1 s, NumPy ~22 ms; see CLAUDE.md, "BLAS
threads decide BLAS results"). With one BLAS thread, LAPACK calls need no
serialization either (`_linalg._call`).

OpenBLAS splits a product by its thread count, and the values depend on
that split. Results are therefore those of NumPy with one BLAS thread
(`OPENBLAS_NUM_THREADS=1`). OpenBLAS has no per-thread setting, so the
count is process-wide: plain NumPy calls in the same process run
single-threaded too while the runtime runs, and give the same values. The
count found before is restored when the runtime shuts down. No BLAS library
found: nothing changes, and large LAPACK calls stay serialized.
"""

import ctypes
import glob
import os
import sys

import numpy as np

_funcs = None  # (set, get) once looked up; False: no BLAS library found
_saved = None  # thread count before viproc set one thread, while that is in effect


def _candidates():
    here = os.path.dirname(os.path.abspath(np.__file__))
    for d in (os.path.join(here, os.pardir, "numpy.libs"), os.path.join(here, ".dylibs")):
        yield from sorted(glob.glob(os.path.join(d, "*openblas*")))
    if sys.platform.startswith("linux"):  # NumPy built against a system OpenBLAS
        try:
            with open("/proc/self/maps") as f:
                seen = set()
                for line in f:
                    path = line.split()[-1]
                    if "openblas" in os.path.basename(path) and path not in seen:
                        seen.add(path)
                        yield path
        except OSError:
            pass


def _lookup():
    for path in _candidates():
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for prefix in ("scipy_openblas_", "openblas_"):
            for suffix in ("64_", "_64_", ""):
                try:
                    set_ = getattr(lib, f"{prefix}set_num_threads{suffix}")
                    get = getattr(lib, f"{prefix}get_num_threads{suffix}")
                except AttributeError:
                    continue
                set_.argtypes, set_.restype = [ctypes.c_int], None
                get.argtypes, get.restype = [], ctypes.c_int
                return set_, get
    return False


def _get_funcs():
    global _funcs
    if _funcs is None:
        _funcs = _lookup()
    return _funcs


def threads():
    """The BLAS library's current thread count, or None without one."""
    funcs = _get_funcs()
    return funcs[1]() if funcs else None


def single():
    """True while viproc runs BLAS on one thread (`apply`), no call."""
    return _saved is not None


def apply():
    """One BLAS thread while the runtime runs (every optimization level)."""
    global _saved
    funcs = _get_funcs()
    if not funcs:
        return
    set_, get = funcs
    if _saved is None:
        _saved = get()
    set_(1)


def restore():
    """The thread count from before `apply` (runtime shut down)."""
    global _saved
    funcs = _get_funcs()
    if funcs and _saved is not None:
        funcs[0](_saved)
        _saved = None
