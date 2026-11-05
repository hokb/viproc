"""Thread count of the BLAS library NumPy uses (OpenBLAS in NumPy's wheels).

Optimization level O2 (docs/design/goals.md) runs every BLAS call on one
thread: viproc's workers parallelize across ops, and OpenBLAS's own threads
on top of them oversubscribe the cores (8 concurrent 300 x 300 `inv` took
20 ms to 1.1 s, NumPy ~22 ms; see CLAUDE.md, "BLAS threads decide BLAS
results"). O1 must not do this: OpenBLAS splits a product by its thread
count, and the values depend on that split.

OpenBLAS has no per-thread setting, so the count is process-wide: plain
NumPy calls in the same process run single-threaded too while O2 is on.
The count found before is restored when the runtime shuts down or a level
below 2 is chosen. No BLAS library found: nothing changes.
"""

import ctypes
import glob
import os
import sys

import numpy as np

_funcs = None  # (set, get) once looked up; False: no BLAS library found
_saved = None  # thread count before O2 took over, while it is in effect


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


def apply_level(level):
    """One BLAS thread from level 2 on; the previous count below it."""
    global _saved
    funcs = _get_funcs()
    if not funcs:
        return
    set_, get = funcs
    if level >= 2:
        if _saved is None:
            _saved = get()
        set_(1)
    elif _saved is not None:
        set_(_saved)
        _saved = None
