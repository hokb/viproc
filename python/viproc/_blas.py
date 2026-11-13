"""Thread count of the BLAS library NumPy uses (OpenBLAS in NumPy's wheels).

Dynamic by default (docs/design/goals.md, "BLAS threads"): a BLAS call that
runs alone, with no other BLAS call in flight and no other active task, gets
the thread count found at init, as in plain NumPy (OpenBLAS parallelizes a
single large product itself). As soon as there is more asynchronous work,
calls run on one thread and viproc's workers parallelize across ops instead:
OpenBLAS's own threads on top of the workers oversubscribe the cores (8
concurrent 300 x 300 `inv` took 20 ms to 1.1 s, NumPy ~22 ms; see CLAUDE.md,
"BLAS threads decide BLAS results"). The task making the call decides it
(src/kernels/blas_threads.cpp), on its thread; OpenBLAS's count is
process-wide, so it is lowered at any time (a running call keeps its split)
and raised only by the only call in flight.

OpenBLAS splits a product by its thread count, and the values depend on that
split: a large product computed with one thread differs in the last bits from
one computed with several. Bit-identity therefore holds per setup (thread
count), as it does for NumPy across machines with different core counts.
`VIPROC_BLAS_THREADS=1`, and the deterministic mode (`adaptive=False`), fix
one thread for every call instead: results are NumPy's with
`OPENBLAS_NUM_THREADS=1`, reproducible whatever runs concurrently.

The count found before is restored when the runtime shuts down. No BLAS
library found: nothing changes, and large LAPACK calls stay serialized
(`_linalg._call`).
"""

import ctypes
import glob
import os
import sys

import numpy as np

_funcs = None  # (set, get) once looked up; False: no BLAS library found
_saved = None  # thread count before viproc took control, while it has it
_mode = None  # "dynamic" or "single" while viproc controls the count


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


def controlled():
    """True while viproc controls the BLAS thread count (`apply`), no call.
    Then concurrent BLAS calls never run threaded side by side."""
    return _mode is not None


def single():
    """True while viproc runs every BLAS call on one thread (fixed mode)."""
    return _mode == "single"


def mode():
    """"dynamic", "single", or None (no control)."""
    return _mode


def _env_mode(adaptive):
    env = os.environ.get("VIPROC_BLAS_THREADS", "dynamic").strip().lower()
    if env in ("1", "single", "one"):
        return "single"
    if env not in ("", "dynamic", "auto"):
        raise ValueError(f"VIPROC_BLAS_THREADS must be 'dynamic' or '1', got {env!r}")
    return "dynamic" if adaptive else "single"


def apply(adaptive=True):
    """Hands the BLAS thread count to the runtime (after `_viproc.init`)."""
    global _saved, _mode
    from . import _viproc

    funcs = _get_funcs()
    if not funcs:
        return
    set_, get = funcs
    if _saved is None:
        _saved = get()
    _mode = _env_mode(adaptive)
    if _mode == "dynamic" and _saved <= 1:
        _mode = "single"  # nothing to switch: OpenBLAS runs one thread anyway
    addr = ctypes.cast(set_, ctypes.c_void_p).value
    _viproc.blas_threads(addr, _saved, _mode == "dynamic")


def stats():
    """(count viproc set last or 0, number of changes): diagnostics."""
    from . import _viproc

    return _viproc.blas_stats()


def restore():
    """The thread count from before `apply` (runtime shut down)."""
    global _saved, _mode
    funcs = _get_funcs()
    if funcs and _saved is not None:
        funcs[0](_saved)
        _saved = None
    _mode = None
