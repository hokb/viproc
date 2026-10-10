"""viproc: array instruction-level parallelism for NumPy programs.

Wrap your input data with ``viproc.asarray`` (or create arrays with the
functions of this module) and keep using NumPy as usual: array ops on viproc
arrays are issued asynchronously and run in parallel when their inputs are
ready. Reading data is a sync point.

``import viproc as np`` also works: names this module does not define come
from NumPy, and NumPy functions dispatch on viproc arrays.
"""

import atexit

import numpy as _np

from . import _viproc
from ._ndarray import (
    AsyncError,
    array,
    asarray,
    init,
    ndarray,
    offload_stats,
    shutdown,
    wait_all,
    workers,
)

__version__ = _viproc.version


def _creator(np_func):
    def create(*args, **kwargs):
        return asarray(np_func(*args, **kwargs))

    create.__name__ = np_func.__name__
    create.__doc__ = f"Like numpy.{np_func.__name__}, returning a viproc array."
    return create


zeros = _creator(_np.zeros)
ones = _creator(_np.ones)
empty = _creator(_np.empty)
full = _creator(_np.full)
arange = _creator(_np.arange)
linspace = _creator(_np.linspace)
eye = _creator(_np.eye)
zeros_like = _creator(_np.zeros_like)
ones_like = _creator(_np.ones_like)
empty_like = _creator(_np.empty_like)
full_like = _creator(_np.full_like)


def __getattr__(name):
    return getattr(_np, name)


atexit.register(shutdown)
