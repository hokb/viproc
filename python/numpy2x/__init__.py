"""numpy2x: array instruction-level parallelism for NumPy programs.

Wrap your input data with ``numpy2x.asarray`` (or create arrays with the
functions of this module) and keep using NumPy as usual: array ops on numpy2x
arrays are issued asynchronously and run in parallel when their inputs are
ready. Reading data is a sync point.

``import numpy2x as np`` also works: names this module does not define come
from NumPy, and NumPy functions dispatch on numpy2x arrays. ``python -m numpy2x
prog.py`` runs an unchanged NumPy program that way (``import numpy`` in its
own code gives numpy2x); ``with numpy2x.disabled():`` runs code as plain NumPy.
"""

import atexit
import contextlib as _contextlib
import copyreg as _copyreg
import types as _types

import numpy as _np

# numpy2x reproduces NumPy's calling patterns, result layouts and wrappers as
# they are in 2.5; older versions give different results (eig dtypes,
# layouts, reduction blocks), so they are refused rather than run.
if _np.lib.NumpyVersion(_np.__version__) < "2.5.0":
    raise ImportError(f"numpy2x needs NumPy >= 2.5, found {_np.__version__}")

from . import _ndarray, _numpy2x
from ._ndarray import fresh as _fresh
from ._ndarray import (
    AsyncError,
    AsyncIndexError,
    AsyncFloatingPointError,
    AsyncLinAlgError,
    AsyncValueError,
    array,
    asarray,
    init,
    ndarray,
    offload_stats,
    physical_cores,
    pool_stats,
    set_pool_limit,
    shutdown,
    trim,
    wait_all,
    workers,
)

__version__ = _numpy2x.version


def _creator(np_func):
    def create(*args, **kwargs):
        r = np_func(*args, **kwargs)
        if isinstance(r, tuple):  # linspace(retstep=True), indices(sparse=True), ...
            return type(r)(_fresh(x, args, kwargs) for x in r)
        return _fresh(r, args, kwargs)  # fresh memory: adopted without a copy (D7)

    create.__name__ = np_func.__name__
    create.__qualname__ = np_func.__name__
    create.__doc__ = f"Like numpy.{np_func.__name__}, returning a numpy2x array."
    return create


# NumPy functions that create arrays from shapes, values or files. Functions
# of arrays need no wrapper: NumPy dispatches them on numpy2x arguments.
for _name in (
    "zeros", "ones", "empty", "full", "zeros_like", "ones_like", "empty_like", "full_like",
    "arange", "linspace", "logspace", "geomspace", "eye", "identity", "tri", "indices",
    "fromfunction", "fromiter", "frombuffer", "fromfile", "fromstring", "loadtxt", "genfromtxt",
    "load",
):
    globals()[_name] = _creator(getattr(_np, _name))
del _name


class _RandomState:
    """A NumPy Generator / RandomState whose methods return numpy2x arrays."""

    __slots__ = ("_rng",)

    def __init__(self, rng):
        object.__setattr__(self, "_rng", rng)

    def __getattr__(self, name):
        attr = getattr(self._rng, name)
        return _random_function(attr) if callable(attr) else attr

    def __setattr__(self, name, value):
        setattr(self._rng, name, value)

    def __repr__(self):
        return repr(self._rng)


def _random_result(r, args, kwargs):
    if isinstance(r, (_np.random.Generator, _np.random.RandomState)):
        return _RandomState(r) if _ndarray._enabled[0] else r
    return _fresh(r, args, kwargs)


def _random_function(fn):
    def call(*args, **kwargs):
        return _random_result(fn(*args, **kwargs), args, kwargs)

    call.__name__ = getattr(fn, "__name__", "random_function")
    call.__doc__ = getattr(fn, "__doc__", None)
    return call


class _RandomModule(_types.ModuleType):
    """numpy.random for numpy2x: the same functions (and the same global
    state), returning numpy2x arrays; default_rng(), Generator(...) and
    RandomState(...) give generators whose methods do too."""

    def __getattr__(self, name):
        attr = getattr(_np.random, name)
        if name in ("Generator", "RandomState") or (
            callable(attr) and not isinstance(attr, type)
        ):
            wrapped = _random_function(attr)
            setattr(self, name, wrapped)  # once per name
            return wrapped
        return attr

    def __dir__(self):
        return dir(_np.random)


random = _RandomModule("numpy2x.random", _np.random.__doc__)


def enable(on=True):
    """Switch numpy2x on or off for array creation. Off, the creation
    functions (asarray, zeros, numpy2x.random, ... and, under python -m
    numpy2x, numpy.*) return NumPy's own arrays: the same code runs as plain
    NumPy. Existing numpy2x arrays stay numpy2x arrays."""
    _ndarray._enabled[0] = bool(on)


def enabled():
    """True while array creation gives numpy2x arrays (see enable())."""
    return _ndarray._enabled[0]


@_contextlib.contextmanager
def disabled():
    """with numpy2x.disabled(): ... runs the block as plain NumPy (see
    enable()), e.g. to time the same code with and without numpy2x."""
    previous = _ndarray._enabled[0]
    _ndarray._enabled[0] = False
    try:
        yield
    finally:
        _ndarray._enabled[0] = previous


# Element-wise ufuncs, reductions and np.where without NumPy's dispatch when called
# through this module (`import numpy2x as np`, python -m numpy2x): a numpy2x
# argument takes the C hot path right away; everything else is NumPy's own
# function (attributes, other arguments, `isinstance(np.add, np.ufunc)`).
for _name in dir(_np):
    _f = getattr(_np, _name)
    if isinstance(_f, _np.ufunc) and _f.signature is None and _f.nout == 1:
        globals()[_name] = _numpy2x.ufunc_proxy(_f)
where = _numpy2x.where
# The reductions the C path takes (setup_reductions), the same way.
for _name in ("sum", "prod", "max", "min", "amax", "amin", "any", "all", "mean", "var", "std",
              "argmax", "argmin"):
    globals()[_name] = _numpy2x.ufunc_proxy(getattr(_np, _name), True)
del _name, _f
# Pickled as NumPy's ufunc: unpickling gives numpy.<name>, no numpy2x needed.
_copyreg.pickle(_numpy2x.ufunc_proxy,
                lambda p: (__import__("pydoc").locate, ("numpy." + p._ufunc.__name__,)))


def __getattr__(name):
    return getattr(_np, name)


atexit.register(shutdown)
