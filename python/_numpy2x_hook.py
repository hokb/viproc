"""Makes ``import numpy`` in user code import numpy2x.

Used by ``python -m numpy2x prog.py`` and, when numpy2x is pip-installed, by
``numpy2x.pth`` at every interpreter start if the environment variable
``NUMPY2X=1`` is set. Only the user's own code gets numpy2x: modules of the
standard library and of installed packages (site-packages, dist-packages),
NumPy and numpy2x themselves keep the real NumPy, so libraries that use NumPy's
internals keep working; numpy2x arrays they receive go through NumPy's array
protocols.

Light on purpose (site imports it at every start): NumPy and numpy2x are
imported only on the first ``import numpy`` of user code.
"""

import builtins
import os
import site
import sys
import sysconfig

_original_import = None
_library_dirs = None
_user_files = {}  # __file__ -> True for user code


def _libraries():
    global _library_dirs
    if _library_dirs is None:
        dirs = set()
        paths = sysconfig.get_paths()
        for key in ("stdlib", "platstdlib", "purelib", "platlib"):
            if paths.get(key):
                dirs.add(paths[key])
        try:
            dirs.update(site.getsitepackages())
            dirs.add(site.getusersitepackages())
        except AttributeError:  # some virtualenv builds lack these
            pass
        dirs.update(p for p in sys.path if os.path.basename(p) in ("site-packages", "dist-packages"))
        _library_dirs = tuple(
            os.path.join(os.path.normcase(os.path.realpath(d)), "") for d in dirs if d
        )
    return _library_dirs


def is_user_code(globals_):
    """True if the module with these globals is the user's code: a file
    outside the standard library and installed packages, or code without a
    file run as ``__main__`` (``-c``, the interactive console)."""
    if not globals_:
        return False
    name = globals_.get("__name__") or ""
    if name.partition(".")[0] in ("numpy", "numpy2x", "_numpy2x_hook"):
        return False
    path = globals_.get("__file__")
    if not path:
        return name == "__main__"
    user = _user_files.get(path)
    if user is None:
        real = os.path.normcase(os.path.realpath(path))
        user = not real.startswith(_libraries())
        _user_files[path] = user
    return user


def _numpy2x_counterpart(name):
    """numpy2x's object for ``numpy`` or ``numpy.<sub>``: the numpy2x module,
    its own submodule replacement (numpy2x.random) or NumPy's submodule."""
    numpy2x = _original_import("numpy2x")
    obj = numpy2x
    for part in name.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if (
        level == 0
        and (name == "numpy" or name.startswith("numpy."))
        and is_user_code(globals)
    ):
        if name != "numpy":
            _original_import(name, globals, locals, fromlist, level)  # load NumPy's submodule
        # `import numpy.linalg` binds the top-level name; `from numpy.random
        # import x` reads x from the submodule itself.
        return _numpy2x_counterpart(name if fromlist else "numpy")
    return _original_import(name, globals, locals, fromlist, level)


def install():
    """Route ``import numpy`` of user code to numpy2x (idempotent)."""
    global _original_import
    if _original_import is None:
        _original_import = builtins.__import__
        builtins.__import__ = _import


def autoload():
    """Called by numpy2x.pth at interpreter start: install() if NUMPY2X=1."""
    if os.environ.get("NUMPY2X", "") == "1":
        install()
