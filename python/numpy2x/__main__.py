"""Run a NumPy program with numpy2x, unchanged:

    python -m numpy2x prog.py [args]      a script
    python -m numpy2x -m module [args]    a module, as python -m
    python -m numpy2x -c "code" [args]    a command, as python -c
    python -m numpy2x                     interactive console

``import numpy`` in the program's own code imports numpy2x instead; the
standard library and installed packages keep the real NumPy (_numpy2x_hook).
"""

import builtins
import os
import runpy
import sys

import _numpy2x_hook

USAGE = __doc__.split("\n\n")[0] + "\n" + __doc__.split("\n\n")[1]


def main(argv):
    _numpy2x_hook.install()
    if not argv:
        import code

        sys.argv = [""]
        code.interact(banner=f"numpy2x {sys.modules['numpy2x'].__version__}: numpy is numpy2x",
                      local={"__name__": "__main__", "__builtins__": builtins})
        return
    if argv[0] in ("-h", "--help"):
        print(USAGE)
        return
    if argv[0] == "-m":
        if len(argv) < 2:
            sys.exit("python -m numpy2x -m: module name missing")
        sys.argv = [argv[1]] + argv[2:]  # runpy replaces argv[0] with the module's path
        runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
        return
    if argv[0] == "-c":
        if len(argv) < 2:
            sys.exit("python -m numpy2x -c: command missing")
        sys.argv = ["-c"] + argv[2:]
        main_globals = {"__name__": "__main__", "__builtins__": builtins}
        exec(compile(argv[1], "<string>", "exec"), main_globals)
        return
    sys.argv = argv
    sys.path[0] = os.path.dirname(os.path.abspath(argv[0]))  # as python prog.py
    runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
