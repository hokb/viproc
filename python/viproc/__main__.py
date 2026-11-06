"""Run a NumPy program with viproc, unchanged:

    python -m viproc prog.py [args]      a script
    python -m viproc -m module [args]    a module, as python -m
    python -m viproc -c "code" [args]    a command, as python -c
    python -m viproc                     interactive console

Options before the program: -O1 ... -O5 (also -O 2, --opt=2, --opt 2) set
the optimization level (docs/design/goals.md; as env VIPROC_OPT, which
child processes inherit). Default O1: results bit-identical to NumPy.

``import numpy`` in the program's own code imports viproc instead; the
standard library and installed packages keep the real NumPy (_viproc_hook).
"""

import builtins
import os
import runpy
import sys

import _viproc_hook

USAGE = "\n".join(__doc__.split("\n\n")[i] for i in (0, 1, 2))
LEVELS = ("1", "2", "3", "4", "5")


def take_options(argv):
    """Takes the leading viproc options off argv (sets VIPROC_OPT)."""
    while argv:
        a = argv[0]
        if a in ("-O", "--opt"):
            if len(argv) < 2:
                sys.exit(f"python -m viproc {a}: level missing")
            value, argv = argv[1], argv[2:]
        elif a.startswith("--opt="):
            value, argv = a[len("--opt="):], argv[1:]
        elif a.startswith("-O") and len(a) > 2:
            value, argv = a[2:], argv[1:]
        else:
            break
        value = value.upper().removeprefix("O")
        if value not in LEVELS:
            sys.exit(f"python -m viproc: optimization level must be one of 1-5, got {value!r}")
        os.environ["VIPROC_OPT"] = value
    return argv


def main(argv):
    argv = take_options(argv)  # before the runtime starts: it reads VIPROC_OPT
    _viproc_hook.install()
    if not argv:
        import code

        sys.argv = [""]
        code.interact(banner=f"viproc {sys.modules['viproc'].__version__}: numpy is viproc",
                      local={"__name__": "__main__", "__builtins__": builtins})
        return
    if argv[0] in ("-h", "--help"):
        print(USAGE)
        return
    if argv[0] == "-m":
        if len(argv) < 2:
            sys.exit("python -m viproc -m: module name missing")
        sys.argv = [argv[1]] + argv[2:]  # runpy replaces argv[0] with the module's path
        runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
        return
    if argv[0] == "-c":
        if len(argv) < 2:
            sys.exit("python -m viproc -c: command missing")
        sys.argv = ["-c"] + argv[2:]
        main_globals = {"__name__": "__main__", "__builtins__": builtins}
        exec(compile(argv[1], "<string>", "exec"), main_globals)
        return
    sys.argv = argv
    sys.path[0] = os.path.dirname(os.path.abspath(argv[0]))  # as python prog.py
    runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
