# Testing viproc: a guide for test users

viproc runs unchanged NumPy programs in parallel. Every array operation becomes a task; independent
operations run at the same time on several cores. Results are bit-identical to NumPy (see the
[README](../README.md)).

What you are testing: does your existing NumPy code run with viproc, and does it get faster?

**Contents:**
[Requirements](#1-requirements) ·
[Install](#2-install-into-your-existing-python) ·
[Check](#3-check-that-it-works) ·
[Run](#4-running-programs-with-viproc) ·
[IDEs](#5-in-your-ide) ·
[Measure](#6-measuring-and-comparing) ·
[Known differences](#7-known-differences) ·
[Feedback](#8-feedback)

## 1. Requirements

- **Python ≥ 3.12, 64-bit.** Check with `python --version`.
- **NumPy ≥ 2.5.** It is upgraded automatically during installation if needed (see the note below).
- **A C++20 compiler.** There are no prebuilt wheels yet, so viproc is compiled during installation:
  - **Windows:** Visual Studio 2022 or the *Build Tools for Visual Studio* with the
    *Desktop development with C++* workload
  - **Linux:** a recent GCC or Clang
- **git**, only for the first install command below. pip fetches CMake and Ninja itself.

## 2. Install into your existing Python

Use the same Python installation that you (or your IDE) normally work with:

```bash
python -m pip install "viproc @ git+https://github.com/hokb/viproc"
```

Without git:

```bash
python -m pip install "viproc @ https://github.com/hokb/viproc/archive/refs/heads/main.zip"
```

The build takes about a minute.

> [!WARNING]
> pip upgrades NumPy in this environment to ≥ 2.5. Packages that require an older NumPy may stop working
> (for example, Numba reports *"Numba needs NumPy 2.1 or less"*). Check beforehand with:
>
> ```bash
> python -m pip install --dry-run "viproc @ git+https://github.com/hokb/viproc"
> ```

**Safer alternative:** a separate environment that still sees all your installed packages. The new NumPy
goes only into it; your existing installation stays unchanged:

```bash
python -m venv --system-site-packages .venv-viproc
```

Activate it (Windows: `.venv-viproc\Scripts\activate`, Linux: `source .venv-viproc/bin/activate`) and
install viproc into it as shown above.

**conda:** activate the environment and use `python -m pip install ...` as shown above.

**Uninstall:**

```bash
python -m pip uninstall viproc
```

## 3. Check that it works

```bash
python -c "import viproc; print(viproc.__version__, 'workers:', viproc.workers())"
python -m viproc -c "import numpy; print(numpy.__name__)"
```

The first command prints the version and the number of worker threads (one per physical core but one, which the main thread keeps). The second
must print `viproc`; if it prints `numpy`, viproc is not active.

## 4. Running programs with viproc

There are four ways. The first two need no code changes.

**a) From the command line:**

```bash
python -m viproc my_program.py arg1 arg2    # also: python -m viproc -m module
python my_program.py arg1 arg2              # the same program with plain NumPy, for comparison
```

**b) With the environment variable `VIPROC=1`.** The easiest way for IDEs: when it is set, viproc switches
itself on whenever this Python starts.

| Shell | Command |
|---|---|
| Linux (bash, zsh) | `VIPROC=1 python my_program.py` |
| Windows cmd | `set VIPROC=1`, then `python my_program.py` |
| PowerShell | `$env:VIPROC="1"`, then `python my_program.py` |

**c) Import viproc instead of NumPy** (one line changed):

```python
import viproc as np          # was: import numpy as np
```

**d) Use viproc for selected arrays only:** keep NumPy and convert individual arrays with
`a = viproc.asarray(a)`.

With a) and b), only **your own code** gets viproc. The standard library and installed packages (SciPy,
pandas, …) keep the real NumPy and receive viproc arrays through NumPy's array protocols.

## 5. In your IDE

### PyCharm (also IntelliJ with the Python plugin)

1. **Pick the interpreter:** under *Settings → Project → Python Interpreter*, select the interpreter viproc
   is installed in (e.g. the `.venv-viproc` from above). Or install from PyCharm's terminal; viproc then goes
   into the project interpreter automatically.
2. **Adjust the run configuration:** *Run → Edit Configurations…*, select your configuration and add
   `VIPROC=1` under *Environment variables*. This applies to Run and Debug alike.
3. **For all new configurations:** *Edit configuration templates… → Python*, set `VIPROC=1` there.
4. **For comparison:** duplicate the configuration and leave the variable out of the copy. The copy runs
   with plain NumPy.
5. **Alternative without the variable:** in the configuration, switch from *Script path* to *Module name*,
   enter `viproc`, and put the script path under *Parameters*.

### VS Code (Python extension)

Add this entry to `.vscode/launch.json`:

```json
{
  "name": "Current file with viproc",
  "type": "debugpy",
  "request": "launch",
  "program": "${file}",
  "env": { "VIPROC": "1" },
  "console": "integratedTerminal"
}
```

Select the interpreter that has viproc in the bottom-right corner.

### Spyder

Spyder runs code in its IPython console. Set `VIPROC=1` before starting Spyder, or use
`import viproc as np` in your code.

### Jupyter (Notebook / Lab)

Set `VIPROC=1` before starting Jupyter (the kernel inherits the environment), or use `import viproc as np`
in the first cell.

### Debugging

Breakpoints work as usual. But every look at array values in the debugger (variables view, watch, `print`)
is a sync point, where the program waits for that result. Never measure times in debug mode.

## 6. Measuring and comparing

- **Whole programs:** run `python prog.py` and `python -m viproc prog.py`, several times each.
- **Within one run:** inside `with viproc.disabled():` a block runs with plain NumPy.
  `viproc.enable(False)` / `viproc.enable(True)` switch it for longer stretches.
- **Timing:** read a value at the end of each measurement, e.g. `float(x.sum())` or `np.asarray(x)`.
  Otherwise you only measure how long it takes to issue the operations, not to compute them.
- **Number of cores:** `VIPROC_WORKERS=4` (or 1, 2, 8, …) sets the number of workers, e.g. for scaling
  tests.
- **Where it pays off:** many array operations on medium to large data that do not all depend on each
  other: iterative algorithms, simulations, image and signal processing. Loops over single scalar values
  gain little and may get slightly slower.

## 7. Known differences

- `isinstance(x, np.ndarray)` in your own code tests for viproc arrays under viproc.
- `np.array(a)` and `np.asarray(a)` always copy a NumPy array into viproc, even with `copy=False`.
- Data types viproc does not run (object, str, float16, …) give plain NumPy arrays.
- On very small data, viproc can be slightly slower than NumPy.

## 8. Feedback

Please send us:

- **Your environment:** the output of
  `python -c "import sys, numpy, viproc; print(sys.version, numpy.__version__, viproc.__version__, viproc.workers())"`,
  plus your CPU and operating system.
- **Timings:** with and without viproc, several runs each.
- **Differences:** any result that differs from NumPy, and any error that only occurs under viproc, ideally
  with a small example.
- **Installation problems:** the complete pip output.

Please [open an issue](https://github.com/hokb/viproc/issues).
