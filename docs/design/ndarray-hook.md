# The numpy.ndarray hook: limitations and known issues

`viproc.enable_hook()`, `python -m viproc prog.py` and `VIPROC=1` route
operators on plain `numpy.ndarray`s (`a + b`, `-a`, `a ** 2`, `a @ b`,
comparisons) in the user's own code through viproc, so an unchanged NumPy
program gets array ILP. In module mode (`import viproc as np`, `python -m
viproc`) the ufunc proxies take ndarray arguments too. `import viproc`
alone never installs it (decided: an import must not change NumPy's
behaviour). This document lists what the hook does not do, where it
behaves differently from plain NumPy in anything but speed, and what is
still open. Results stay bit-identical to NumPy (G2) in every case listed
here; the items are about types, timing, memory and coverage.

## How it decides

Per operator call, on the main thread, before anything is computed:

- **Taken** if an operand is a viproc array, or if an operand is a plain,
  dense `numpy.ndarray` (C- or F-contiguous, aligned), no operand is a
  strided one, and the calling code is the user's (not the standard
  library, `site-packages`/`dist-packages`, NumPy or viproc).
- **Left to NumPy**: subclasses (`np.matrix`, masked arrays) and any
  operand NumPy would defer to (`__array_ufunc__ = None`, other array
  types); strided ndarray operands (below); in-place operators on
  ndarrays (below); calls from other threads; small results.
- **Small results stay `numpy.ndarray`**, computed right away (nothing to
  overlap). Large ones become `viproc.ndarray`, computed asynchronously.
- **ndarray operands are copied** at issue (the user may write them while
  the task waits), except where nobody can write the memory any more:
  read-only arrays over `bytes` or a read-only `mmap`, and, for operators,
  temporaries that own their memory (`np.sin(x) * 2`: only the
  interpreter's stack holds `np.sin(x)`). Those are read in place.

## Differences visible to the program

1. **Result types.** Large results are `viproc.ndarray`, not
   `numpy.ndarray`: `type(r) is np.ndarray` and
   `isinstance(r, np.ndarray)` are false. Every NumPy function accepts
   them (`__array_ufunc__`, `__array_function__`, `__array__`), and every
   method and attribute of `ndarray` is there. Code that dispatches on the
   type (C extensions, `isinstance` checks) takes its non-ndarray path,
   usually `np.asarray(r)`: a sync point and an export (item 3).
2. **Errors arrive later.** An error that depends on element values (an
   integer `**` with a negative exponent, `errstate(all="raise")`, an
   index out of bounds) is found when the task runs. It is raised at the
   next check of the running user code or at the next use of the result,
   as `AsyncFloatingPointError`, `AsyncValueError`, `AsyncIndexError`
   (subclasses of NumPy's types and of `viproc.AsyncError`), with the
   issue site of the failed op in the message
   (`docs/design/error-handling.md`). `except FloatingPointError:` still
   matches; a `try` block that ends before the error is found does not.
3. **Exports are aliases.** `np.asarray(v)`, `v.numpy()`, `__array__` give
   NumPy views of the same memory, writable, as `np.asarray` of an
   ndarray (memory-management.md, D7). While such a view lives, **every op
   on that memory runs synchronously**: NumPy code may write it at any
   time, so viproc cannot run ahead. Keep exports short-lived (`print`,
   a NumPy function call); a long-lived `b = np.asarray(v)` makes the
   following ops on `v` as slow as NumPy (never slower in results).
4. **Warnings.** Floating-point warnings carry the issue site (file, line,
   module's `__warningregistry__`), as NumPy's do, but may be emitted a
   few statements later, when the task has run. `errstate` is the one in
   effect at the issue, including `call`, `log` and `print` modes.
5. **Pickle and copy.** viproc arrays pickle as `numpy.ndarray` (the
   receiver needs no viproc), with every protocol, and `copy.copy` /
   `copy.deepcopy` give viproc arrays, as NumPy gives ndarrays.
6. **Debugging.** A task's error shows the issue site, not the worker's
   stack. In a debugger, results are pending until something reads them;
   printing one is a sync point and shows the right values.

## Not taken (yet): stays NumPy, synchronous

- **Strided ndarray operands** (`m[:, 1]`, `m[::2]`, a transposed
  non-contiguous view of a plain ndarray). viproc cannot read them in
  place (the user may write the memory later) and a dense copy changes
  the strides of NumPy's loop calls, which can change results. Decision
  (maintainer): at **O1** (bit-identical, today) they stay NumPy, no copy;
  at **O2** and up (results may deviate, not built yet) a dense copy goes
  to viproc. Measured with `python/benchmarks/strided_identity.py`
  (NumPy 2.5.3, Xeon with AVX-512, also with `NPY_DISABLE_CPU_FEATURES`
  down to the X86_V2 baseline): 10,310 cases (every element-wise ufunc
  with one output, one and two operands and scalars, six dtypes;
  `sum`/`prod`/`mean`/`var`/`std`/`max`/`cumsum` along every axis;
  `matmul`), strided view vs. its dense copy in order K: **no
  difference**. Not a proof for other machines, NumPy versions or loops;
  with a dense copy in **C order** instead, `sum` of a transposed view
  does differ (the order of the pairwise blocks). Run the script on every
  new machine (step 5); a per-loop proof would let O1 take the copy too.
  viproc views (`v[:, 1]` of a viproc array) are not affected: they run
  asynchronously with their own strides.
- **In-place operators on ndarrays** (`x += y`, `x` a plain ndarray):
  NumPy's, synchronous. Planned: when the instruction after the operator
  stores into a plain name (`STORE_FAST`, `STORE_NAME`, `STORE_GLOBAL`,
  `STORE_DEREF`, which cannot fail) and nothing but that name references
  `x`, return a new viproc array instead of writing `x`. Attribute and
  item targets keep NumPy's in-place write: their store can fail after
  the write (`t[0] += y` on a tuple writes the array, then raises).
- **ndarray methods and `np.*` functions** called on plain ndarrays outside
  module mode go to NumPy (only operators are hooked); with
  `import viproc as np` or `python -m viproc` the ufuncs and `np.where`
  take them.
- **Other threads**: only the thread that started the runtime issues;
  operators on other threads are NumPy's.

## Process and interpreter

- **fork.** Worker threads do not survive `fork()` (`multiprocessing`'s
  fork start method, the default on Linux before Python 3.14). Checked: a
  child that issues an op after `os.fork()` hangs (Python warns about
  forking a multi-threaded process). Use the `spawn` or `forkserver`
  start method. Open: an `os.register_at_fork` handler (wait for all
  tasks before the fork, a fresh runtime in the child).
- **Memory peaks.** The main thread runs up to 1,000 ops ahead; each
  in-flight result holds its memory, where NumPy frees a temporary at once.
  A program with many large temporaries can need much more memory than
  under NumPy. Open: a memory-based look-ahead limit.
- **Slot patching.** The hook replaces `numpy.ndarray`'s number and
  comparison slots (`tp_as_number`, `tp_richcompare`).
  `enable_hook(False)` restores the slots it saved, which drops a patch
  another package made later. Free-threaded CPython (3.13t) is not
  supported: the slots are patched without a lock and the runtime
  assumes one Python thread.
- **Library code** keeps NumPy's operators (decided by the caller's file
  path): a library that computes on viproc arrays it got from the user
  still gets viproc results, through the protocols.

## Opt-in or default

Open (step 5): decided after suite runs on several machines. Until then
the hook is opt-in.
