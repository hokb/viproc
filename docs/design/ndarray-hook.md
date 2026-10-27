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
  ndarrays others can see (next item); calls from other threads; small
  results.
- **In-place operators on ndarrays** (`x += y`, `-=`, `*=`, `/=`, `//=`,
  `%=`, `**=`, `&=`, `|=`, `^=`, `<<=`, `>>=`; not `@=`): NumPy writes
  `x`. viproc takes `x`'s memory over instead (no copy, `x`'s layout kept)
  and runs the op on it asynchronously when nobody can tell the
  difference: the instruction after the operator stores into a plain
  name (`STORE_FAST`, `STORE_DEREF`, `STORE_GLOBAL`, `STORE_NAME` into an
  exact dict: stores that cannot fail), that name holds `x`, nothing else
  references `x` (no other name, view, container or buffer: reference
  count 2, the name and the interpreter's stack; no weak reference), `x` owns its memory and
  is writable, `y` is no strided ndarray (O1, below), and the result
  would be large. The name is then bound to
  a viproc array with `x`'s values after the op. Everything else stays
  NumPy's in-place write, visible to every other reference: `y = x; x +=
  1` (`y` sees it), views of `x` or `x` a view, `o.x += 1`, `d[k] += 1`,
  and `t[0] += 1` on a tuple (NumPy writes the array, then the store
  raises). Measured (4 cores, four 2e6-element float64 arrays, 80 in-place
  ops): NumPy 73 ms, hook 29 ms, bit-identical; a small in-place op pays
  ~110 instructions for the checks (NumPy's ~5,700). **CPython 3.12 and
  3.13 only**: their loads always push a strong reference, so the count
  is exactly 2. CPython 3.14 borrows loads of fast locals; a count of 2
  could then be the name plus another reference, so there the in-place
  operators stay NumPy's (open: tell borrowed from strong stack
  references).
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

## Testing

- `python/tests/test_ndarray_hook.py`: the hook on user code, each check in
  a child process (operators bit for bit against NumPy, result types,
  library code, borrowing, in-place takeover; also on CPython 3.12).
- The whole pytest suite also runs with the hook installed
  (`VIPROC_NDARRAY_HOOK=1`; CI and ctest's `python_hook`). `conftest.py`
  then treats the test files as library code, so their NumPy references
  stay NumPy's, and keeps small results of viproc arrays (otherwise the
  small-data tests would compare NumPy with NumPy). Running the suite in
  this mode found a deadlock: a task needing the GIL while the main thread
  waited inside an issue (fixed, `RuntimeOptions::block_enter`).
- `python/benchmarks/suite.py --mode hook|module` runs the suite's plain
  NumPy programs on numpy.ndarray inputs under the hook (`hook`: operators
  only; `module`: as `python -m viproc`, the cases' `np` is viproc), bit
  for bit against NumPy; `test_benchmark_suite.py` runs a few of them.
  Quick run, 4 cores (91 configurations, all identical): score 79 with
  viproc arrays as inputs, 68 in module mode, 61 in hook mode. Indexing
  and reductions score lowest with ndarray inputs: `x[idx]` and methods
  such as `x.sum()` on an ndarray stay NumPy's (only operators are
  hooked), until the first viproc result takes over.

## Opt-in or default

Full suite, 4 cores, one build (`docs/benchmarks.md`, 10-27): all 100
configurations bit-identical in every mode; score 85 with viproc arrays as
inputs, 62 as `python -m viproc` with ndarray inputs, 57 with
`enable_hook()` alone. Programs that hold their state in ndarrays and
assign into them (`u[1:-1] = ...`) run below NumPy's speed under the hook
(`stencil_1d` medium 0.56x, `kmeans_points` large 0.60x): every op copies
the ndarray slices it reads, and every assignment into the ndarray syncs.
That breaks G3's "never below 0.95x", so the hook stays **opt-in**.
Open: decide per issue site whether the hook takes an op (as the dispatch
policy decides inline vs. async per site): a site whose results are
assigned into ndarrays or synced right away, or whose ndarray operands
must be copied, would stay NumPy's; and run the suite on more machines
(`--mode hook|module`).
