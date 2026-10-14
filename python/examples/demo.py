"""viproc demo: the same NumPy code, once with NumPy and once with viproc.

Run (with the build's package on PYTHONPATH, see README):
    python python/examples/demo.py
"""

import time

import numpy as np

import viproc


def program(xs):
    """Ordinary sequential NumPy code: 8 independent chains of work."""
    out = []
    for x in xs:
        for _ in range(10):
            x = np.sin(x) * 0.5 + np.cos(x) * 0.25
        out.append(x)
    return sum(out[1:], out[0])


rng = np.random.default_rng(0)
data = [rng.standard_normal(1_000_000) for _ in range(8)]

# 1) plain NumPy
t0 = time.perf_counter()
expected = program(data)
t_numpy = time.perf_counter() - t0

# 2) viproc: wrap the inputs, run the very same function
vdata = [viproc.asarray(d) for d in data]
viproc.wait_all()
t0 = time.perf_counter()
result = program(vdata)      # returns almost immediately ...
t_issue = time.perf_counter() - t0
result.wait()                # ... the work runs in the background
t_viproc = time.perf_counter() - t0

print(f"workers:              {viproc.workers()}")
print(f"NumPy:                {t_numpy * 1000:7.1f} ms")
print(f"viproc (issuing):     {t_issue * 1000:7.1f} ms")
print(f"viproc (total):       {t_viproc * 1000:7.1f} ms  -> {t_numpy / t_viproc:.2f}x")
print(f"identical to NumPy:   {np.array_equal(np.asarray(result), expected)}")
print("first values:        ", result[:4])
