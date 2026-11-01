"""numpy2x demo: the same NumPy code, once with NumPy and once with numpy2x.

Run (with the build's package on PYTHONPATH, see README):
    python python/examples/demo.py
"""

import time

import numpy as np

import numpy2x


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

# 2) numpy2x: wrap the inputs, run the very same function
vdata = [numpy2x.asarray(d) for d in data]
numpy2x.wait_all()
t0 = time.perf_counter()
result = program(vdata)      # returns almost immediately ...
t_issue = time.perf_counter() - t0
result.wait()                # ... the work runs in the background
t_numpy2x = time.perf_counter() - t0

print(f"workers:              {numpy2x.workers()}")
print(f"NumPy:                {t_numpy * 1000:7.1f} ms")
print(f"numpy2x (issuing):     {t_issue * 1000:7.1f} ms")
print(f"numpy2x (total):       {t_numpy2x * 1000:7.1f} ms  -> {t_numpy / t_numpy2x:.2f}x")
print(f"identical to NumPy:   {np.array_equal(np.asarray(result), expected)}")
print("first values:        ", result[:4])
