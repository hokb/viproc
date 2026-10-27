"""Hook mode (VIPROC_NDARRAY_HOOK=1): the whole suite runs with the
numpy.ndarray operator hook installed. The tests' own code keeps NumPy's
operators, as library code does, so its NumPy references stay NumPy's and
every test still compares viproc arrays with NumPy; viproc arrays keep
their small results (the hook leaves small results of ndarrays to NumPy,
which these tests would see as NumPy computing NumPy). What the hook does to
user code is tested by test_ndarray_hook.py (in child processes, also for
small results) and by the benchmark suite's hook and module modes
(python/benchmarks/suite.py --mode hook|module: whole NumPy programs, bit
for bit)."""

import os

import viproc

if viproc.hook_enabled():
    viproc._ndarray.hook_exclude(os.path.dirname(os.path.abspath(__file__)))
    viproc._ndarray._HOST_SMALL = False
    viproc._viproc.set_host_small(False)
