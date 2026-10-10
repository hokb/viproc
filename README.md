# viproc
A virtual processor, executing serial array instruction streams on parallel, heterogeneous hardware.

viproc adds instruction-level parallelism for array instructions to regular NumPy programs:
instructions of a sequentially written program run in parallel whenever their inputs are ready.
Dependencies are resolved at runtime by local decisions, without a DAG or ahead-of-time analysis.
The goal is strong scaling of whole array programs; the actual computation is delegated to
NumPy's own inner loops.
See [CLAUDE.md](CLAUDE.md) for architecture and build instructions.

## Usage

```python
import numpy as np
import viproc

a = viproc.asarray(np.random.rand(1_000_000))   # or: import viproc as np
b = viproc.asarray(np.random.rand(1_000_000))
c = np.sin(a) * b + 1.0      # returns immediately, computed in the background
d = np.cos(b) - a            # independent of c: runs in parallel
print((c + d)[:5])           # sync point
```

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure     # needs: pip install pytest
PYTHONPATH=build/python python3 python/benchmarks/strong_scaling.py
```
