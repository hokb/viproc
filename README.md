# viproc
A virtual processor, executing serial array instruction streams on parallel, heterogeneous hardware.

Phase 1 is an MVP that runs array instructions autonomously and asynchronously on CPUs,
using NumPy's C kernels and a Python bridge derived from Bohrium's `npbackend`.
See [CLAUDE.md](CLAUDE.md) for architecture and build instructions.

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```
