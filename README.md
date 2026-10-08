# viproc
A virtual processor, executing serial array instruction streams on parallel, heterogeneous hardware.

viproc adds instruction-level parallelism for array instructions to regular NumPy programs:
instructions of a sequentially written program run in parallel whenever their inputs are ready.
Dependencies are resolved at runtime by local decisions, without a DAG or ahead-of-time analysis.
The goal is strong scaling of whole array programs; the actual computation is delegated to
NumPy's own inner loops.
See [CLAUDE.md](CLAUDE.md) for architecture and build instructions.

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```
