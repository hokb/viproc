# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**viproc** is a virtual processor: it takes a *serial* stream of array
instructions (think `c = a + b`, `d = sum(c)`, …) and executes it efficiently on
*parallel, heterogeneous* hardware. The program author writes sequential array
code; viproc is responsible for scheduling, fusing, and dispatching that work to
the available compute devices.

Current phase: **architecture first.** The instruction model / IR and the
backend abstraction are being defined. The only planned backend for now is a
multithreaded/SIMD **CPU backend**, which also serves as the reference
implementation for correctness. GPU backends come later and must not be
precluded by design decisions made now.

## Tech stack

- C# / .NET 10 (`net10.0`, SDK pinned via `global.json` with `latestFeature` roll-forward)
- Solution file: `Viproc.slnx`
- Tests: xUnit
- Shared build settings live in `Directory.Build.props` (nullable enabled,
  warnings as errors, latest recommended analyzers, code style enforced in build)

## Repository layout

```
src/
  Viproc.Core/           # Instruction model, IR, scheduling, backend abstractions
  Viproc.Backends.Cpu/   # CPU backend (multithreaded + SIMD), reference implementation
tests/
  Viproc.Core.Tests/     # Unit tests for Core and the CPU backend
```

Dependency rule: `Core` depends on nothing in this repo. Backends depend on
`Core` only, never on each other.

## Commands

```bash
dotnet build                 # build everything
dotnet test                  # run all tests
dotnet test --filter "FullyQualifiedName~SomeClass"   # run a subset
dotnet format                # apply .editorconfig code style
dotnet format --verify-no-changes   # check style (CI-style)
```

Always run `dotnet build` and `dotnet test` before committing; the build treats
warnings as errors.

## Architecture (draft — to be refined)

1. **Instruction stream** — the user-facing, serial API that records array
   operations instead of executing them eagerly.
2. **IR** — a device-independent representation of the recorded operations
   (data-flow graph over array values with shapes and element types).
3. **Scheduler / optimizer** — analyzes dependencies, fuses element-wise
   operations, partitions work, and assigns it to devices.
4. **Backends** — implement a common interface to allocate memory, transfer
   data, and execute (possibly fused) kernels on a specific device.

Open design questions are tracked in the "Open questions" section below.

## Conventions

- Language for code, comments, docs, and commit messages: **English**.
- Follow `.editorconfig`; standard .NET naming (PascalCase types/members,
  `_camelCase` private fields).
- Prefer immutable types (`record`, `readonly struct`) for IR nodes.
- Hot paths in backends: avoid allocations, prefer `Span<T>`,
  `System.Numerics.Vector<T>` / `System.Runtime.Intrinsics`.
- Every new instruction needs a CPU-backend implementation and a test that
  checks its result against a straightforward scalar reference.
- Keep public API surface small; mark types `internal` unless they are meant
  to be used by backends or callers.

## Open questions

- [ ] Eager vs. lazy execution model (when is a recorded stream flushed?)
- [ ] Supported element types and shape semantics (broadcasting? strides?)
- [ ] Memory model across devices (who owns buffers, when are transfers made?)
- [ ] Kernel representation for backends (expression trees, own IR, codegen?)
- [ ] Planned GPU route (ILGPU, CUDA, Vulkan/OpenCL, …)
