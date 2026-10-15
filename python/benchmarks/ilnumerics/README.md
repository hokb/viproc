# ILNumerics Accelerator reference for the shape map

The programs of `../shape_map_cases.py`, written for ILNumerics in C#
(`Cases.cs`), run by `../shape_map_ilnumerics.py` at exactly the grid
points and step counts of a viproc shape map run on the same machine. It
produces two maps in the style of `shape_map.py`:

- `shape_map_ilnumerics.*`: ILNumerics Accelerator vs. NumPy (NumPy times
  from the viproc run),
- `shape_map_vs_ilnumerics.*`: viproc vs. ILNumerics Accelerator, per case
  ILNumerics time / viproc time (> 1: viproc faster), median and IQR over
  the cases.

## Run

Requires the .NET 8 SDK and a viproc build (see the repository README).

```bash
PYTHONPATH=build/python python3 python/benchmarks/shape_map.py --out reports/shape_map
python3 python/benchmarks/shape_map_ilnumerics.py --viproc reports/shape_map/shape_map.json \
    --out reports/shape_map
```

The driver builds this project in Release (the ILNumerics Accelerator
compiles `Cases.cs`; `Program.cs`, the harness, is excluded with
`//ILN(enabled=false)`), writes the plan (`ilnumerics_plan.json`: case,
shape, reps, input shapes per point), runs it, checks every point against
NumPy and writes the reports. `--iln ilnumerics_raw.json` only redraws.
One program start runs all points. Only if a point gives no result for
`--timeout` seconds (default 300) is it recorded as not measured and the
program restarted after it; such points are listed in the reports and left
out of the medians.

## License

ILNumerics is commercial. A build without a license runs, but throttled
(measured: the same program about 6x slower), so its times are useless for
this comparison. Build once on a licensed seat: the build writes
`ILNumerics_deploy/net8.0/ilnumerics.lic`, a runtime license for this
assembly (`ShapeMapILNumerics`). With that file in the source tree the
unchanged project can be rebuilt and run on other machines (Linux too),
as in hokb/decentralized-array-execution-artifacts2026. The license is bound
to these sources (any change to the C# files needs a new license) and to
the configuration it was made for (`Release|x64`, the driver builds that:
`dotnet build -c Release -p:Platform=x64`, output in `bin/x64/Release/net8.0`).
A build whose license does not verify runs under ILNumerics' Starter plan
(the build log says so: "Perform checks failed ... 'Starter' plan
conditions"): throttled, with runtime and data size limits, and later
`LicenseException`s. Such times must not go into the maps.

## What is compared

- Same programs, same dependency chains and sync points, same reps and
  grid; inputs are generated identically on both sides (splitmix64, see
  `generated` in the driver and `MakeInputs` in `Program.cs`), so results
  can be compared: shape, sum and 16 sampled values per output after
  3 steps, relative tolerance 1e-9 (float32: 2e-5). ILNumerics uses its
  own kernels: results match NumPy, they are not bit-identical.
- Timing as in `shape_map.py`: warm-up run, best of 3 (2 for runs over
  0.3 s), each run ends when all outputs are finished (`Finish()`, the
  counterpart of `viproc.wait_all()`).
- Kernel compilation: the Accelerator compiles its kernels at runtime and by
  default runs the regular ILNumerics functions until a kernel is ready.
  The harness sets `ExecutionContext.Default.AllowLegacyFallback = false`:
  the first run of a segment waits for its kernel, so compilation falls into
  the warm-up run and the timed runs use compiled kernels
  (`--legacy-fallback` keeps ILNumerics' default). The setting, the build
  configuration and whether a debugger was attached are recorded in the raw
  results (`machine`).
- Arrays are ILNumerics' default (column-major) arrays holding the same
  elements as the NumPy arrays; 0-d data are `[1, 1]` arrays.
- ILNumerics runs with its own thread settings (`Settings.MaxNumberThreads`,
  recorded in the report; `--threads N` overrides).
- The Accelerator fuses expressions into kernels (`cos(pow(-x, 3))` is one
  segment); viproc runs one NumPy loop per op. The map compares the two
  systems as users get them, not array ILP alone.
- API differences: `log1p(x * x)` is `log(1 + x * x)` (no `log1p` in
  ILNumerics), in-place updates assign with `x.a = ...` (see `Cases.cs`).
