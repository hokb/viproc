# ILNumerics Accelerator shape map

12th Gen Intel(R) Core(TM) i7-12700K, 20 cores, ILNumerics 7.4.0.0, accelerator on, 12 threads, NumPy 2.5.4, Python 3.13.2, 2026-10-15T13:29:41.

Median speedup of ILNumerics vs. NumPy over all cases (NumPy time / ILNumerics time; > 1: ILNumerics faster), and in brackets the interquartile range (25th..75th percentile) over the 19 cases. Rows down, columns across.

ILNumerics has no 0-d arrays: the 0-d point runs on [1, 1] arrays. Same programs, reps and grid as the viproc run; inputs generated identically on both sides.

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | **0.68** (0.42–2.54) | **0.30** (0.18–0.40) | **0.40** (0.21–0.48) | **0.56** (0.39–0.78) |
| **10** | **0.39** (0.19–0.51) | **0.30** (0.21–0.44) | **0.46** (0.33–0.68) | **0.43** (0.24–0.72) |
| **100** | **0.41** (0.23–0.58) | **0.47** (0.33–0.73) | **0.50** (0.25–0.98) | **0.89** (0.56–1.36) |
| **1,000** | **0.57** (0.38–0.76) | **0.71** (0.28–1.18) | **0.92** (0.63–1.47) | **1.98** (0.99–3.45) |

0-d scalars: median **0.33x** (0.08–0.63, 15 cases).

Matches NumPy (rel. tol., 3 steps): 319 / 319 measurements.

## Median per case group

### dependent

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 1.79 | 0.26 | 0.33 | 0.45 |
| **10** | 0.33 | 0.30 | 0.42 | 0.33 |
| **100** | 0.34 | 0.35 | 0.37 | 0.90 |
| **1,000** | 0.43 | 0.39 | 0.91 | 1.02 |

### independent

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.44 | 0.34 | 0.40 | 0.56 |
| **10** | 0.45 | 0.43 | 0.52 | 0.35 |
| **100** | 0.47 | 0.47 | 0.33 | 0.89 |
| **1,000** | 0.57 | 0.29 | 0.92 | 2.05 |

### mixed

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 2.26 | 0.19 | 0.22 | 0.65 |
| **10** | 0.19 | 0.20 | 0.38 | 0.80 |
| **100** | 0.24 | 0.43 | 1.34 | 2.54 |
| **1,000** | 0.69 | 1.01 | 2.87 | 3.39 |

### reductions

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 2.98 | 0.39 | 0.43 | 0.73 |
| **10** | 0.43 | 0.42 | 0.67 | 0.49 |
| **100** | 0.44 | 0.66 | 0.60 | 0.92 |
| **1,000** | 0.66 | 1.37 | 1.01 | 2.41 |

### sync

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.52 | 0.33 | 0.41 | 0.35 |
| **10** | 0.39 | 0.38 | 0.30 | 0.23 |
| **100** | 0.41 | 0.31 | 0.24 | 0.29 |
| **1,000** | 0.32 | 0.24 | 0.29 | 0.78 |

## Cases

| case | group | P | program |
|---|---|---:|---|
| dep1_abs | dependent | 1 | x = abs(x): 1 cheap op per step |
| dep2_abs_pow | dependent | 1 | x = abs(x ** 0.5): 2 ops per step |
| dep3_cos_pow_neg | dependent | 1 | x = cos((-x) ** 3): 3 ops per step |
| ind2 | independent | 2 | 2 independent chains: abs(a), cos(b) |
| ind3 | independent | 3 | 3 independent chains: abs(a), cos(b), sqrt(c) |
| ind_cores_3ops | independent | cores | one chain per core, cos((-x) ** 3) per step |
| ind8_cheap | independent | 8 | 8 chains of a cheap 2-op step (x * 0.5 + 0.5) |
| ind_cores_f32 | independent | cores | one float32 chain per core, sin(x) * 0.5 + 0.4 |
| expr2 | mixed | 2 | x = sin(x) + cos(x + pi) * 3: two subterms |
| expr8 | mixed | 8 | x = mean of 8 independent unary/binary terms of x and y |
| inplace | mixed | cores | per core: x *= 0.99; x += y; sqrt(x, out=x) |
| broadcast | mixed | 1 | m = m * 0.5 + col - row: (rows, cols) with (rows, 1) and (1, cols) |
| stencil | mixed | 1 | 5-point stencil on slices (Jacobi step, interior) |
| center_cols | reductions | 1 | x = abs(x - mean(x, axis=0) * 0.5) + 0.1 |
| normalize_rows | reductions | 1 | x = x / max(x, axis=1, keepdims) + 0.5 |
| sum_cores | reductions | cores | per core: x = cos(x * 0.5); s += sum(x) |
| dep3_sync | sync | 1 | x = cos((-x) ** 3), then read max(x) into Python |
| ind3_sync | sync | 3 | 3 independent chains, read max of each step into Python |
| converge | sync | 1 | iterate x = cos(x) until a max-change test in Python (fixed step count) |
