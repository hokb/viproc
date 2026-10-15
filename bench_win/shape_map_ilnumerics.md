# ILNumerics Accelerator shape map

12th Gen Intel(R) Core(TM) i7-12700K, 20 cores, ILNumerics 7.4.0.0, accelerator on, 20 threads, NumPy 2.5.4, Python 3.13.2, 2026-10-15T21:00:06.

Median speedup of ILNumerics vs. NumPy over all cases (NumPy time / ILNumerics time; > 1: ILNumerics faster), and in brackets the interquartile range (25th..75th percentile) over the 19 cases. Rows down, columns across.

ILNumerics has no 0-d arrays: the 0-d point runs on [1, 1] arrays. Same programs, reps and grid as the viproc run; inputs generated identically on both sides.

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | **0.93** (0.42–2.46) | **0.35** (0.18–0.48) | **0.33** (0.22–0.58) | **0.56** (0.39–0.80) |
| **10** | **0.33** (0.20–0.56) | **0.33** (0.21–0.51) | **0.45** (0.31–0.66) | **0.40** (0.24–0.70) |
| **100** | **0.35** (0.23–0.57) | **0.48** (0.36–0.75) | **0.45** (0.27–0.89) | **1.03** (0.64–1.26) |
| **1,000** | **0.56** (0.37–0.77) | **0.74** (0.30–1.68) | **1.05** (0.64–1.54) | **2.11** (1.05–4.00) |

0-d scalars: median **0.40x** (0.09–0.53, 15 cases).

Matches NumPy (rel. tol., 3 steps): 319 / 319 measurements.

## Median per case group

### dependent

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.46 | 0.35 | 0.33 | 0.47 |
| **10** | 0.33 | 0.33 | 0.40 | 0.35 |
| **100** | 0.35 | 0.43 | 0.39 | 0.89 |
| **1,000** | 0.44 | 0.39 | 0.91 | 1.04 |

### independent

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.44 | 0.43 | 0.45 | 0.56 |
| **10** | 0.48 | 0.45 | 0.55 | 0.39 |
| **100** | 0.45 | 0.52 | 0.39 | 1.12 |
| **1,000** | 0.56 | 0.33 | 1.04 | 2.34 |

### mixed

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 2.23 | 0.19 | 0.22 | 0.73 |
| **10** | 0.19 | 0.21 | 0.39 | 0.75 |
| **100** | 0.24 | 0.36 | 1.27 | 2.89 |
| **1,000** | 0.70 | 2.10 | 2.99 | 3.98 |

### reductions

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 2.71 | 0.40 | 0.42 | 0.69 |
| **10** | 0.40 | 0.43 | 0.74 | 0.47 |
| **100** | 0.46 | 0.72 | 0.68 | 1.03 |
| **1,000** | 0.71 | 1.35 | 1.07 | 2.79 |

### sync

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.72 | 0.29 | 0.30 | 0.26 |
| **10** | 0.32 | 0.29 | 0.24 | 0.22 |
| **100** | 0.32 | 0.25 | 0.26 | 0.32 |
| **1,000** | 0.25 | 0.25 | 0.33 | 0.86 |

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
