# viproc vs. ILNumerics Accelerator

12th Gen Intel(R) Core(TM) i7-12700K, 20 cores, viproc 20 workers, build 2026-10-15T13:44:33 #756d13ec; ILNumerics 7.4.0.0, accelerator on, 20 threads, Python 3.13.2, 2026-10-15T21:00:06.

Median speedup of viproc vs. ILNumerics over all cases (ILNumerics time / viproc time; > 1: viproc faster), and in brackets the interquartile range (25th..75th percentile) over the 19 cases. Rows down, columns across.

ILNumerics has no 0-d arrays: the 0-d point runs on [1, 1] arrays. Same programs, reps and grid as the viproc run; inputs generated identically on both sides.

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | **0.81** (0.30–1.82) | **2.14** (1.45–3.27) | **1.84** (1.19–2.96) | **1.56** (1.13–2.15) |
| **10** | **1.86** (1.29–3.01) | **2.15** (1.30–2.73) | **1.89** (1.32–2.92) | **1.97** (1.59–3.49) |
| **100** | **1.78** (1.19–2.67) | **1.77** (1.09–2.42) | **1.75** (1.16–3.21) | **1.21** (0.97–2.55) |
| **1,000** | **1.53** (1.12–2.13) | **2.14** (1.01–3.68) | **1.29** (0.92–2.51) | **1.11** (0.68–1.63) |

0-d scalars: median **2.47x** (1.90–10.14, 15 cases).

viproc bit-identical to NumPy and ILNumerics matching NumPy: 319 / 319 measurements.

## Median per case group

### dependent

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 1.40 | 1.78 | 1.84 | 1.84 |
| **10** | 1.79 | 1.78 | 2.09 | 2.39 |
| **100** | 1.78 | 1.90 | 1.59 | 1.10 |
| **1,000** | 1.96 | 2.14 | 1.08 | 1.11 |

### independent

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 1.82 | 1.79 | 1.65 | 1.51 |
| **10** | 1.65 | 1.57 | 1.45 | 3.88 |
| **100** | 1.65 | 1.60 | 3.38 | 2.50 |
| **1,000** | 1.51 | 4.93 | 2.47 | 1.47 |

### mixed

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.35 | 3.40 | 3.04 | 1.30 |
| **10** | 3.26 | 3.16 | 2.05 | 1.60 |
| **100** | 2.88 | 2.32 | 1.25 | 0.89 |
| **1,000** | 1.49 | 0.96 | 0.87 | 0.71 |

### reductions

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.25 | 1.63 | 1.61 | 1.18 |
| **10** | 1.65 | 1.48 | 1.07 | 1.97 |
| **100** | 1.54 | 1.12 | 1.08 | 1.06 |
| **1,000** | 1.16 | 0.65 | 0.98 | 0.69 |

### sync

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 1.00 | 2.36 | 2.31 | 3.21 |
| **10** | 2.13 | 2.15 | 3.28 | 3.20 |
| **100** | 2.10 | 3.08 | 3.05 | 3.14 |
| **1,000** | 3.29 | 3.19 | 3.09 | 1.77 |

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
