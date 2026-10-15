# viproc vs. ILNumerics Accelerator

12th Gen Intel(R) Core(TM) i7-12700K, 20 cores, viproc 20 workers, build 2026-10-15T11:14:29 #756d13ec; ILNumerics 7.4.0.0, accelerator on, 12 threads, Python 3.13.2, 2026-10-15T13:29:41.

Median speedup of viproc vs. ILNumerics over all cases (ILNumerics time / viproc time; > 1: viproc faster), and in brackets the interquartile range (25th..75th percentile) over the 19 cases. Rows down, columns across.

ILNumerics has no 0-d arrays: the 0-d point runs on [1, 1] arrays. Same programs, reps and grid as the viproc run; inputs generated identically on both sides.

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | **1.02** (0.29–1.69) | **2.27** (1.69–3.35) | **1.82** (1.36–2.97) | **1.59** (1.06–2.17) |
| **10** | **1.80** (1.39–3.10) | **1.90** (1.49–2.78) | **1.90** (1.29–2.69) | **2.22** (1.52–3.57) |
| **100** | **1.68** (1.21–2.71) | **1.78** (1.11–2.34) | **1.78** (1.26–4.13) | **1.38** (1.09–3.11) |
| **1,000** | **1.58** (1.13–2.14) | **2.23** (1.37–4.08) | **1.33** (0.93–2.85) | **1.25** (0.81–1.77) |

0-d scalars: median **3.02x** (1.58–11.11, 15 cases).

viproc bit-identical to NumPy and ILNumerics matching NumPy: 319 / 319 measurements.

## Median per case group

### dependent

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.36 | 2.31 | 1.90 | 1.94 |
| **10** | 1.80 | 1.90 | 1.98 | 2.49 |
| **100** | 1.74 | 2.28 | 2.08 | 1.10 |
| **1,000** | 1.96 | 2.23 | 1.10 | 1.12 |

### independent

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 1.69 | 1.77 | 1.66 | 1.44 |
| **10** | 1.57 | 1.53 | 1.45 | 3.77 |
| **100** | 1.38 | 1.61 | 5.44 | 2.61 |
| **1,000** | 1.36 | 4.98 | 2.59 | 1.56 |

### mixed

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.33 | 3.37 | 3.06 | 1.44 |
| **10** | 3.17 | 2.98 | 2.10 | 1.53 |
| **100** | 2.77 | 2.11 | 1.36 | 1.07 |
| **1,000** | 1.47 | 1.57 | 0.87 | 0.74 |

### reductions

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.24 | 1.65 | 1.45 | 1.06 |
| **10** | 1.50 | 1.46 | 1.15 | 2.22 |
| **100** | 1.44 | 1.17 | 1.20 | 1.38 |
| **1,000** | 1.21 | 0.67 | 0.99 | 1.29 |

### sync

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 1.43 | 2.21 | 1.69 | 2.65 |
| **10** | 1.84 | 1.66 | 2.64 | 3.39 |
| **100** | 1.68 | 2.41 | 3.14 | 3.45 |
| **1,000** | 2.58 | 3.43 | 3.39 | 1.95 |

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
