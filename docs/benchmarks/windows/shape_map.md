# viproc shape map

12th Gen Intel(R) Core(TM) i7-12700K, 20 cores, 20 workers, NumPy 2.5.4, build 2026-10-15T13:44:33 #756d13ec, Python 3.13.2, 2026-10-15T21:00:06.

Median speedup of viproc vs. NumPy over all cases (NumPy time / viproc time; > 1: viproc faster), and in brackets the interquartile range (25th..75th percentile) over the 19 cases. Rows down, columns across.

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | **0.72** (0.64–0.76) | **0.66** (0.61–0.72) | **0.68** (0.60–0.78) | **0.85** (0.81–0.95) |
| **10** | **0.67** (0.59–0.73) | **0.64** (0.59–0.70) | **0.80** (0.78–0.90) | **0.96** (0.61–1.77) |
| **100** | **0.69** (0.62–0.76) | **0.82** (0.76–0.93) | **0.96** (0.62–1.63) | **1.09** (0.85–2.00) |
| **1,000** | **0.85** (0.76–0.95) | **1.00** (0.82–1.96) | **1.11** (0.94–2.03) | **1.99** (1.61–3.61) |

0-d scalars: median **1.00x** (0.99–1.01, 15 cases).

Bit-identical to NumPy: 319 / 319 measurements.

## Median per case group

### dependent

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.64 | 0.62 | 0.61 | 0.87 |
| **10** | 0.59 | 0.60 | 0.82 | 0.83 |
| **100** | 0.62 | 0.82 | 0.80 | 0.84 |
| **1,000** | 0.86 | 0.84 | 0.88 | 1.82 |

### independent

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.63 | 0.60 | 0.60 | 0.85 |
| **10** | 0.59 | 0.58 | 0.82 | 1.34 |
| **100** | 0.59 | 0.83 | 1.33 | 1.38 |
| **1,000** | 0.85 | 1.71 | 1.34 | 3.95 |

### mixed

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.77 | 0.66 | 0.68 | 0.93 |
| **10** | 0.65 | 0.65 | 0.80 | 1.43 |
| **100** | 0.69 | 0.82 | 1.44 | 1.73 |
| **1,000** | 0.94 | 1.42 | 1.72 | 2.33 |

### reductions

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.68 | 0.65 | 0.67 | 0.81 |
| **10** | 0.67 | 0.64 | 0.80 | 0.53 |
| **100** | 0.70 | 0.81 | 0.49 | 0.86 |
| **1,000** | 0.72 | 0.88 | 1.03 | 2.97 |

### sync

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.74 | 0.69 | 0.71 | 0.83 |
| **10** | 0.70 | 0.66 | 0.80 | 0.96 |
| **100** | 0.72 | 0.76 | 0.94 | 1.01 |
| **1,000** | 0.82 | 0.94 | 1.02 | 1.53 |

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
