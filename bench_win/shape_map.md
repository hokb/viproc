# viproc shape map

12th Gen Intel(R) Core(TM) i7-12700K, 20 cores, 20 workers, NumPy 2.5.4, build 2026-10-15T11:14:29 #756d13ec, Python 3.13.2, 2026-10-15T13:29:41.

Median speedup of viproc vs. NumPy over all cases (NumPy time / viproc time; > 1: viproc faster), and in brackets the interquartile range (25th..75th percentile) over the 19 cases. Rows down, columns across.

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | **0.70** (0.64–0.75) | **0.64** (0.59–0.70) | **0.66** (0.60–0.75) | **0.86** (0.78–0.96) |
| **10** | **0.66** (0.61–0.71) | **0.63** (0.56–0.66) | **0.79** (0.76–0.91) | **0.98** (0.67–1.91) |
| **100** | **0.66** (0.59–0.75) | **0.79** (0.76–0.87) | **0.98** (0.70–1.79) | **1.11** (0.96–2.20) |
| **1,000** | **0.83** (0.74–0.92) | **1.02** (0.84–1.87) | **1.13** (0.96–2.01) | **2.27** (1.66–3.47) |

0-d scalars: median **0.99x** (0.98–1.00, 15 cases).

Bit-identical to NumPy: 319 / 319 measurements.

## Median per case group

### dependent

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.64 | 0.59 | 0.62 | 0.88 |
| **10** | 0.59 | 0.56 | 0.83 | 0.83 |
| **100** | 0.59 | 0.80 | 0.85 | 0.89 |
| **1,000** | 0.85 | 0.86 | 0.87 | 1.95 |

### independent

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.61 | 0.59 | 0.59 | 0.85 |
| **10** | 0.60 | 0.56 | 0.83 | 1.33 |
| **100** | 0.57 | 0.83 | 1.35 | 1.37 |
| **1,000** | 0.86 | 1.43 | 1.38 | 3.94 |

### mixed

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.72 | 0.65 | 0.68 | 0.92 |
| **10** | 0.65 | 0.63 | 0.79 | 1.50 |
| **100** | 0.68 | 0.79 | 1.46 | 1.78 |
| **1,000** | 0.91 | 1.50 | 1.64 | 2.27 |

### reductions

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.70 | 0.64 | 0.63 | 0.78 |
| **10** | 0.66 | 0.61 | 0.77 | 0.58 |
| **100** | 0.68 | 0.78 | 0.63 | 1.02 |
| **1,000** | 0.70 | 0.91 | 1.00 | 3.11 |

### sync

| rows \ cols | 1 | 10 | 100 | 1,000 |
|---:|---:|---:|---:|---:|
| **1** | 0.71 | 0.68 | 0.68 | 0.91 |
| **10** | 0.69 | 0.66 | 0.79 | 0.98 |
| **100** | 0.67 | 0.76 | 0.96 | 1.02 |
| **1,000** | 0.83 | 0.96 | 0.99 | 1.53 |

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
