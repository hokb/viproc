using ILNumerics;
using static ILNumerics.Globals;
using static ILNumerics.ILMath;

namespace ShapeMap;

/// <summary>
/// The programs of python/benchmarks/shape_map_cases.py, written as an
/// ILNumerics user writes them: one array instruction per NumPy call, the
/// same dependency chains, the same sync points. Keep both files in step;
/// shape_map_ilnumerics.py checks the results against NumPy.
///
/// Differences that ILNumerics' API forces:
/// - log1p(x * x) (expr8) is log(1.0 + x * x): ILNumerics has no log1p
///   (one more element-wise op, same parallel structure).
/// - In-place updates (x *= 0.99, out=x) assign into the existing array
///   (`x.a = ...`), ILNumerics' way of writing into an array.
/// - mean(x, axis=0) is mean(x, dim: 0), a [1, cols] row that broadcasts
///   like NumPy's (cols,) vector; full reductions return [1, 1] arrays.
/// - 0-d data are [1, 1] arrays (ILNumerics has no 0-d arrays).
/// </summary>
public static class Cases {

    /// Runs case <paramref name="name"/> on its inputs for <paramref name="reps"/>
    /// steps and returns its outputs (not yet finished: the caller syncs).
    public static BaseArray[] Run(string name, Array<double>[] d, Array<float>[] f, int reps) {
        switch (name) {
            case "dep1_abs": return One(Dep1Abs(d[0], reps));
            case "dep2_abs_pow": return One(Dep2AbsPow(d[0], reps));
            case "dep3_cos_pow_neg": return One(Dep3CosPowNeg(d[0], reps));
            case "ind2": return Ind2(d, reps);
            case "ind3": return Ind3(d, reps);
            case "ind_cores_3ops": return IndCores3Ops(d, reps);
            case "ind8_cheap": return Ind8Cheap(d, reps);
            case "ind_cores_f32": return IndCoresF32(f, reps);
            case "expr2": return One(Expr2(d[0], reps));
            case "expr8": return One(Expr8(d[0], d[1], reps));
            case "inplace": return Inplace(d, reps);
            case "broadcast": return One(Broadcast(d[0], d[1], d[2], reps));
            case "stencil": return One(Stencil(d[0], reps));
            case "center_cols": return One(CenterCols(d[0], reps));
            case "normalize_rows": return One(NormalizeRows(d[0], reps));
            case "sum_cores": return SumCores(d, reps);
            case "dep3_sync": return One(Dep3Sync(d[0], reps));
            case "ind3_sync": return Ind3Sync(d, reps);
            case "converge": return One(Converge(d[0], reps));
            default: throw new ArgumentException($"unknown case '{name}'");
        }
    }

    // RetArray results are volatile (used once): keep them as Array<double>.
    static BaseArray[] One(Array<double> a) => new BaseArray[] { a };

    /// A sync point as the NumPy program has it: read a value into the host language.
    static double Sync(InArray<double> x) {
        using var _ = Scope.Enter(x);
        return (double)maxall(x);
    }

    // --- dependent chains (P = 1) -------------------------------------------------

    static RetArray<double> Dep1Abs(InArray<double> x0, int reps) {
        using var _ = Scope.Enter(x0);
        Array<double> x = x0;
        for (int i = 0; i < reps; i++) {
            x = abs(x);
        }
        return x;
    }

    static RetArray<double> Dep2AbsPow(InArray<double> x0, int reps) {
        using var _ = Scope.Enter(x0);
        Array<double> x = x0;
        for (int i = 0; i < reps; i++) {
            x = abs(pow(x, 0.5));
        }
        return x;
    }

    static RetArray<double> Dep3CosPowNeg(InArray<double> x0, int reps) {
        using var _ = Scope.Enter(x0);
        Array<double> x = x0;
        for (int i = 0; i < reps; i++) {
            x = cos(pow(-x, 3.0));
        }
        return x;
    }

    // --- independent chains -------------------------------------------------------

    static BaseArray[] Ind2(Array<double>[] d, int reps) {
        Array<double> a = d[0], b = d[1];
        for (int i = 0; i < reps; i++) {
            a = abs(a);
            b = cos(b);
        }
        return new BaseArray[] { a, b };
    }

    static BaseArray[] Ind3(Array<double>[] d, int reps) {
        Array<double> a = d[0], b = d[1], c = d[2];
        for (int i = 0; i < reps; i++) {
            a = abs(a);
            b = cos(b);
            c = sqrt(c);
        }
        return new BaseArray[] { a, b, c };
    }

    static BaseArray[] IndCores3Ops(Array<double>[] d, int reps) {
        var xs = (Array<double>[])d.Clone();
        for (int i = 0; i < reps; i++) {
            for (int k = 0; k < xs.Length; k++) {
                xs[k] = cos(pow(-xs[k], 3.0));
            }
        }
        return xs;
    }

    static BaseArray[] Ind8Cheap(Array<double>[] d, int reps) {
        var xs = (Array<double>[])d.Clone();
        for (int i = 0; i < reps; i++) {
            for (int k = 0; k < xs.Length; k++) {
                xs[k] = xs[k] * 0.5 + 0.5;
            }
        }
        return xs;
    }

    static BaseArray[] IndCoresF32(Array<float>[] f, int reps) {
        var xs = (Array<float>[])f.Clone();
        for (int i = 0; i < reps; i++) {
            for (int k = 0; k < xs.Length; k++) {
                xs[k] = sin(xs[k]) * 0.5f + 0.4f;
            }
        }
        return xs;
    }

    // --- mixed unary/binary expressions ------------------------------------------

    static RetArray<double> Expr2(InArray<double> x0, int reps) {
        using var _ = Scope.Enter(x0);
        Array<double> x = x0;
        for (int i = 0; i < reps; i++) {
            x = sin(x) + cos(x + pi) * 3.0;
        }
        return x;
    }

    static RetArray<double> Expr8(InArray<double> x0, InArray<double> y, int reps) {
        using var _ = Scope.Enter(x0, y);
        Array<double> x = x0;
        for (int i = 0; i < reps; i++) {
            Array<double> t0 = sin(x);
            Array<double> t1 = cos(y);
            Array<double> t2 = exp(-x);
            Array<double> t3 = sqrt(y);
            Array<double> t4 = tanh(x * y);
            Array<double> t5 = atan(x - y);
            Array<double> t6 = log(1.0 + x * x);
            Array<double> t7 = y * 0.3;
            Array<double> s = t0 + t1;
            s = s + t2;
            s = s + t3;
            s = s + t4;
            s = s + t5;
            s = s + t6;
            s = s + t7;
            x = s * 0.125;
        }
        return x;
    }

    static BaseArray[] Inplace(Array<double>[] d, int reps) {
        Array<double> y = d[d.Length - 1];
        var xs = new Array<double>[d.Length - 1];
        for (int k = 0; k < xs.Length; k++) {
            xs[k] = d[k].C;  // mutated below
        }
        for (int i = 0; i < reps; i++) {
            foreach (Array<double> x in xs) {
                x.a = x * 0.99;
                x.a = x + y;
                x.a = sqrt(x);
            }
        }
        return xs;
    }

    static RetArray<double> Broadcast(InArray<double> m0, InArray<double> col, InArray<double> row, int reps) {
        using var _ = Scope.Enter(m0, col, row);
        Array<double> m = m0;
        for (int i = 0; i < reps; i++) {
            m = m * 0.5 + col - row;
        }
        return m;
    }

    static RetArray<double> Stencil(InArray<double> x0, int reps) {
        using var _ = Scope.Enter(x0);
        Array<double> x = x0.C;
        if (x.S[0] < 3 || x.S[1] < 3) {
            return x;  // empty interior: NumPy's slices are empty, nothing to do
        }
        for (int i = 0; i < reps; i++) {
            x[r(1, end - 1), r(1, end - 1)] = 0.25 * (x[r(0, end - 2), r(1, end - 1)] + x[r(2, end), r(1, end - 1)]
                                                     + x[r(1, end - 1), r(0, end - 2)] + x[r(1, end - 1), r(2, end)]);
        }
        return x;
    }

    // --- reductions -------------------------------------------------------------------

    static RetArray<double> CenterCols(InArray<double> x0, int reps) {
        using var _ = Scope.Enter(x0);
        Array<double> x = x0;
        for (int i = 0; i < reps; i++) {
            x = abs(x - mean(x, dim: 0) * 0.5) + 0.1;
        }
        return x;
    }

    static RetArray<double> NormalizeRows(InArray<double> x0, int reps) {
        using var _ = Scope.Enter(x0);
        Array<double> x = x0;
        for (int i = 0; i < reps; i++) {
            x = x / max(x, dim: 1) + 0.5;
        }
        return x;
    }

    static BaseArray[] SumCores(Array<double>[] d, int reps) {
        var xs = (Array<double>[])d.Clone();
        var sums = new Array<double>[xs.Length];
        for (int k = 0; k < xs.Length; k++) {
            sums[k] = sumall(xs[k]);
        }
        for (int i = 0; i < reps; i++) {
            for (int k = 0; k < xs.Length; k++) {
                xs[k] = cos(xs[k] * 0.5);
                sums[k] = sums[k] + sumall(xs[k]);
            }
        }
        return xs.Concat(sums).Cast<BaseArray>().ToArray();
    }

    // --- sync after every step ------------------------------------------------------

    static RetArray<double> Dep3Sync(InArray<double> x0, int reps) {
        using var _ = Scope.Enter(x0);
        Array<double> x = x0;
        for (int i = 0; i < reps; i++) {
            x = cos(pow(-x, 3.0));
            Sync(x);
        }
        return x;
    }

    static BaseArray[] Ind3Sync(Array<double>[] d, int reps) {
        Array<double> a = d[0], b = d[1], c = d[2];
        for (int i = 0; i < reps; i++) {
            a = abs(a);
            b = cos(b);
            c = sqrt(c);
            Sync(a); Sync(b); Sync(c);
        }
        return new BaseArray[] { a, b, c };
    }

    static RetArray<double> Converge(InArray<double> x0, int reps) {
        using var _ = Scope.Enter(x0);
        Array<double> x = x0;
        for (int i = 0; i < reps; i++) {
            Array<double> prev = x;
            x = cos(x);
            if ((double)maxall(abs(x - prev)) < 0.0) {  // never true: same work every run
                break;
            }
        }
        return x;
    }
}
