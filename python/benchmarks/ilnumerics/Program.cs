//ILN(enabled=false)    <- the harness itself is not accelerated; Cases.cs is
// ILNumerics side of the viproc shape map: runs the programs of
// shape_map_cases.py (Cases.cs) on ILNumerics arrays, at the grid points,
// step counts (reps) and input shapes of a plan written by
// python/benchmarks/shape_map_ilnumerics.py, and writes the times and a
// result check per point as JSON lines: first {"machine": ...}, then one
// result per point, flushed at once, so the driver sees progress, can stop
// a point that hangs and resume after it. See README.md.
//
//   dotnet run -c Release -- PLAN.json OUT.jsonl [--threads N]
//
// Timing follows shape_map.py: one warm-up run, then the best of `repeat`
// timed runs (2 for runs slower than 0.3 s); every run ends with Finish() on
// all outputs (the counterpart of viproc.wait_all()). The check runs the
// program for `check_reps` steps and reports, per output, its shape, sum and
// values at a few row-major positions, which the driver compares with NumPy
// on the same inputs.

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using ILNumerics;
using ShapeMap;
using static ILNumerics.ILMath;

if (args.Length < 2) {
    Console.Error.WriteLine("usage: ShapeMapILNumerics PLAN.json OUT.jsonl [--threads N]");
    return 2;
}
string planPath = args[0], outPath = args[1];
for (int i = 2; i < args.Length; i++) {
    if (args[i] == "--threads" && i + 1 < args.Length) {
        Settings.MaxNumberThreads = uint.Parse(args[++i]);
    }
}

var plan = JsonNode.Parse(File.ReadAllText(planPath))!.AsObject();
var points = plan["points"]!.AsArray();
int repeat = (int?)plan["repeat"] ?? 3;
int checkReps = (int?)plan["check_reps"] ?? 3;
int samples = (int?)plan["samples"] ?? 16;

var machine = new JsonObject {
    ["os"] = RuntimeInformation.OSDescription,
    ["cpu"] = CpuName(),
    ["logical_cpus"] = Environment.ProcessorCount,
    ["dotnet"] = RuntimeInformation.FrameworkDescription,
    ["ilnumerics"] = typeof(BaseArray).Assembly.GetName().Version?.ToString(),
    ["ilnumerics_threads"] = Settings.MaxNumberThreads,
    ["accelerator"] = AcceleratorEnabled(),
    ["date"] = DateTime.Now.ToString("s"),
};
Console.WriteLine($"ILNumerics shape map: {points.Count} measurements, ILNumerics {machine["ilnumerics"]}, "
                  + $"accelerator {machine["accelerator"]}, {machine["ilnumerics_threads"]} threads");

using var output = new StreamWriter(outPath, append: false);
output.WriteLine(new JsonObject { ["machine"] = machine }.ToJsonString());
output.Flush();
var total = Stopwatch.StartNew();
for (int pi = 0; pi < points.Count; pi++) {
    var p = points[pi]!.AsObject();
    string name = (string)p["case"]!;
    int reps = (int)p["reps"]!;
    long seed = (long)p["seed"]!;
    var inputs = p["inputs"]!.AsArray();
    var res = new JsonObject { ["case"] = name, ["shape"] = p["shape"]!.DeepClone(), ["reps"] = reps };
    try {
        var (d, f) = MakeInputs(inputs, seed);
        foreach (var a in d) a.Finish();
        foreach (var a in f) a.Finish();

        double Once(int steps) {
            var sw = Stopwatch.StartNew();
            var outs = Cases.Run(name, d, f, steps);
            foreach (var o in outs) o.Finish();
            return sw.Elapsed.TotalSeconds;
        }

        double warm = Once(reps);
        int n = warm < 0.3 ? repeat : Math.Min(repeat, 2);
        var times = new JsonArray();
        double best = double.MaxValue;
        for (int k = 0; k < n; k++) {
            double t = Once(reps);
            times.Add(t);
            best = Math.Min(best, t);
        }
        res["iln_s"] = best;
        res["warm_s"] = warm;
        res["times"] = times;
        res["check"] = Check(Cases.Run(name, d, f, checkReps), samples);
    } catch (Exception e) {
        var inner = e;
        while (inner.InnerException is not null) inner = inner.InnerException;
        res["error"] = $"{e.GetType().Name}: {e.Message}" + (inner != e ? $" <- {inner.GetType().Name}: {inner.Message}" : "");
        if (Environment.GetEnvironmentVariable("SHAPE_MAP_TRACE") == "1") Console.Error.WriteLine(e);
    }
    output.WriteLine(res.ToJsonString());
    output.Flush();
    Console.WriteLine($"[{pi + 1,5}/{points.Count}] {name,-17} {string.Join("x", p["shape"]!.AsArray()),11} "
                      + (res["error"] is null ? $"{(double)res["iln_s"]! * 1e3,10:F3} ms" : $"ERROR {res["error"]}")
                      + $"   ({total.Elapsed.TotalSeconds:F0} s)");
}
Console.WriteLine($"written: {outPath}");
return 0;

// --- inputs ------------------------------------------------------------------------

// Input k of a point: element i (row-major) is 0.5 + u, u in [0, 1) from
// splitmix64 of (seed, k, i): the same values the driver gives NumPy
// (shape_map_ilnumerics.py, `generated`). float32 inputs are rounded from it.
static (Array<double>[] d, Array<float>[] f) MakeInputs(JsonArray inputs, long seed) {
    var d = new List<Array<double>>();
    var f = new List<Array<float>>();
    for (int k = 0; k < inputs.Count; k++) {
        var spec = inputs[k]!.AsArray();
        var shape = spec[0]!.AsArray().Select(x => (long)x!).ToArray();
        string dtype = (string)spec[1]!;
        long rows = shape.Length > 0 ? shape[0] : 1, cols = shape.Length > 1 ? shape[1] : 1;
        long n = rows * cols;
        var vals = new double[n];
        ulong key = ((ulong)seed << 40) | ((ulong)k << 32);
        for (long i = 0; i < n; i++) {
            vals[i] = 0.5 + 1.0 * Uniform(key | (ulong)i);
        }
        if (dtype == "float32") {
            var fv = vals.Select(v => (float)v).ToArray();
            f.Add(FromRowMajor(fv, rows, cols));
        } else {
            d.Add(FromRowMajor(vals, rows, cols));
        }
    }
    return (d.ToArray(), f.ToArray());
}

static double Uniform(ulong key) {
    ulong z = key + 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    z ^= z >> 31;
    return (z >> 11) * (1.0 / (1UL << 53));
}

// Row-major element order, ILNumerics' default storage (column-major) for
// the array: the layout ILNumerics programs use. Element (r, c) matches NumPy's.
static Array<T> FromRowMajor<T>(T[] vals, long rows, long cols) where T : unmanaged {
    var colMajor = new T[vals.Length];
    for (long r = 0; r < rows; r++)
        for (long c = 0; c < cols; c++)
            colMajor[c * rows + r] = vals[r * cols + c];
    return array<T>(colMajor, (InArray<long>)new long[] { rows, cols }, StorageOrders.ColumnMajor);
}

// --- result check ----------------------------------------------------------------------

static JsonObject Check(BaseArray[] outs, int samples) {
    var list = new JsonArray();
    foreach (var o in outs) {
        o.Finish();
        double[] v;  // row-major
        long[] shape = Enumerable.Range(0, (int)o.S.NumberOfDimensions).Select(i => o.S[i]).ToArray();
        if (o is Array<double> ad) {
            v = RowMajor(ad);
        } else if (o is Array<float> af) {
            v = RowMajor(af).Select(x => (double)x).ToArray();
        } else {
            throw new InvalidOperationException($"unexpected output type {o.GetType()}");
        }
        var idx = new JsonArray();
        var val = new JsonArray();
        int m = (int)Math.Min(samples, v.Length);
        for (int j = 0; j < m; j++) {
            long i = m == 1 ? 0 : (long)((double)j * (v.Length - 1) / (m - 1));
            idx.Add(i);
            val.Add(v[i]);
        }
        list.Add(new JsonObject {
            ["shape"] = new JsonArray(shape.Select(x => (JsonNode)x).ToArray()),
            ["sum"] = v.Sum(),
            ["index"] = idx,
            ["values"] = val,
        });
    }
    return new JsonObject { ["outputs"] = list };
}

static T[] RowMajor<T>(Array<T> a) where T : unmanaged {
    T[] col = null;
    a.ExportValues(ref col);  // column-major
    long rows = a.S[0], cols = a.S.NumberOfElements / Math.Max(1, rows);
    var v = new T[col.Length];
    for (long r = 0; r < rows; r++)
        for (long c = 0; c < cols; c++)
            v[r * cols + c] = col[c * rows + r];
    return v;
}

static string CpuName() {
    try {
        if (File.Exists("/proc/cpuinfo")) {
            return File.ReadLines("/proc/cpuinfo")
                .FirstOrDefault(l => l.StartsWith("model name", StringComparison.OrdinalIgnoreCase))?
                .Split(':', 2)[1].Trim() ?? "?";
        }
        return Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "?";
    } catch {
        return "?";
    }
}

// Whether this build ran through the Accelerator: its output embeds the
// kernel resource (ILNumerics.Accelerator.Command.targets).
static bool AcceleratorEnabled() {
    return typeof(Cases).Assembly.GetManifestResourceNames().Contains("ILNumerics.Segment.Kernels.resources");
}
