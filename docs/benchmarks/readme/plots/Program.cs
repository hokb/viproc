// README figures of viproc, drawn with ILNumerics.Drawing.
//   dotnet run -c Release -- <data dir> <output dir>
// Input: CSV files written by ../make_charts.py (shape_map.csv, loop.csv,
// kmeans.csv). Output: shape_map, loop, kmeans as .svg and .bmp (GDI).
using System.Drawing;
using System.Globalization;
using ILNumerics;
using ILNumerics.Drawing;
using ILNumerics.Drawing.Plotting;
using static ILNumerics.Globals;
using static ILNumerics.ILMath;

var dataDir = args.Length > 0 ? args[0] : ".";
var outDir = args.Length > 1 ? args[1] : ".";
var font = new ILNumerics.Drawing.Compat.Font("DejaVu Sans", 11f);
var small = new ILNumerics.Drawing.Compat.Font("DejaVu Sans", 8f);
Label.DefaultFont = font;

Color Hex(string h) => ColorTranslator.FromHtml(h);
var numpyColor = Hex("#7a7974");
var viprocColor = Hex("#2a78d6");
var depColor = Hex("#eb6834");
var ink2 = Hex("#52514e");

List<double[]> ReadCsv(string name) {
    var rows = new List<double[]>();
    foreach (var line in File.ReadLines(Path.Combine(dataDir, name)).Skip(1))
        if (line.Trim().Length > 0)
            rows.Add(line.Split(',').Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray());
    return rows;
}

void Save(Scene scene, string name, int w, int h) {
    scene.Configure();
    using (var s = new FileStream(Path.Combine(outDir, name + ".svg"), FileMode.Create))
        new SVGDriver(s, w, h, scene: scene).Render();
    var gdi = new GDIDriver(w, h) { Scene = scene, BackColor = Color.White };
    gdi.Render();
    gdi.BackBuffer.SaveBitmap(Path.Combine(outDir, name + ".bmp"));   // -> .png: ../make_charts.py
    Console.WriteLine("written: " + name);
}

void Style(PlotCube pc, string xlabel, string ylabel) {
    pc.Axes.XAxis.Label.Text = xlabel;
    pc.Axes.YAxis.Label.Text = ylabel;
    pc.Axes.XAxis.Label.Font = font;
    pc.Axes.YAxis.Label.Font = font;
    pc.Axes.XAxis.Ticks.DefaultLabel.Font = font;
    pc.Axes.YAxis.Ticks.DefaultLabel.Font = font;
}

// --- shape map: median speedup over rows x columns, two panels ---------------
// Diverging colormap over log2(speedup): red (slower) - gray (1x) - blue.
Colormap SpeedupColormap() {
    // ILNumerics colormap data: one row per key, columns position, r, g, b, a
    Array<float> keys = new float[,] {
        { 0.0f, 0.70f, 0.16f, 0.14f, 1 },
        { 0.5f, 0.95f, 0.95f, 0.94f, 1 },
        { 0.75f, 0.45f, 0.66f, 0.92f, 1 },
        { 1.0f, 0.08f, 0.33f, 0.70f, 1 } };
    return new Colormap(keys);
}

void ShapeMap() {
    var rows = ReadCsv("shape_map.csv");   // rows, cols, elements, all, single
    var grid = rows.Where(r => r[0] > 0).ToList();
    var rs = grid.Select(r => r[0]).Distinct().OrderBy(v => v).ToArray();
    var cs = grid.Select(r => r[1]).Distinct().OrderBy(v => v).ToArray();
    var scene = new Scene();
    string[] titles = { "single operations: x = op(x)", "all 43 programs (median)" };
    int[] column = { 4, 3 };
    for (int panel = 0; panel < 2; panel++) {
        Array<float> z = zeros<float>(rs.Length, cs.Length);
        var pc = new PlotCube(twoDMode: true);
        pc.ScreenRect = new RectangleF(0.02f + panel * 0.47f, 0.08f, 0.46f, 0.84f);
        foreach (var r in grid) {
            int i = Array.IndexOf(rs, r[0]), j = Array.IndexOf(cs, r[1]);
            var v = r[column[panel]];
            z[i, j] = (float)Math.Log2(v);
            var lab = new Label(v.ToString("0.00", CultureInfo.InvariantCulture) + "x") {
                Position = new Vector3(j, i, 1), Font = small,
                Anchor = new PointF(0.5f, 0.5f) };
            lab.Color = Math.Abs(Math.Log2(v)) > 1.2 ? Color.White : Color.Black;
            pc.Add(lab);
        }
        var img = new ImageSCPlot(z, Colormaps.Custom, Tuple.Create(-2f, 2f));
        img.Colormap = SpeedupColormap();
        pc.Add(img);
        Style(pc, "columns", panel == 0 ? "rows" : "");
        pc.Axes.XAxis.Ticks.Mode = TickMode.Manual;
        pc.Axes.YAxis.Ticks.Mode = TickMode.Manual;
        pc.Axes.XAxis.Ticks.Clear();
        pc.Axes.YAxis.Ticks.Clear();
        for (int j = 0; j < cs.Length; j++) pc.Axes.XAxis.Ticks.Add(j, cs[j].ToString("N0", CultureInfo.InvariantCulture));
        for (int i = 0; i < rs.Length; i++) pc.Axes.YAxis.Ticks.Add(i, rs[i].ToString("N0", CultureInfo.InvariantCulture));
        pc.Axes.XAxis.GridMajor.Visible = false;
        pc.Axes.YAxis.GridMajor.Visible = false;
        // rows x cols = 10,000 elements: index i + j = 8 on the half-decade grid
        Array<float> lx = new float[] { 8f - (rs.Length - 1), cs.Length - 1 };
        Array<float> ly = new float[] { rs.Length - 1, 8f - (cs.Length - 1) };
        var ten4 = new LinePlot(lx, ly, lineColor: Color.Black, lineWidth: 2, lineStyle: DashStyle.Dashed);
        Array<float> lp3 = new float[,] { { lx.GetValue(0), ly.GetValue(0), 2 }, { lx.GetValue(1), ly.GetValue(1), 2 } };
        ten4.Line.Positions.Update(lp3.T);
        pc.Add(ten4);
        pc.Add(new Title(titles[panel]));
        scene.Add(pc);
    }
    var note = new Label("speedup = NumPy time / viproc time (red: viproc slower, blue: faster); dashed: 10,000 elements") {
        Position = new Vector3(0.5, 0.97, 0), Anchor = new PointF(0.5f, 1f), Font = font };
    note.Color = ink2;
    scene.Screen.Add(note);
    Save(scene, "shape_map", 1300, 640);
}

// --- Parallel.For: the column loop over matrix sizes ---------------------------
void Loop() {
    var rows = ReadCsv("loop.csv");   // n, threads, loop, loop_dep, vectorized (speedups)
    Array<float> x = rows.Select(r => (float)r[0]).ToArray();
    LinePlot Line(int col, string tag, Color c, DashStyle d, MarkerStyle m) {
        Array<float> y = rows.Select(r => (float)r[col]).ToArray();
        var lp = new LinePlot(x, y, tag: tag, lineColor: c, lineWidth: 3, lineStyle: d, markerStyle: m);
        lp.Marker.Size = 11;
        lp.Marker.Fill.Color = c;
        lp.Marker.Border.Color = c;
        return lp;
    }
    var pc = new PlotCube(twoDMode: true) {
        Line(2, "viproc: C[i] = sum(abs(sin(A[:, i])))", viprocColor, DashStyle.Solid, MarkerStyle.Dot),
        Line(3, "viproc: C[:i+1] = ...  (order matters)", depColor, DashStyle.Solid, MarkerStyle.Diamond),
        Line(1, "NumPy + thread pool (Parallel.For): C[i] = ...", numpyColor, DashStyle.Dashed, MarkerStyle.Square),
    };
    // 1x: plain NumPy
    Array<float> one = new float[] { (float)rows.Min(r => r[0]), (float)rows.Max(r => r[0]) };
    pc.Add(new LinePlot(one, ones<float>(2, 1), lineColor: ink2, lineWidth: 1, lineStyle: DashStyle.PointDash, tag: "plain NumPy (1x)"));
    Style(pc, "matrix size n (A: n x n, loop over its n columns)", "speedup vs. NumPy");
    pc.Axes.XAxis.Ticks.Mode = TickMode.Manual;
    pc.Axes.XAxis.Ticks.Clear();
    foreach (var r in rows) pc.Axes.XAxis.Ticks.Add((float)r[0], r[0].ToString("N0", CultureInfo.InvariantCulture));
    pc.Add(new Legend { Location = new PointF(0.12f, 0.15f), Anchor = new PointF(0, 0) });
    pc.Add(new Title("Python loop over the n columns of A, 4 cores"));
    pc.AutoScaleOnAdd = false;
    pc.Limits.Set(new Vector3(150, 0, 0), new Vector3(3150, 4.2, 0));
    var scene = new Scene { pc };
    Save(scene, "loop", 1000, 560);
}

// --- k-means: time per run over the program's running time ----------------------
// ids: 2 NumPy, 5 NumPy with MALLOC_MMAP_THRESHOLD_/MALLOC_TRIM_THRESHOLD_ raised, 4 viproc
void KMeans() {
    var rows = ReadCsv("kmeans.csv");   // id, elapsed ms, ms per run
    LinePlot Line(int id, string tag, Color c, MarkerStyle m) {
        var sel = rows.Where(r => (int)r[0] == id).ToList();
        Array<float> x = sel.Select(r => (float)(r[1] / 1000)).ToArray();
        Array<float> y = sel.Select(r => (float)r[2]).ToArray();
        var lp = new LinePlot(x, y, tag: tag, lineColor: c, lineWidth: 3, markerStyle: m);
        lp.Marker.Size = 11;
        lp.Marker.Fill.Color = c;
        lp.Marker.Border.Color = c;
        return lp;
    }
    var pc = new PlotCube(twoDMode: true) {
        Line(2, "NumPy", numpyColor, MarkerStyle.Square),
        Line(5, "NumPy, malloc tuned: no page faults", Hex("#a8a7a0"), MarkerStyle.TriangleUp),
        Line(4, "the same NumPy program on viproc", viprocColor, MarkerStyle.Dot),
    };
    Style(pc, "elapsed time [s]", "time per k-means run [ms]");
    pc.Add(new Legend { Location = new PointF(0.97f, 0.47f), Anchor = new PointF(1, 0.5f) });
    pc.Add(new Title("k-means (ILNumerics' Part 3 program), 1203 x 1203, k = 50, 4 cores"));
    pc.AutoScaleOnAdd = false;
    pc.Limits.Set(new Vector3(0, 0, 0), new Vector3(22, 3300, 0));
    var scene = new Scene { pc };
    Save(scene, "kmeans", 1000, 560);
}

var only = args.Length > 2 ? args[2] : "all";
if (only is "all" or "shape_map") ShapeMap();
if (only is "all" or "loop") Loop();
if (only is "all" or "kmeans") KMeans();
