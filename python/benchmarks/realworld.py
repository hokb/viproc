"""A few real-world shaped NumPy programs, run as plain NumPy and through viproc (module mode).

Run from a build: PYTHONPATH=build/python python3 python/benchmarks/realworld.py
"""
import time, sys, numpy, viproc

def logreg(np, X, y, steps):
    # logistic regression, full-batch gradient descent
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(steps):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w = w - 0.1 * (X.T @ g) / X.shape[0]
        b = b - 0.1 * float(g.mean())
    return w

def heat(np, u, steps):
    # explicit 2-d heat equation, 5-point stencil
    for _ in range(steps):
        u[1:-1, 1:-1] = u[1:-1, 1:-1] + 0.1 * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1])
    return u

def features(np, X):
    # column-wise standardization + a few derived features (typical preprocessing)
    mu = X.mean(axis=0); sd = X.std(axis=0)
    Z = (X - mu) / sd
    Z = np.clip(Z, -3, 3)
    out = [Z, Z ** 2, np.tanh(Z), np.log1p(np.abs(Z)), np.where(Z > 0, Z, 0.01 * Z)]
    return np.concatenate(out, axis=1).sum(axis=0)

def bs_greeks(np, S, K, T, r, v):
    # Black-Scholes price + delta + vega for many options
    from math import sqrt, pi
    d1 = (np.log(S / K) + (r + 0.5 * v * v) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    cdf = lambda x: 0.5 * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
    price = S * cdf(d1) - K * np.exp(-r * T) * cdf(d2)
    delta = cdf(d1)
    vega = S * np.exp(-0.5 * d1 * d1) / sqrt(2 * pi) * np.sqrt(T)
    return price.sum() + delta.sum() + vega.sum()

def run(name, fn, make, reps=5):
    out = {}
    for label, mod in (("numpy", numpy), ("viproc", viproc)):
        best = 1e9
        for _ in range(reps):
            args = make(mod)
            if label == "viproc": viproc.wait_all()
            t = time.perf_counter(); r = fn(mod, *args); r = numpy.asarray(r).copy(); best = min(best, time.perf_counter() - t)
        out[label] = (best, r)
    same = numpy.array_equal(out["numpy"][1], out["viproc"][1], equal_nan=True)
    tn, tv = out["numpy"][0] * 1e3, out["viproc"][0] * 1e3
    print(f"{name:34s} NumPy {tn:9.2f} ms  viproc {tv:9.2f} ms  {tn/tv:5.2f}x  identical={same}", flush=True)

rng = numpy.random.default_rng(1)
for n, d, steps in ((200, 10, 200), (5000, 50, 50), (100000, 50, 10)):
    X = rng.random((n, d)); y = (rng.random(n) > 0.5) * 1.0
    run(f"logreg {n}x{d}, {steps} steps", lambda np, X, y: logreg(np, X, y, steps), lambda m: (m.asarray(X), m.asarray(y)))
for side, steps in ((32, 500), (256, 200), (2048, 20)):
    U = rng.random((side, side))
    run(f"heat {side}x{side}, {steps} steps", lambda np, u: heat(np, u, steps), lambda m: (m.array(U),))
for n, d in ((100, 8), (10000, 20), (200000, 20)):
    X = rng.random((n, d)) + 0.1
    run(f"features {n}x{d}", features, lambda m: (m.asarray(X),))
for n in (100, 10000, 1000000):
    S = rng.random(n) * 100 + 50; K = rng.random(n) * 100 + 50; T = rng.random(n) + 0.1
    run(f"bs_greeks {n}", lambda np, S, K, T: bs_greeks(np, S, K, T, 0.02, 0.3), lambda m: (m.asarray(S), m.asarray(K), m.asarray(T)))
