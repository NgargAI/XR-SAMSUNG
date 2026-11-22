"""Apply Track-3 style degradation to a real IMU csv (to test failure handling).
usage: degrade.py in.csv out.csv [--secs 120] [--fail zeros|nan|freeze]
noise x8 ramp from 30%, gyro/accel bias drift from 45%, 25% frame loss in 55-70%, IMU dead for the last 5 s."""

import sys, argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("i")
ap.add_argument("o")
ap.add_argument("--secs", type=float, default=120)
ap.add_argument("--fail", default="zeros")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
L = open(a.i).read().splitlines()
hdr = L[0]
d = np.array([[float(x) for x in l.split(",")] for l in L[1:]])
d = d[d[:, 0] <= d[0, 0] + a.secs * 1e9]
t = (d[:, 0] - d[0, 0]) * 1e-9
T = t[-1]
n = len(d)
sc = 1 + 7 * np.clip((t - 0.30 * T) / (0.05 * T), 0, 1)
dr = np.clip((t - 0.45 * T) / (0.10 * T), 0, 1)[:, None]
d[:, 1:4] += dr * np.array([0.004, -0.003, 0.002]) + rng.normal(0, 0.01, (n, 3)) * (sc[:, None] - 1)
d[:, 4:7] += dr * np.array([0.05, -0.04, 0.03]) + rng.normal(0, 0.06, (n, 3)) * (sc[:, None] - 1)
keep = np.ones(n, bool)
w = (t > 0.55 * T) & (t < 0.70 * T)
keep[w] = rng.random(w.sum()) > 0.25
fail = t >= T - 5
if a.fail == "zeros":
    d[fail, 1:7] = 0
elif a.fail == "nan":
    d[fail, 1:7] = np.nan
else:
    d[fail, 1:7] = d[np.argmax(fail), 1:7]
keep[fail] = True
with open(a.o, "w") as f:
    f.write(hdr + "\n")
    for r in d[keep]:
        f.write("%d,%s\n" % (r[0], ",".join("nan" if np.isnan(x) else "%.9g" % x for x in r[1:])))
print("rows kept", int(keep.sum()), "of", n, " dead from t=%.1f s" % (T - 5))
