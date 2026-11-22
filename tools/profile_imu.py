import sys, numpy as np

d = np.genfromtxt(sys.argv[1], delimiter=",", skip_header=1)
t = d[:, 0] * 1e-9
dt = np.diff(t)
print("rows", len(d), "duration %.1f s" % (t[-1] - t[0]), "NaN rows", int(np.isnan(d).any(1).sum()))
print(
    "dt ms: median %.3f  p1 %.3f  p99 %.3f  max %.1f"
    % (np.median(dt) * 1e3, np.percentile(dt, 1) * 1e3, np.percentile(dt, 99) * 1e3, dt.max() * 1e3)
)
print("gaps >15ms:", int((dt > 0.015).sum()), " non-monotonic:", int((dt <= 0).sum()))
an = np.linalg.norm(d[:, 4:7], axis=1)
wn = np.linalg.norm(d[:, 1:4], axis=1)
print("|a| mean %.3f std %.3f min %.2f max %.2f" % (an.mean(), an.std(), an.min(), an.max()))
print("|w| mean %.3f max %.2f  all-zero rows %d" % (wn.mean(), wn.max(), int((an < 1).sum())))
rep = int(((np.diff(d[:, 1:7], axis=0) == 0).all(1)).sum())
print("bit-identical consecutive rows:", rep)
# noise level over time (10 windows): std of first difference of accel
for k in range(0, 10):
    s = slice(k * len(d) // 10, (k + 1) * len(d) // 10)
    print(
        " decile %d  acc-diff std %.3f  gyro-diff std %.4f  |w| mean %.3f"
        % (k, np.diff(d[s, 4:7], axis=0).std(), np.diff(d[s, 1:4], axis=0).std(), wn[s].mean())
    )
