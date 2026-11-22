"""Find Vicon<->IMU time offset by correlating |omega| and write Vicon as TUM (metres, seconds).
usage: sync_vicon.py gyro_accel.csv vicon.csv out_gt_tum.txt [vicon_hz=100]"""

import sys, numpy as np

imu = np.genfromtxt(sys.argv[1], delimiter=",", skip_header=1)


def load_vicon(path):
    rows = []
    for ln in open(path).read().splitlines()[1:]:
        f = ln.strip().split(",")
        if not f or not f[0]:
            continue
        r = [np.nan] * 9
        try:
            r[0] = float(f[0])
            if len(f) >= 9:
                r[1:] = [float(x) if x.strip() else np.nan for x in f[1:9]]
        except ValueError:
            pass
        rows.append(r)
    return np.array(rows)


vic = load_vicon(sys.argv[2])
hz = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0
print(
    "vicon rows",
    len(vic),
    "NaN rows",
    int(np.isnan(vic).any(1).sum()),
    "frame gaps",
    int((np.diff(vic[:, 0]) != 1).sum()),
    "first/last frame",
    int(vic[0, 0]),
    int(vic[-1, 0]),
    "quat norm range",
    np.linalg.norm(vic[:, 2:6], axis=1).min().round(4),
    np.linalg.norm(vic[:, 2:6], axis=1).max().round(4),
    "zero-quat rows",
    int((np.linalg.norm(vic[:, 2:6], axis=1) < 0.5).sum()),
)
q = vic[:, 2:6]  # x y z w (NaN on occluded frames)
# angular speed from relative rotation between consecutive frames
dot = np.abs(np.sum(q[1:] * q[:-1], 1)).clip(0, 1)
wv = 2 * np.arccos(dot) * hz
# IMU |w| averaged to vicon rate on the IMU clock
t = (imu[:, 0] - imu[0, 0]) * 1e-9
wn = np.linalg.norm(imu[:, 1:4], axis=1)
tv = np.arange(len(wv)) / hz + 0.5 / hz


def imu_at(off):  # sample IMU |w| at vicon times shifted by offset (vicon time + off = imu time)
    return np.interp(tv + off, t, wn)


def lowpass(x, n=5):
    return np.convolve(x, np.ones(n) / n, "same")


wv = np.nan_to_num(wv, nan=0.0)
a = lowpass(wv)
best = None
res = []
for off in np.arange(-15, 15.001, 0.01):
    b = lowpass(imu_at(off))
    m = np.isfinite(a) & np.isfinite(b)
    c = np.corrcoef(a[m], b[m])[0, 1]
    res.append((c, off))
res.sort(reverse=True)
print("best offsets (corr, offset s):", [(round(c, 3), round(o, 2)) for c, o in res[:4]])
off = res[0][1]
t0 = imu[0, 0] * 1e-9
tum = np.column_stack([t0 + (vic[:, 0] - vic[0, 0]) / hz + off, vic[:, 6:9] / 1000.0, q])
tum = tum[~np.isnan(tum).any(1)]
np.savetxt(sys.argv[3], tum, fmt="%.6f %.6f %.6f %.6f %.8f %.8f %.8f %.8f")
print(
    "offset used %.2f s;  vicon path length %.1f m, extent %s m"
    % (off, np.linalg.norm(np.diff(tum[:, 1:4], axis=0), axis=1).sum(), np.ptp(tum[:, 1:4], axis=0).round(2))
)
