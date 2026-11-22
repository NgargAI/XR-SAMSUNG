"""Diagnostic: how good could IMU integration be with PERFECT (Vicon) orientation?"""

import sys, numpy as np

imu = np.genfromtxt(sys.argv[1], delimiter=",", skip_header=1)
G = np.loadtxt(sys.argv[2])
t0 = imu[0, 0] * 1e-9
ti = imu[:, 0] * 1e-9
tg = G[:, 0]
N = int(float(sys.argv[3]) * 100) if len(sys.argv) > 3 else 6000
tg, q, p = tg[:N], G[:N, 4:8], G[:N, 1:4]


def qR(q):
    x, y, z, w = q.T
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


R = qR(q)
hz = 100.0
# body-frame angular rate of the marker from consecutive rotations
dR = np.einsum("nji,njk->nik", R[:-1], R[1:])
wm = np.stack([dR[:, 2, 1] - dR[:, 1, 2], dR[:, 0, 2] - dR[:, 2, 0], dR[:, 1, 0] - dR[:, 0, 1]], 1) / 2 * hz
gyr = np.stack([np.interp(tg[:-1] + 0.5 / hz, ti, imu[:, k]) for k in (1, 2, 3)], 1)
m = np.linalg.norm(wm, axis=1) > 0.4
U, S, Vt = np.linalg.svd(gyr[m].T @ wm[m])
D = np.eye(3)
D[2, 2] = np.sign(np.linalg.det(U @ Vt))
Rim = U @ D @ Vt  # w_imu = Rim w_marker
print("extrinsic (marker->imu) rotation:\n", Rim.round(3))
res = gyr[m] - wm[m] @ Rim.T
print(
    "gyro-vs-vicon residual rms %.4f rad/s over %d moving samples (|w| rms %.3f)"
    % (np.sqrt(np.mean(res**2)), m.sum(), np.sqrt(np.mean(wm[m] ** 2)))
)
Rwi = R @ Rim.T  # world<-imu
f = np.stack([np.interp(tg + 0.0, ti, imu[:, k]) for k in (4, 5, 6)], 1)
aw = np.einsum("nij,nj->ni", Rwi, f)
print("still-ish gravity check: mean R*f over first 5 s =", aw[:500].mean(0).round(3), "(expect ~[0,0,9.8] if vicon z is up)")
g = np.linalg.norm(aw[:500].mean(0))
aw = aw - np.array([0, 0, g])
# GT accel from smoothed position
k = np.ones(9) / 9
ps = np.stack([np.convolve(p[:, i], k, "same") for i in range(3)], 1)
a_gt = np.gradient(np.gradient(ps, 1 / hz, axis=0), 1 / hz, axis=0)
sl = slice(20, -20)
r = aw[sl] - a_gt[sl]
print(
    "accel residual (IMU-oracle-orientation minus GT accel): mean %s  rms %.3f  | GT accel rms %.3f"
    % (r.mean(0).round(3), np.sqrt(np.mean(r**2)), np.sqrt(np.mean(a_gt[sl] ** 2)))
)
# integrate from rest with oracle orientation, bias removed or not
for label, bias in (("raw", np.zeros(3)), ("mean-bias removed (oracle)", r.mean(0))):
    v = np.zeros(3)
    x = p[0].copy()
    out = []
    for i in range(N):
        v += (aw[i] - bias) / hz
        x = x + v / hz
        out.append(x.copy())
    out = np.array(out)
    e = np.linalg.norm(out - p, axis=1)
    print(f"oracle-orientation integration {label:28s}: pos err @5s {e[500]:.2f} m  @10s {e[1000]:.2f}  @30s {e[min(3000, N - 1)]:.2f}")
