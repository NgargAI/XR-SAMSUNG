"""Orientation vs Vicon on the full run, and position on 60 s clips taken from later in the log."""

import sys, subprocess, numpy as np

IMU, GT, BIN = sys.argv[1], sys.argv[2], "./pathfinder"
G = np.loadtxt(GT)


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


def assoc(E, tg):
    i = np.clip(np.searchsorted(E[:, 0], tg), 1, len(E) - 1)
    i -= np.abs(E[i - 1, 0] - tg) < np.abs(E[i, 0] - tg)
    return i


def ang(R):
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def umeyama(P, Q):
    mp, mq = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((Q - mq).T @ (P - mp) / len(P))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    return R, mq - R @ mp


# ---------- 1) orientation, whole run
subprocess.run([BIN, IMU, "/tmp/real/o.est"], check=True)
E = np.loadtxt("/tmp/real/o.est")
i = assoc(E, G[:, 0])
Re = qR(E[i, 4:8])
Rg = qR(G[:, 4:8])  # marker frame ~= imu frame (extrinsic ~ I, see oracle_check)
t = G[:, 0] - G[0, 0]
M = np.einsum("nij,nkj->ik", Rg[t < 5], Re[t < 5])
U, S, Vt = np.linalg.svd(M)
C = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt  # world_est -> world_vicon (fit on first 5 s)
Rerr = np.einsum("ij,njk->nik", C, Re)
rel = np.einsum("nji,njk->nik", Rg, Rerr)  # body-frame error rotation
tot = ang(rel)
# tilt error = angle between estimated and true gravity direction in body frame
up_e = np.einsum("nji,j->ni", Rerr, [0, 0, 1])
up_g = np.einsum("nji,j->ni", Rg, [0, 0, 1])
tilt = np.degrees(np.arccos(np.clip(np.sum(up_e * up_g, 1), -1, 1)))
print("ORIENTATION vs Vicon (one fixed world-frame offset fitted on the first 5 s):")
for a, b in [(0, 10), (10, 60), (60, 300), (300, 900), (900, 1300)]:
    m = (t >= a) & (t < b)
    print(
        f"  {a:5d}-{b:<5d}s  total {tot[m].mean():6.2f} deg (median {np.median(tot[m]):5.2f})   tilt(roll/pitch) {tilt[m].mean():5.2f} deg"
    )
# ---------- 2) position on 60 s clips from later in the log
lines = open(IMU).read().splitlines()
hdr, rows = lines[0], lines[1:]
ts = np.array([int(r.split(",", 1)[0]) for r in rows]) * 1e-9
print("\nPOSITION on 60 s clips started at various times (default settings; each clip starts from rest-assumption):")
print(f"  {'start':>6s} {'GT moved(extent)':>17s} {'ATE est':>8s} {'stand-still':>12s} {'est end-pos err':>16s} {'tilt err':>9s}")
for s0 in [150, 300, 450, 600, 750, 900, 1050, 1200]:
    a, b = np.searchsorted(ts, ts[0] + s0), np.searchsorted(ts, ts[0] + s0 + 60)
    open("/tmp/real/clip.csv", "w").write("\n".join([hdr] + rows[a:b]) + "\n")
    subprocess.run([BIN, "/tmp/real/clip.csv", "/tmp/real/clip.est"], check=True)
    Ec = np.loadtxt("/tmp/real/clip.est")
    m = (G[:, 0] >= Ec[0, 0]) & (G[:, 0] <= Ec[-1, 0])
    g = G[m]
    ii = assoc(Ec, g[:, 0])
    P, Q = Ec[ii, 1:4], g[:, 1:4]
    R, tt = umeyama(P, Q)
    err = np.linalg.norm(P @ R.T + tt - Q, axis=1)
    ext = np.ptp(Q, axis=0)
    st = np.sqrt(np.mean(np.sum((Q - Q.mean(0)) ** 2, 1)))
    Rc = qR(Ec[ii, 4:8])
    Rgc = qR(g[:, 4:8])
    Mx = np.einsum("nij,nkj->ik", Rgc[:100], Rc[:100])
    U, S, Vt = np.linalg.svd(Mx)
    Cx = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    ue = np.einsum("nji,j->ni", np.einsum("ij,njk->nik", Cx, Rc), [0, 0, 1])
    ug = np.einsum("nji,j->ni", Rgc, [0, 0, 1])
    tl = np.degrees(np.arccos(np.clip(np.sum(ue * ug, 1), -1, 1))).mean()
    print(f"  {s0:5d}s {np.linalg.norm(ext):14.2f} m {np.sqrt(np.mean(err**2)):7.2f}m {st:11.2f}m {err[-1]:15.2f}m {tl:8.2f}")
