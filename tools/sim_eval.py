#!/usr/bin/env python3
"""Synthetic degraded-IMU benchmark for pathfinder (no dataset download needed).

Generates a smooth 6-DoF flight, converts it to EuRoC-style IMU readings, applies the
Track-3 style degradation (growing noise -> bias drift -> dropped frames -> total
failure), runs the estimator and prints ATE / blind-phase error.

    python3 tools/sim_eval.py --bin ./pathfinder --out /tmp/sim [--fail-mode zeros|nan|freeze]

Ground truth is written in TUM format so you can also run:
    evo_ape tum /tmp/sim/gt_tum.txt /tmp/sim/estimated_trajectory.txt -a
"""

import argparse, os, subprocess, time
import numpy as np


def rot_zyx(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    R = np.empty(yaw.shape + (3, 3))
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    return R


def R_to_q(R):  # returns x y z w
    q = np.empty((R.shape[0], 4))
    for i, M in enumerate(R):
        t = np.trace(M)
        if t > 0:
            s = 2 * np.sqrt(t + 1)
            w = s / 4
            x = (M[2, 1] - M[1, 2]) / s
            y = (M[0, 2] - M[2, 0]) / s
            z = (M[1, 0] - M[0, 1]) / s
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = 2 * np.sqrt(1 + M[0, 0] - M[1, 1] - M[2, 2])
            w = (M[2, 1] - M[1, 2]) / s
            x = s / 4
            y = (M[0, 1] + M[1, 0]) / s
            z = (M[0, 2] + M[2, 0]) / s
        elif M[1, 1] > M[2, 2]:
            s = 2 * np.sqrt(1 + M[1, 1] - M[0, 0] - M[2, 2])
            w = (M[0, 2] - M[2, 0]) / s
            x = (M[0, 1] + M[1, 0]) / s
            y = s / 4
            z = (M[1, 2] + M[2, 1]) / s
        else:
            s = 2 * np.sqrt(1 + M[2, 2] - M[0, 0] - M[1, 1])
            w = (M[1, 0] - M[0, 1]) / s
            x = (M[0, 2] + M[2, 0]) / s
            y = (M[1, 2] + M[2, 1]) / s
            z = s / 4
        q[i] = (x, y, z, w)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./pathfinder")
    ap.add_argument("--out", default="/tmp/sim")
    ap.add_argument("--rate", type=float, default=200.0)
    ap.add_argument("--dur", type=float, default=60.0)
    ap.add_argument("--fail-mode", default="zeros", choices=["zeros", "nan", "freeze", "noise0"])
    ap.add_argument("--clean", action="store_true", help="no degradation at all (sanity check)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--extra", default="", help="extra pathfinder options as one quoted string")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.join(a.out, "imu0"), exist_ok=True)

    dt = 1.0 / a.rate
    t = np.arange(0, a.dur, dt)
    n = len(t)
    # ---- ground truth motion (MAV-like: ~1 m/s, gentle attitude changes), starts at rest
    ramp = np.clip((t - 3.0) / 3.0, 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)
    p = np.stack([2.0 * np.sin(0.35 * t), 1.5 * np.sin(0.5 * t + 0.3) - 1.5 * np.sin(0.3), 0.4 * np.sin(0.7 * t)], 1) * ramp[:, None]
    yaw = 0.6 * np.sin(0.2 * t) * ramp
    pitch = 0.15 * np.sin(0.45 * t) * ramp
    roll = 0.15 * np.sin(0.6 * t + 1.0) * ramp
    R = rot_zyx(yaw, pitch, roll)
    v = np.gradient(p, dt, axis=0)
    acc = np.gradient(v, dt, axis=0)
    # body angular rate: skew(w) = R^T dR/dt
    dR = np.gradient(R, dt, axis=0)
    S = np.einsum("nji,njk->nik", R, dR)
    wb = np.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], 1)
    fb = np.einsum("nji,nj->ni", R, acc + np.array([0, 0, 9.80665]))

    # ---- sensor model + degradation schedule (seconds)
    t_noise, t_drift, t_drop0, t_drop1, t_fail = 0.30 * a.dur, 0.45 * a.dur, 0.55 * a.dur, 0.70 * a.dur, a.dur - 5.0
    sg0, sa0 = 0.002, 0.02
    scale = np.where(t > t_noise, 1 + 9 * np.clip((t - t_noise) / 5.0, 0, 1), 1.0)
    bg = np.cumsum(rng.normal(0, 2e-5, (n, 3)), 0) + rng.normal(0, 0.003, 3)
    ba = np.cumsum(rng.normal(0, 2e-4, (n, 3)), 0) + rng.normal(0, 0.02, 3)
    drift = np.clip((t - t_drift) / 10.0, 0, 1)[:, None]
    if a.clean:
        scale = np.ones(n)
        drift = 0 * drift
        bg = bg * 0
        ba = ba * 0
    bg = bg + drift * np.array([0.002, -0.0015, 0.001])
    ba = ba + drift * np.array([0.02, -0.015, 0.01])
    gyr = wb + bg + rng.normal(0, 1, (n, 3)) * sg0 * scale[:, None]
    acl = fb + ba + rng.normal(0, 1, (n, 3)) * sa0 * scale[:, None]

    keep = np.ones(n, bool)
    win = (t > t_drop0) & (t < t_drop1)
    keep[win] = rng.random(win.sum()) > 0.25  # 25 % frame loss
    for s0 in rng.uniform(t_drop0, t_drop1, 3):  # plus a few long gaps
        keep[(t > s0) & (t < s0 + 0.08)] = False
    fail = t >= t_fail
    if a.clean:
        keep[:] = True
        fail = t > 1e9
    if a.fail_mode == "zeros":
        gyr[fail] = 0
        acl[fail] = 0
    elif a.fail_mode == "nan":
        gyr[fail] = np.nan
        acl[fail] = np.nan
    elif a.fail_mode == "freeze":
        i0 = np.argmax(fail)
        gyr[fail] = gyr[i0]
        acl[fail] = acl[i0]
    else:
        gyr[fail] = rng.normal(0, 0.01, (fail.sum(), 3))
        acl[fail] = rng.normal(0, 0.05, (fail.sum(), 3))
    keep[fail] = True

    ts = (1.403636580e18 + t * 1e9).astype(np.int64)
    with open(os.path.join(a.out, "imu0", "data.csv"), "w") as f:
        f.write(
            "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
            "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
        )
        for i in np.flatnonzero(keep):
            f.write("%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n" % (ts[i], *gyr[i], *acl[i]))
    q = R_to_q(R)
    with open(os.path.join(a.out, "gt_tum.txt"), "w") as f:
        for i in np.flatnonzero(keep):
            f.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" % (ts[i] * 1e-9, *p[i], *q[i]))

    # ---- run
    est = os.path.join(a.out, "estimated_trajectory.txt")
    t0 = time.perf_counter()
    subprocess.run([a.bin, os.path.join(a.out, "imu0", "data.csv"), est] + a.extra.split(), check=True)
    wall = time.perf_counter() - t0

    E = np.loadtxt(est)
    G = np.loadtxt(os.path.join(a.out, "gt_tum.txt"))
    assert len(E) == len(G), f"pose count {len(E)} != frames {len(G)}"
    err = np.linalg.norm(E[:, 1:4] - G[:, 1:4], axis=1)
    tt = G[:, 0] - G[0, 0]
    seg = lambda m: np.sqrt(np.mean(err[m] ** 2))
    print(f"frames={len(E)}  wall={wall * 1e3:.1f} ms for {a.dur:.0f}s of data ({wall / a.dur * 1e3:.2f} ms/s incl. python spawn)")
    print(f"ATE RMSE (same origin, no alignment) all : {seg(tt >= 0):.3f} m")
    print(f"  healthy      [0,{t_noise:.0f}]s     : {seg(tt < t_noise):.3f} m")
    print(f"  noisy/drift  [{t_noise:.0f},{t_drop0:.0f}]s  : {seg((tt >= t_noise) & (tt < t_drop0)):.3f} m")
    print(f"  drops        [{t_drop0:.0f},{t_drop1:.0f}]s  : {seg((tt >= t_drop0) & (tt < t_drop1)):.3f} m")
    i_f = np.argmax(tt >= t_fail) if (tt >= t_fail).any() else len(tt) - 1
    print(
        f"  blind last 5s                    : {seg(tt >= t_fail):.3f} m   (err at IMU death {err[i_f]:.2f} m -> end {err[-1]:.2f} m, growth {err[-1] - err[i_f]:.2f} m)"
    )
    # attitude error
    qe, qg = E[:, 4:8], G[:, 4:8]
    ang = 2 * np.degrees(np.arccos(np.clip(np.abs(np.sum(qe * qg, 1)), 0, 1)))
    print(f"attitude error mean/max: {ang.mean():.2f} / {ang.max():.2f} deg")


if __name__ == "__main__":
    main()
