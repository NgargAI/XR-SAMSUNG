"""Run pathfinder on the first N seconds of a real IMU csv and score with evo.
usage: eval_real.py imu.csv gt_tum.txt [--extra "--kp 1"] [--secs 30 60 120 300 1275] [--degrade]"""

import argparse, subprocess, re, os, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("imu")
ap.add_argument("gt")
ap.add_argument("--bin", default="./pathfinder")
ap.add_argument("--extra", default="")
ap.add_argument("--secs", type=float, nargs="+", default=[30, 60, 120, 300, 1300])
ap.add_argument("--baseline", action="store_true", help="zero the positions (stand-still baseline)")
ap.add_argument("--work", default="/tmp/real/work")
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
lines = open(a.imu).read().splitlines()
hdr, rows = lines[0], lines[1:]
ts = np.array([int(r.split(",", 1)[0]) for r in rows])


def evo(cmd, ref, est):
    o = subprocess.run([cmd[0], "tum", ref, est, "-a", "--t_max_diff", "0.01"] + cmd[1:], capture_output=True, text=True).stdout
    g = lambda k: float(re.search(k + r"\s+([0-9.eE+-]+)", o).group(1)) if re.search(k + r"\s+([0-9.eE+-]+)", o) else float("nan")
    return g("rmse"), g("max")


for S in a.secs:
    n = int(np.searchsorted(ts, ts[0] + S * 1e9))
    n = min(n, len(rows))
    f_imu, f_est, f_gt = (os.path.join(a.work, x) for x in ("imu.csv", "est.txt", "gt.txt"))
    open(f_imu, "w").write("\n".join([hdr] + rows[:n]) + "\n")
    subprocess.run([a.bin, f_imu, f_est] + a.extra.split(), check=True)
    if a.baseline:
        E = np.loadtxt(f_est)
        E[:, 1:4] = 0
        np.savetxt(f_est, E, fmt="%.9f")
    G = np.loadtxt(a.gt)
    t_end = ts[n - 1] * 1e-9
    G = G[(G[:, 0] >= ts[0] * 1e-9) & (G[:, 0] <= t_end)]
    np.savetxt(f_gt, G, fmt="%.6f %.6f %.6f %.6f %.8f %.8f %.8f %.8f")
    ate = evo(["evo_ape"], f_gt, f_est)[0]
    rpe1 = evo(["evo_rpe", "--delta", "100", "--delta_unit", "f"], f_gt, f_est)[0]
    rpe10 = evo(["evo_rpe", "--delta", "1000", "--delta_unit", "f"], f_gt, f_est)[0] if S >= 30 else float("nan")
    print(f"first {S:6.0f}s  rows= {n:6d}  ATE(SE3-aligned) {ate:9.3f} m   RPE@1s {rpe1:7.3f} m   RPE@10s {rpe10:8.3f} m")
