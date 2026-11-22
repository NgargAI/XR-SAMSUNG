import sys, numpy as np

sys.argv, (est, gt) = sys.argv[:1], sys.argv[1:3]
E = np.loadtxt(est)
G = np.loadtxt(gt)
G = G[(G[:, 0] >= E[0, 0]) & (G[:, 0] <= E[-1, 0])]
t = G[:, 0] - E[0, 0]
T = E[-1, 0] - E[0, 0]
idx = np.clip(np.searchsorted(E[:, 0], G[:, 0]), 1, len(E) - 1)
idx -= np.abs(E[idx - 1, 0] - G[:, 0]) < np.abs(E[idx, 0] - G[:, 0])
P, Q = E[idx, 1:4], G[:, 1:4]
mp, mq = P.mean(0), Q.mean(0)
U, S, Vt = np.linalg.svd((Q - mq).T @ (P - mp) / len(P))
D = np.eye(3)
D[2, 2] = np.sign(np.linalg.det(U @ Vt))
R = U @ D @ Vt
Pa = (P - mp) @ R.T + mq
err = np.linalg.norm(Pa - Q, axis=1)
rms = lambda m: np.sqrt(np.mean(err[m] ** 2))
st = lambda m: np.sqrt(np.mean(np.sum((Q[m] - Q[m].mean(0)) ** 2, 1)))
ph = {
    "healthy 0-30%": (t < 0.30 * T),
    "noisy 30-45%": (t >= 0.30 * T) & (t < 0.45 * T),
    "drift 45-55%": (t >= 0.45 * T) & (t < 0.55 * T),
    "drops 55-70%": (t >= 0.55 * T) & (t < 0.70 * T),
    "late 70-95%": (t >= 0.70 * T) & (t < T - 5),
    "BLIND last 5s": (t >= T - 5),
}
print(f"{'phase':14s} {'ATE(est)':>9s} {'stand-still':>12s}")
for k, m in ph.items():
    print(f"{k:14s} {rms(m):8.2f}m {st(m):11.2f}m")
print(
    f"overall ATE {np.sqrt(np.mean(err**2)):.2f} m ; blind-phase path: est moved {np.linalg.norm(Pa[t >= T - 5][-1] - Pa[t >= T - 5][0]):.2f} m, GT moved {np.linalg.norm(Q[t >= T - 5][-1] - Q[t >= T - 5][0]):.2f} m"
)
