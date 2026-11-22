import sys, numpy as np

E = np.loadtxt(sys.argv[1])
G = np.loadtxt(sys.argv[2])
# associate: nearest est row for each GT row
idx = np.clip(np.searchsorted(E[:, 0], G[:, 0]), 1, len(E) - 1)
idx -= np.abs(E[idx - 1, 0] - G[:, 0]) < np.abs(E[idx, 0] - G[:, 0])
P, Q = E[idx, 1:4], G[:, 1:4]
t = G[:, 0] - G[0, 0]


def umeyama(P, Q):
    mp, mq = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((Q - mq).T @ (P - mp) / len(P))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    return R, mq - R @ mp


def ate(P, Q, still=False):
    if still:
        return np.sqrt(np.mean(np.sum((Q - Q.mean(0)) ** 2, 1)))
    R, tt = umeyama(P, Q)
    return np.sqrt(np.mean(np.sum((Q - (P @ R.T + tt)) ** 2, 1)))


print("window      ATE(est, own SE3 align)   ATE(stand-still)   GT 1s-motion RMS   est 1s-RPE-like")
for a0 in [0, 10, 30, 60, 120, 300, 600, 900, 1200]:
    for L in [30]:
        m = (t >= a0) & (t < a0 + L)
        if m.sum() < 100:
            continue
        R, tt = umeyama(P[m], Q[m])
        d1 = lambda X: np.sqrt(np.mean(np.sum((X[100:] - X[:-100]) ** 2, 1)))
        Pa = P[m] @ R.T + tt
        print(
            f"{a0:5d}-{a0 + L:<5d} {ate(P[m], Q[m]):8.3f} m            {ate(P[m], Q[m], True):8.3f} m       {d1(Q[m]):8.3f} m      {np.sqrt(np.mean(np.sum(((Pa[100:] - Pa[:-100]) - (Q[m][100:] - Q[m][:-100])) ** 2, 1))):8.3f} m"
        )
