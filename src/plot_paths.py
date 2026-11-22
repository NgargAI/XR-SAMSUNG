#!/usr/bin/env python3
"""Draw the predicted path (and the actual path, if given) to a PNG.

Normally called by ``demo_python.py``. Run on its own after that, it reads the files in ``output/``::

    python plot_paths.py
    python plot_paths.py --est estimated_trajectory.txt --gt gt_tum.txt --out paths.png

Both inputs are TUM format: ``timestamp[s] x y z qx qy qz qw``. With ground truth, the predicted
path is rotated and shifted once onto the actual path (same as ``evo_ape -a``), because the IMU
cannot know where the room's axes are; the error is reported after that single alignment.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

log = logging.getLogger("pathfinder.plot")


class AteResult(TypedDict):
    """Absolute trajectory error after one rigid (SE3) alignment."""

    rmse_m: float  # RMS position error, predicted vs actual
    never_move_m: float  # RMS error of a path that never moves (sits at the mean): the baseline to beat
    n_matched: int  # ground-truth poses used
    duration_s: float


def load_tum(path: Path) -> npt.NDArray[np.float64]:
    """Load a TUM file as an (N, 8) float array, checking shape, finiteness and time order."""
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {path}. Run demo_python.py first (it creates this file).")
    data = np.loadtxt(path, ndmin=2, dtype=np.float64)
    if data.shape[1] != 8:
        raise ValueError(f"{path.name}: expected 8 columns (t x y z qx qy qz qw), got {data.shape[1]}")
    if len(data) < 2:
        raise ValueError(f"{path.name}: need at least 2 poses")
    if not np.isfinite(data).all():
        raise ValueError(f"{path.name}: contains NaN/inf values")
    if np.any(np.diff(data[:, 0]) < 0):
        raise ValueError(f"{path.name}: timestamps are not in increasing order")
    return data


def nearest_indices(t_ref: npt.NDArray[np.float64], t_query: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """Index into sorted ``t_ref`` of the nearest sample for every ``t_query``."""
    i = np.clip(np.searchsorted(t_ref, t_query), 1, len(t_ref) - 1)
    return i - (np.abs(t_ref[i - 1] - t_query) < np.abs(t_ref[i] - t_query))


def umeyama(p: npt.NDArray[np.float64], q: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Best rotation R and shift t so that ``p @ R.T + t`` matches ``q`` (least squares, no scale)."""
    mp, mq = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((q - mq).T @ (p - mp) / len(p))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))  # forbid reflections
    r = u @ d @ vt
    return r, mq - r @ mp


def align(est: npt.NDArray[np.float64], gt: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], AteResult]:
    """Match est to gt by time, align once, and return (aligned predicted xyz, actual xyz, metrics)."""
    gt = gt[(gt[:, 0] >= est[0, 0]) & (gt[:, 0] <= est[-1, 0])]
    if len(gt) < 3:
        raise ValueError("ground truth does not overlap the estimate in time (check the Vicon sync)")
    q = gt[:, 1:4]
    pm = est[nearest_indices(est[:, 0], gt[:, 0]), 1:4]
    r, t = umeyama(pm, q)
    pa = pm @ r.T + t
    err = np.linalg.norm(pa - q, axis=1)
    result = AteResult(
        rmse_m=float(np.sqrt(np.mean(err**2))),
        never_move_m=float(np.sqrt(np.mean(np.sum((q - q.mean(0)) ** 2, 1)))),
        n_matched=len(gt),
        duration_s=float(gt[-1, 0] - est[0, 0]),
    )
    return pa, q, result


def render(est_path: Path, out_path: Path, gt: Path | None = None, secs: float | None = None, title: str = "") -> AteResult | None:
    """Draw the picture to ``out_path``; returns the error metrics when ground truth is given."""
    # Imported here so demo_python --no-plot and the tests of the maths do not pay for matplotlib.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # file output only; works without a display
    import matplotlib.pyplot as plt  # noqa: PLC0415

    est = load_tum(est_path)
    if secs:
        est = est[est[:, 0] <= est[0, 0] + secs]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    red, green = "#d62728", "#2ca02c"

    if gt is None:
        p, t_e = est[:, 1:4], est[:, 0] - est[0, 0]
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(p[:, 0], p[:, 1], color=red, lw=1.5, label="predicted")
        ax[0].plot(*p[0, :2], "ko", label="start")
        ax[0].set_aspect("equal", "datalim")
        ax[0].set_xlabel("x [m]")
        ax[0].set_ylabel("y [m]")
        ax[0].set_title("Predicted path (top view)")
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        for k, n in enumerate("xyz"):
            ax[1].plot(t_e, p[:, k], label=n)
        ax[1].set_xlabel("time [s]")
        ax[1].set_ylabel("position [m]")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        fig.suptitle(title or "IMU-only predicted trajectory")
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        log.info("saved %s", out_path)
        return None

    g = load_tum(gt)
    pa, q, res = align(est, g)
    g = g[(g[:, 0] >= est[0, 0]) & (g[:, 0] <= est[-1, 0])]
    tg = g[:, 0] - est[0, 0]
    err = np.linalg.norm(pa - q, axis=1)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(q[:, 0], q[:, 1], color=green, lw=2.2, label="actual (Vicon)")
    ax.plot(pa[:, 0], pa[:, 1], color=red, lw=1.4, label="predicted (IMU only)")
    ax.plot(*q[0, :2], "ko", ms=8, label="start")
    ax.plot(*q[-1, :2], "ks", ms=8, label="end (actual)")
    ax.plot(*pa[-1, :2], "s", color=red, ms=8, label="end (predicted)")
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top view: actual vs predicted")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    ax2 = fig.add_subplot(gs[0, 1])
    for k, (n, c) in enumerate(zip("xyz", ("tab:blue", "tab:orange", "tab:purple"), strict=True)):
        ax2.plot(tg, q[:, k], color=c, lw=2, alpha=0.45)
        ax2.plot(tg, pa[:, k], color=c, lw=1, ls="--", label=f"{n}: thick = actual, dashed = predicted")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("position [m]")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title("Position over time")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(tg, err, color=red)
    ax3.axhline(res["never_move_m"], color="gray", ls=":", label=f"'never move' baseline RMS = {res['never_move_m']:.2f} m")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("position error [m]")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)
    ax3.set_title(f"Error over time  (RMSE {res['rmse_m']:.2f} m)")

    fig.suptitle(
        title or f"IMU-only path estimation - first {tg[-1]:.0f} s   (ATE RMSE {res['rmse_m']:.2f} m, after one fixed alignment)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info("saved %s   ATE RMSE %.3f m   never-move baseline %.3f m", out_path, res["rmse_m"], res["never_move_m"])
    return res


def positive(text: str) -> float:
    """Parse a finite float > 0 (argparse ``type=``)."""
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not np.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError(f"must be a number > 0: {text!r}")
    return v


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns an exit code (2 on bad input)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--est", default="output/estimated_trajectory.txt", help="predicted path (TUM)")
    ap.add_argument("--gt", help="actual path (TUM); default output/gt_tum.txt if it exists")
    ap.add_argument("--out", default="output/paths.png")
    ap.add_argument("--secs", type=positive, help="only plot the first N seconds")
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    gt = Path(a.gt) if a.gt else (Path("output/gt_tum.txt") if Path("output/gt_tum.txt").is_file() else None)
    try:
        render(Path(a.est), Path(a.out), gt=gt, secs=a.secs, title=a.title)
    except (FileNotFoundError, ValueError) as e:
        log.error("error: %s", e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
