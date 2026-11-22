#!/usr/bin/env python3
"""Learned speed model: train on the first part of a recording, test on the rest.

The plain estimator (demo_python.py) gets orientation right but cannot get speed right from a
phone accelerometer, so its position drifts. This script learns speed from IMU patterns instead
(the RoNIN idea), using Vicon as the teacher:

1. Run the estimator on the whole recording to get orientation (and the "current code" path).
2. Every 0.05 s, describe the last 2 s of IMU readings with 134 numbers (features).
3. Teacher: the true velocity in the phone's own frame, from Vicon.
4. Train a small neural network (MLP) on the first ``--train-frac`` of the recording only.
5. On the remaining part (never seen in training) predict velocity, rotate it with the
   estimator's orientation, add it up into a path, and compare with Vicon.

Usage (same folder as demo_python.py, plot_paths.py and the two csv files)::

    pip install scikit-learn
    python train_and_test.py                    # 90 % train / 10 % test
    python train_and_test.py --train-frac 0.8   # 80 % / 20 %

Outputs in ``output/``: ``learned_test.png`` (actual vs learned vs current code on the test part),
``speed_true_vs_predicted.png``,
``learned_test_trajectory.txt`` (TUM), ``gt_tum.txt``.

Limits: it needs Vicon to train, so it is NOT usable for the contest's IMU-only Track 3
submission; it learns this phone and this kind of motion and will not transfer to other IMUs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import numpy as np

import demo_python as dp
import plot_paths as pp

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

log = logging.getLogger("pathfinder.learned")

HZ = 20.0  # feature / prediction rate
WINDOW_S = 2.0  # IMU history per feature vector
N_BINS = 20  # time bins inside the window
IMU_HZ = 200  # expected IMU rate (window length in samples = WINDOW_S * IMU_HZ)
SPEED_GATE = 0.05  # predicted speeds below this [m/s] are treated as standing still
GRAVITY = 9.81


class Dataset(NamedTuple):
    """Feature/target table at HZ, aligned with the estimator and Vicon."""

    t: npt.NDArray[np.float64]  # seconds since the first feature
    features: npt.NDArray[np.float64]  # (N, 134)
    v_body: npt.NDArray[np.float64]  # (N, 3) true velocity in the phone frame (teacher)
    r_world_body: npt.NDArray[np.float64]  # (N, 3, 3) estimator orientation
    p_gt: npt.NDArray[np.float64]  # (N, 3) Vicon position
    p_current: npt.NDArray[np.float64]  # (N, 3) current estimator path (with drift removal)
    t_abs: npt.NDArray[np.float64]  # absolute timestamps [s] (IMU clock)


class Scores(TypedDict):
    """Position error [m] after one rigid alignment."""

    learned: float
    current: float
    never_move: float
    windows: int


def quat_to_rot(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(N, 4) quaternions (w, x, y, z) -> (N, 3, 3) rotation matrices world<-body."""
    w, x, y, z = q.T
    r = np.empty((len(q), 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def window_features(
    gyro: npt.NDArray[np.float64], accel: npt.NDArray[np.float64], up_body: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """134 numbers describing one IMU window (gyro, accel: (n, 3); up_body: world-up in the phone frame)."""
    lin = accel - GRAVITY * up_body  # accelerometer with gravity removed (using the estimator's tilt)
    return np.concatenate(
        [
            gyro.reshape(N_BINS, -1, 3).mean(1).ravel(),  # rotation-rate shape over the window
            lin.reshape(N_BINS, -1, 3).mean(1).ravel(),  # motion-acceleration shape
            gyro.std(0),
            accel.std(0),
            up_body,
            lin.sum(0) / IMU_HZ,  # velocity change over the window
            [np.linalg.norm(gyro, axis=1).mean(), np.linalg.norm(lin, axis=1).mean()],
        ]
    )


def build_dataset(imu_path: Path, gt_path: Path) -> Dataset:
    """Run the estimator, build features and the Vicon teacher signal."""
    samples = list(dp.read_imu(imu_path))
    good = [s for s in samples if s.valid]
    if len(good) < len(samples):
        raise dp.InputError("training needs a clean recording: the IMU file has invalid rows")
    ins = dp.run_estimator(samples, dp.Config())
    current = dp.remove_drift(ins.poses, 2.0)

    ti = np.array([s.t_ns for s in samples], dtype=np.float64) * 1e-9
    dt = np.median(np.diff(ti))
    if not 0.004 < dt < 0.006:
        raise dp.InputError(f"expected a ~200 Hz IMU, median sample period is {dt * 1e3:.2f} ms")
    gyro = np.array([s.gyro for s in samples])
    accel = np.array([s.accel for s in samples])
    quats = np.array([p.orientation for p in ins.poses])
    p_cur_all = np.array([p.position for p in current])

    gt = pp.load_tum(gt_path)
    ts = np.arange(ti[0] + WINDOW_S, min(ti[-1], gt[-1, 0]) - 0.2, 1 / HZ)
    idx = np.searchsorted(ti, ts)
    n_win = int(WINDOW_S * IMU_HZ)
    r_wb = quat_to_rot(quats[idx])
    up_body = r_wb[:, 2, :]  # third row of R(world<-body) = world z expressed in the body frame

    def gt_pos(tt: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.stack([np.interp(tt, gt[:, 0], gt[:, k]) for k in (1, 2, 3)], 1)

    v_world = (gt_pos(ts + 0.1) - gt_pos(ts - 0.1)) / 0.2
    gi = np.clip(np.searchsorted(gt[:, 0], ts), 0, len(gt) - 1)
    r_gt = quat_to_rot(gt[gi][:, [7, 4, 5, 6]])  # TUM stores qx qy qz qw
    v_body = np.einsum("nji,nj->ni", r_gt, v_world)  # Vicon marker frame ~= phone frame on this rig

    feats = np.stack([window_features(gyro[i - n_win : i], accel[i - n_win : i], up_body[k]) for k, i in enumerate(idx)])
    return Dataset(ts - ts[0], feats, v_body, r_wb, gt_pos(ts), p_cur_all[idx], ts)


def train_speed_model(features: npt.NDArray[np.float64], v_body: npt.NDArray[np.float64]) -> list[object]:
    """One small MLP per velocity axis (deterministic: fixed random_state)."""
    from sklearn.neural_network import MLPRegressor  # noqa: PLC0415  # optional dependency, only needed here
    from sklearn.pipeline import make_pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    models: list[object] = []
    for k in range(3):
        m = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-2, early_stopping=True, max_iter=300, random_state=0),
        )
        m.fit(features, v_body[:, k])
        models.append(m)
    return models


def predict_velocity(models: Sequence[object], features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Predicted phone-frame velocity; tiny predicted speeds are zeroed (standing still)."""
    v: npt.NDArray[np.float64] = np.stack([m.predict(features) for m in models], 1)  # type: ignore[attr-defined]  # sklearn pipelines
    v[np.linalg.norm(v, axis=1) < SPEED_GATE] = 0.0
    return v


def integrate_path(v_body: npt.NDArray[np.float64], r_wb: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Rotate phone-frame velocity into the world with the estimator's orientation and add it up."""
    v_world: npt.NDArray[np.float64] = np.einsum("nij,nj->ni", r_wb, v_body)
    return np.cumsum(v_world, 0) / HZ


def aligned_error(p: npt.NDArray[np.float64], q: npt.NDArray[np.float64]) -> float:
    """RMS position error after one rigid alignment of p onto q."""
    r, t = pp.umeyama(p, q)
    return float(np.sqrt(np.mean(np.sum((p @ r.T + t - q) ** 2, 1))))


def window_scores(
    t: npt.NDArray[np.float64],
    learned: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
    gt: npt.NDArray[np.float64],
    length: float,
    *,
    moving_only: bool = False,
) -> Scores:
    """Average error over consecutive windows of ``length`` seconds, each aligned on its own.

    ``moving_only`` keeps only windows where the phone really travelled (> 0.8 m extent).
    """
    rows = []
    for s0 in np.arange(t[0], t[-1] - length + 1e-9, length):
        s = (t >= s0) & (t < s0 + length)
        q = gt[s]
        if moving_only and np.linalg.norm(np.ptp(q, 0)) < 0.8:
            continue
        rows.append((aligned_error(learned[s], q), aligned_error(current[s], q), float(np.sqrt(np.mean(np.sum((q - q.mean(0)) ** 2, 1))))))
    if not rows:
        raise dp.InputError(f"no {length:.0f} s test windows{' with movement' if moving_only else ''}")
    a = np.mean(rows, 0)
    return Scores(learned=float(a[0]), current=float(a[1]), never_move=float(a[2]), windows=len(rows))


def draw(  # noqa: PLR0917  # one call site; all seven inputs are needed for the picture
    out: Path,
    t: npt.NDArray[np.float64],
    learned: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
    gt: npt.NDArray[np.float64],
    whole: Scores,
    title: str,
) -> None:
    """Top view + error-over-time picture for the test part."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    def align_to_gt(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r, tt = pp.umeyama(p, gt)
        return p @ r.T + tt

    la, ca = align_to_gt(learned), align_to_gt(current)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.1, 1]})
    ax.plot(gt[:, 0], gt[:, 1], color="#2ca02c", lw=2.4, label="actual (Vicon)")
    ax.plot(la[:, 0], la[:, 1], color="#1f77b4", lw=1.4, label=f"learned model ({whole['learned']:.2f} m)")
    ax.plot(ca[:, 0], ca[:, 1], color="#d62728", lw=1.0, alpha=0.7, label=f"current code ({whole['current']:.2f} m)")
    ax.plot(*gt[0, :2], "ko", ms=7, label="start")
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top view, test part only (never seen in training)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax2.plot(t - t[0], np.linalg.norm(la - gt, axis=1), color="#1f77b4", label="learned model")
    ax2.plot(t - t[0], np.linalg.norm(ca - gt, axis=1), color="#d62728", alpha=0.7, label="current code")
    ax2.axhline(whole["never_move"], color="gray", ls=":", label=f"never-move baseline ({whole['never_move']:.2f} m)")
    ax2.set_xlabel("time into test part [s]")
    ax2.set_ylabel("position error [m]")
    ax2.set_title("Error over time")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def draw_speed(out: Path, t: npt.NDArray[np.float64], v_true: npt.NDArray[np.float64], v_pred: npt.NDArray[np.float64], title: str) -> None:
    """Draw true vs predicted speed over the test part (lightly smoothed, 0.25 s)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    k = np.ones(5) / 5
    st = np.convolve(np.linalg.norm(v_true, axis=1), k, "same")
    sp = np.convolve(np.linalg.norm(v_pred, axis=1), k, "same")
    tt = t - t[0]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(tt, st, color="#2ca02c", lw=2.2, label="true speed (Vicon)")
    ax.plot(tt, sp, color="#1f77b4", lw=1.4, label="predicted speed (learned model)")
    ax.set_xlabel("time into test part [s]")
    ax.set_ylabel("speed [m/s]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fraction(text: str) -> float:
    """Parse a train fraction strictly between 0.1 and 0.95 (argparse ``type=``)."""
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0.1 <= v <= 0.95:
        raise argparse.ArgumentTypeError("--train-frac must be between 0.1 and 0.95")
    return v


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: returns 0 on success, 2 on bad input."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imu", nargs="?", default="gyro_accel.csv")
    ap.add_argument("--vicon", default="vicon_capture_quaternion.csv")
    ap.add_argument("--vicon-hz", type=float, default=100.0)
    ap.add_argument("--train-frac", type=fraction, default=0.9, help="first fraction used for training (default 0.9)")
    ap.add_argument("--out", default="output")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    try:
        try:
            import sklearn  # noqa: F401, PLC0415
        except ImportError:
            raise dp.InputError("this script needs scikit-learn: pip install scikit-learn") from None
        imu_path = dp.resolve_input_file(a.imu, "IMU")
        vicon_path = dp.resolve_input_file(a.vicon, "Vicon")
        out = Path(a.out).expanduser().resolve()
        gt_path = out / "gt_tum.txt"
        sync = dp.vicon_to_tum(list(dp.read_imu(imu_path)), vicon_path, gt_path, a.vicon_hz)
        log.info("Vicon<->IMU time offset %.3f s (correlation %.2f)", sync.offset_s, sync.correlation)

        log.info("building features (runs the estimator over the whole recording)...")
        d = build_dataset(imu_path, gt_path)
        cut = a.train_frac * (d.t[-1] + WINDOW_S + 0.2)
        train, test = d.t < cut, d.t >= cut
        if test.sum() < 60 * HZ:
            raise dp.InputError("the test part is shorter than 60 s; lower --train-frac")
        log.info("training on the first %.0f s (%.0f%%), testing on the last %.0f s", cut, a.train_frac * 100, d.t[-1] - cut)
        models = train_speed_model(d.features[train], d.v_body[train])

        v = predict_velocity(models, d.features[test])
        learned = integrate_path(v, d.r_world_body[test])
        t, gt, cur = d.t[test], d.p_gt[test], d.p_current[test]
        whole = Scores(
            learned=aligned_error(learned, gt),
            current=aligned_error(cur, gt),
            never_move=float(np.sqrt(np.mean(np.sum((gt - gt.mean(0)) ** 2, 1)))),
            windows=1,
        )
        log.info("\nTEST part (never used for training), position error after one alignment:")
        log.info(
            "  %-20s learned %.3f m   current code %.3f m   never-move %.3f m",
            f"whole {t[-1] - t[0]:.0f} s",
            whole["learned"],
            whole["current"],
            whole["never_move"],
        )
        for length in (30.0, 60.0):
            if t[-1] - t[0] >= length:
                s = window_scores(t, learned, cur, gt, length)
                log.info(
                    "  %-20s learned %.3f m   current code %.3f m   never-move %.3f m",
                    f"{length:.0f} s windows (n={s['windows']})",
                    s["learned"],
                    s["current"],
                    s["never_move"],
                )
        try:
            s = window_scores(t, learned, cur, gt, 10.0, moving_only=True)
            log.info(
                "  %-20s learned %.3f m   current code %.3f m   never-move %.3f m",
                f"moving 10 s (n={s['windows']})",
                s["learned"],
                s["current"],
                s["never_move"],
            )
        except dp.InputError:
            log.info("  (no 10 s test windows with more than 0.8 m of movement)")

        lines = "".join(
            "%.9f %.6f %.6f %.6f 0 0 0 1\n" % (ta, *p) for ta, p in zip(d.t_abs[test], learned, strict=True)
        )  # orientation column unused for this comparison
        dp.atomic_write_text(out / "learned_test_trajectory.txt", lines)
        png = out / "learned_test.png"
        draw(
            png,
            t,
            learned,
            cur,
            gt,
            whole,
            f"Learned speed model - trained on first {a.train_frac:.0%}, tested on last {1 - a.train_frac:.0%}",
        )
        sp_png = out / "speed_true_vs_predicted.png"
        fast_train = float(np.sum(np.linalg.norm(d.v_body[train], axis=1) > 0.5)) / HZ
        draw_speed(
            sp_png,
            t,
            d.v_body[test],
            v,
            f"True vs predicted speed on the test part (training data had only {fast_train:.1f} s above 0.5 m/s)",
        )
        log.info("\nDone. Open %s and %s", png, sp_png)
    except dp.InputError as e:
        log.error("error: %s", e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
