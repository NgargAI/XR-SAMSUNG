#!/usr/bin/env python3
"""Path Finder demo: IMU-only path estimation + comparison with Vicon, in pure Python.

Reads accelerometer + gyroscope readings, estimates the 6-DoF path by strapdown dead
reckoning, writes ``estimated_trajectory.txt`` (TUM format) and draws ``paths.png``
(actual path vs predicted path) via ``plot_paths.py``.

Usage (VS Code terminal, run from the folder holding this file)::

    python demo_python.py                                   # uses gyro_accel.csv (+ vicon csv if present)
    python demo_python.py my_imu.csv --vicon my_vicon.csv --secs 60

IMU csv columns: ``timestamp[ns], gx, gy, gz [rad/s], ax, ay, az [m/s^2]``; one header line is fine.

This is a line-for-line port of ``pathfinder.cpp`` (identical results). It is slower and uses far
more memory than the C++ binary, so submit the C++ version for the contest's 8 MB / 20 ms limits.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import plot_paths  # local module next to this file; imports matplotlib only when drawing

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    import numpy.typing as npt

log = logging.getLogger("pathfinder")

G0 = 9.80665  # standard gravity [m/s^2]

# ======================================================================================
#  Types
# ======================================================================================
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z)


class ImuSample(NamedTuple):
    """One IMU row. ``valid`` is False when any of the six numbers was missing or not finite."""

    t_ns: int
    gyro: Vec3
    accel: Vec3
    valid: bool


class Pose(NamedTuple):
    """One output pose: IMU body pose in the estimator's world frame (z up, yaw 0 at start)."""

    t_ns: int
    position: Vec3
    orientation: Quat


class InputError(Exception):
    """A problem with the user's input files or options; reported without a traceback."""


@dataclass(frozen=True)
class Config:
    """Estimator tuning. Defaults were tuned on the supplied phone + Vicon recording."""

    kp: float = 4.0  # gravity-correction gain for roll/pitch [1/s]
    ki: float = 1.0  # gyro-bias learning gain
    acc_tol: float = 1.0  # |a|-g tolerance for the gravity correction [m/s^2]
    zupt: bool = True  # zero-velocity updates when the IMU looks still
    zupt_speed: float = 3.0  # only zero the velocity if slower than this [m/s]
    zupt_time: float = 0.15  # IMU must look still for this long [s]
    vleak: float = 0.2  # velocity leak while healthy [1/s] (bounded-motion prior); 0 = off
    dead_decay: float = 0.0  # velocity decay after the IMU dies [1/s]; 0 = constant velocity
    vmax: float = 40.0  # sanity clamp [m/s]
    stuck_n: int = 20  # identical consecutive rows => sensor frozen
    bad_n: int = 20  # invalid consecutive rows => sensor dead
    init_time: float = 0.6  # stationary start-up window [s]
    max_init: int = 400  # cap on start-up samples

    def __post_init__(self) -> None:
        """Reject values that would make the filter meaningless (negative gains/times)."""
        for name in ("kp", "ki", "acc_tol", "zupt_speed", "zupt_time", "vleak", "dead_decay", "vmax", "init_time"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise InputError(f"--{name.replace('_', '-')} must be a finite number >= 0 (got {value})")
        if self.acc_tol == 0 or self.vmax == 0:
            raise InputError("acc_tol and vmax must be > 0")
        if self.stuck_n < 2 or self.bad_n < 1 or self.max_init < 1:
            raise InputError("stuck_n must be >= 2, bad_n and max_init >= 1")


# ======================================================================================
#  Vector / quaternion maths (plain tuples: no numpy in the hot loop, results match C++)
# ======================================================================================
def add(a: Vec3, b: Vec3) -> Vec3:
    """Vector sum."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    """Vector difference a - b."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Vec3, s: float) -> Vec3:
    """Scale a vector."""
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    """Dot product."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Vec3) -> float:
    """Euclidean length."""
    return math.sqrt(dot(a, a))


def cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product a x b."""
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp v into [lo, hi]."""
    return lo if v < lo else (hi if v > hi else v)


def qmul(a: Quat, b: Quat) -> Quat:
    """Hamilton product of (w, x, y, z) quaternions."""
    return (
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


def qnormalize(q: Quat) -> Quat:
    """Return the unit quaternion; a degenerate (near-zero) input becomes the identity."""
    n = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def qexp(r: Vec3) -> Quat:
    """Rotation vector (axis * angle, rad) -> unit quaternion."""
    a = norm(r)
    if a < 1e-10:
        return qnormalize((1.0, r[0] * 0.5, r[1] * 0.5, r[2] * 0.5))
    s = math.sin(0.5 * a) / a
    return (math.cos(0.5 * a), r[0] * s, r[1] * s, r[2] * s)


def qrot(q: Quat, v: Vec3) -> Vec3:
    """Rotate v from body to world frame."""
    qv = (q[1], q[2], q[3])
    t = mul(cross(qv, v), 2.0)
    return add(add(v, mul(t, q[0])), cross(qv, t))


def qrot_inv(q: Quat, v: Vec3) -> Vec3:
    """Rotate v from world to body frame."""
    return qrot((q[0], -q[1], -q[2], -q[3]), v)


# ======================================================================================
#  Estimator (port of pathfinder.cpp)
# ======================================================================================
def is_plausible(s: ImuSample) -> bool:
    """Return True for a reading usable for integration: finite, not zeroed, not absurd."""
    if not s.valid:
        return False
    an, wn = norm(s.accel), norm(s.gyro)
    return 2.0 < an < 160.0 and wn < 35.0


class StrapdownEstimator:
    """Streaming strapdown INS with gravity aiding, ZUPT and IMU-failure handling.

    Feed samples in file order with :meth:`push`, call :meth:`finish` once, then read
    :attr:`poses` (exactly one pose per pushed sample).
    """

    def __init__(self, cfg: Config) -> None:
        """Create an estimator in its pre-initialisation state."""
        self.c = cfg
        self.poses: list[Pose] = []
        self.dead_since_ns: int | None = None  # when the IMU was declared dead (for reporting)
        self._inited = False
        self._init_buf: list[ImuSample] = []
        self.q: Quat = (1.0, 0.0, 0.0, 0.0)
        self.p: Vec3 = (0.0, 0.0, 0.0)
        self.v: Vec3 = (0.0, 0.0, 0.0)
        self.bg: Vec3 = (0.0, 0.0, 0.0)  # gyro bias
        self.ei: Vec3 = (0.0, 0.0, 0.0)  # integral term of the gravity correction
        self.g = G0
        self.dt0 = 0.005
        self.tlast = 0
        self._have_t = False
        self.mw: Vec3 = (0.0, 0.0, 0.0)  # last good gyro reading
        self.ma: Vec3 = (0.0, 0.0, G0)  # last good accel reading
        self._have_meas = False
        self.dead = False
        self._bad_run = 0
        self._bad_since = 0  # timestamp of the first row in the current invalid run
        self._pending: list[int] = []  # timestamps of a possibly-frozen run
        self._ema_g2 = 0.0
        self._ema_a2 = 0.0
        self._gm: Vec3 = (0.0, 0.0, 0.0)
        self._am: Vec3 = (0.0, 0.0, G0)
        self._thr_g = 0.01
        self._thr_a = 0.05
        self._still_t = 0.0

    # ---------------------------------------------------------------- public API
    def push(self, s: ImuSample) -> None:
        """Add one sample (start-up samples are buffered until the init window is full)."""
        if not self._inited:
            self._init_buf.append(s)
            if len(self._init_buf) >= self.c.max_init or (s.t_ns - self._init_buf[0].t_ns) * 1e-9 >= self.c.init_time:
                self._finish_init()
            return
        self._feed(s)

    def finish(self) -> None:
        """Flush buffered samples. Call exactly once after the last :meth:`push`."""
        if not self._inited:
            self._finish_init()
        self._flush_pending(as_dead=False)

    # ---------------------------------------------------------------- internals
    def _emit(self, t: int) -> None:
        self.poses.append(Pose(t, self.p, self.q))

    def _finish_init(self) -> None:
        """Gyro bias, gravity magnitude, initial roll/pitch and noise thresholds from start-up."""
        self._inited = True
        buf = self._init_buf
        dts = [d for d in ((buf[i].t_ns - buf[i - 1].t_ns) * 1e-9 for i in range(1, len(buf))) if 0 < d < 0.2]
        if dts:
            dts.sort()
            self.dt0 = dts[len(dts) // 2]  # median sample period
        good = [s for s in buf if is_plausible(s)]
        if good:
            ng = len(good)
            mw: Vec3 = (0.0, 0.0, 0.0)
            ma: Vec3 = (0.0, 0.0, 0.0)
            for s in good:
                mw, ma = add(mw, s.gyro), add(ma, s.accel)
            mw, ma = mul(mw, 1.0 / ng), mul(ma, 1.0 / ng)
            sg = sa = 0.0
            for s in good:
                dw, da = sub(s.gyro, mw), sub(s.accel, ma)
                sg += dot(dw, dw)
                sa += dot(da, da)
            sg, sa = math.sqrt(sg / ng), math.sqrt(sa / ng)
            if sg < 0.08:  # roughly at rest: the mean gyro reading is its bias
                self.bg = mw
            gn = norm(ma)
            self.g = gn if 9.0 < gn < 10.6 else G0
            up = mul(ma, 1.0 / max(gn, 1e-9))  # rotate measured "up" onto world +z; yaw = 0
            ax = cross(up, (0.0, 0.0, 1.0))
            s_, cth = norm(ax), clamp(up[2], -1.0, 1.0)
            if s_ < 1e-9:
                self.q = (1.0, 0.0, 0.0, 0.0) if cth > 0 else (0.0, 1.0, 0.0, 0.0)
            else:
                self.q = qexp(mul(ax, math.atan2(s_, cth) / s_))
            self._thr_g = max(0.01, 4.0 * sg)
            self._thr_a = max(0.05, 4.0 * sa)
            self._gm = self.bg
            self._am = ma
        for s in buf:
            self._feed(s)
        self._init_buf = []

    def _propagate(self, h: float, w: Vec3, a: Vec3) -> None:
        """One integration step of length h seconds with gyro w and accel a."""
        c = self.c
        an = norm(a)
        # Mahony gravity correction, smoothly switched off when |a| is far from g.
        up_b = qrot_inv(self.q, (0.0, 0.0, 1.0))
        e = cross(mul(a, 1.0 / max(an, 1e-9)), up_b)
        dev = (an - self.g) / c.acc_tol
        kg = 1.0 / (1.0 + dev**4)
        ei = add(self.ei, mul(e, c.ki * kg * h))
        self.ei = (clamp(ei[0], -0.05, 0.05), clamp(ei[1], -0.05, 0.05), clamp(ei[2], -0.05, 0.05))
        wc = add(add(sub(w, self.bg), self.ei), mul(e, c.kp * kg))

        qmid = qnormalize(qmul(self.q, qexp(mul(wc, 0.5 * h))))  # midpoint attitude for the accel
        self.q = qnormalize(qmul(self.q, qexp(mul(wc, h))))
        aw = sub(qrot(qmid, a), (0.0, 0.0, self.g))
        vn = add(self.v, mul(aw, h))
        if c.vleak > 0:
            vn = mul(vn, math.exp(-c.vleak * h))
        sp = norm(vn)
        if sp > c.vmax:
            vn = mul(vn, c.vmax / sp)
        self.p = add(self.p, mul(add(self.v, vn), 0.5 * h))
        self.v = vn

        # Stationarity detection -> zero-velocity update (+ gyro-bias refinement).
        al = 1.0 - math.exp(-h / 0.2)
        wr = sub(w, self.bg)
        self._ema_g2 += al * (dot(wr, wr) - self._ema_g2)
        da = sub(a, self._am)
        self._ema_a2 += al * (dot(da, da) - self._ema_a2)
        self._am = add(self._am, mul(sub(a, self._am), 1.0 - math.exp(-h / 1.0)))
        self._gm = add(self._gm, mul(sub(w, self._gm), al))
        still = math.sqrt(self._ema_g2) < self._thr_g and math.sqrt(self._ema_a2) < self._thr_a
        self._still_t = self._still_t + h if still else 0.0
        if c.zupt and self._still_t >= c.zupt_time and norm(self.v) < c.zupt_speed:
            self.v = (0.0, 0.0, 0.0)
            self.bg = add(self.bg, mul(sub(self._gm, self.bg), min(1.0, h / 2.0)))

    def _integrate_to(self, cur: ImuSample) -> None:
        """Integrate up to cur.t_ns; gaps (dropped frames) are sub-stepped with interpolated input."""
        dt = (cur.t_ns - self.tlast) * 1e-9
        if dt <= 0:
            return
        w0 = self.mw if self._have_meas else cur.gyro
        a0 = self.ma if self._have_meas else cur.accel
        n = int(clamp(math.floor(dt / self.dt0 + 0.5), 1.0, 1000.0))
        h = dt / n
        for k in range(n):
            fm = 0.5 * (k / n + (k + 1) / n)
            self._propagate(h, add(w0, mul(sub(cur.gyro, w0), fm)), add(a0, mul(sub(cur.accel, a0), fm)))
        self.mw, self.ma, self._have_meas, self.tlast = cur.gyro, cur.accel, True, cur.t_ns

    def _coast_to(self, t: int) -> None:
        """Dead reckoning without IMU input: constant velocity (optionally decaying), held attitude."""
        dt = (t - self.tlast) * 1e-9
        if dt > 0:
            self.p = add(self.p, mul(self.v, dt))
            if self.c.dead_decay > 0:
                self.v = mul(self.v, math.exp(-self.c.dead_decay * dt))
            self.tlast = t
        self._still_t = 0.0

    def _declare_dead(self, t: int) -> None:
        self.dead = True
        self.dead_since_ns = t

    def _feed(self, s: ImuSample) -> None:
        if not self._have_t:
            self._have_t = True
            self.tlast = s.t_ns
            if is_plausible(s):
                self.mw, self.ma, self._have_meas = s.gyro, s.accel, True
            self._emit(s.t_ns)
            return
        if self.dead:
            self._coast_to(s.t_ns)
            self._emit(s.t_ns)
            return
        if not is_plausible(s):  # NaN / empty / zeroed / absurd reading
            self._flush_pending(as_dead=False)
            self._coast_to(s.t_ns)
            self._have_meas = False
            if self._bad_run == 0:
                self._bad_since = s.t_ns
            self._bad_run += 1
            if self._bad_run >= self.c.bad_n:
                self._declare_dead(self._bad_since)
            self._emit(s.t_ns)
            return
        self._bad_run = 0
        if self._have_meas and s.gyro == self.mw and s.accel == self.ma:  # bit-identical => maybe frozen
            self._pending.append(s.t_ns)
            if len(self._pending) >= self.c.stuck_n:
                self._declare_dead(self._pending[0])
                self._flush_pending(as_dead=True)  # frozen rows are coasted, never integrated
            return
        self._flush_pending(as_dead=False)
        self._step(s)

    def _step(self, s: ImuSample) -> None:
        if s.t_ns <= self.tlast:  # duplicate / out-of-order stamp: still one pose per row
            self._emit(s.t_ns)
            return
        self._integrate_to(s)
        self._emit(s.t_ns)

    def _flush_pending(self, *, as_dead: bool) -> None:
        pending, self._pending = self._pending, []
        for t in pending:
            if as_dead:
                self._coast_to(t)
                self._emit(t)
            else:
                self._step(ImuSample(t, self.mw, self.ma, True))


# ======================================================================================
#  Offline drift removal (second pass)
# ======================================================================================
def remove_drift(poses: Sequence[Pose], tau: float) -> list[Pose]:
    """Subtract slow drift: zero-phase (forward + backward) first-order low-pass, time constant tau.

    Keeps the short-term motion the IMU measures well and removes the drift it accumulates.
    Assumes the device stays inside a bounded area (a room); ``tau <= 0`` returns the input unchanged.
    """
    if tau <= 0 or len(poses) < 2:
        return list(poses)
    t = [p.t_ns * 1e-9 for p in poses]
    raw = [list(p.position) for p in poses]
    n = len(raw)
    low = [r[:] for r in raw]
    for i in range(1, n):  # forward pass
        a = 1.0 - math.exp(-(t[i] - t[i - 1]) / tau)
        for k in range(3):
            low[i][k] = low[i - 1][k] + a * (raw[i][k] - low[i - 1][k])
    for i in range(n - 2, -1, -1):  # backward pass -> zero phase lag
        a = 1.0 - math.exp(-(t[i + 1] - t[i]) / tau)
        for k in range(3):
            low[i][k] = low[i + 1][k] + a * (low[i][k] - low[i + 1][k])
    return [Pose(p.t_ns, (raw[i][0] - low[i][0], raw[i][1] - low[i][1], raw[i][2] - low[i][2]), p.orientation) for i, p in enumerate(poses)]


# ======================================================================================
#  Input
# ======================================================================================
def resolve_input_file(path: str | os.PathLike[str], what: str) -> Path:
    """Canonicalise a user-supplied path and check it is an existing regular file."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise InputError(f"Cannot find the {what} file: {p}. Put it next to this script or give its full path.")
    if not p.is_file():
        raise InputError(f"The {what} path is not a file: {p}")
    return p


def parse_imu_line(line: str) -> ImuSample | None:
    """Parse one csv row; None for blank/comment/header rows. Bad numbers give ``valid=False``."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    fields = line.split(",")
    try:
        t = int(fields[0])
    except ValueError:
        try:
            tf = float(fields[0])
        except ValueError:
            return None  # header text
        if not math.isfinite(tf):
            return None
        t = round(tf)
    vals: list[float] = []
    ok = True
    for i in range(1, 7):
        try:
            x = float(fields[i])
            if not math.isfinite(x):
                raise ValueError
        except (ValueError, IndexError):
            x, ok = 0.0, False
        vals.append(x)
    return ImuSample(t, (vals[0], vals[1], vals[2]), (vals[3], vals[4], vals[5]), ok)


def read_imu(path: Path, secs: float = 0.0) -> Iterator[ImuSample]:
    """Yield samples from an IMU csv; ``secs > 0`` stops after the first ``secs`` seconds."""
    t_first: int | None = None
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            s = parse_imu_line(line)
            if s is None:
                continue
            if t_first is None:
                t_first = s.t_ns
            if secs > 0 and s.t_ns > t_first + secs * 1e9:
                break
            yield s


def run_estimator(samples: Iterable[ImuSample], cfg: Config) -> StrapdownEstimator:
    """Run the estimator over all samples and return it (poses in ``.poses``)."""
    ins = StrapdownEstimator(cfg)
    for s in samples:
        ins.push(s)
    ins.finish()
    return ins


# ======================================================================================
#  Output
# ======================================================================================
def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + rename, so a crash never leaves a half-written output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def format_tum(poses: Iterable[Pose]) -> str:
    """TUM lines ``t[s] tx ty tz qx qy qz qw``; the timestamp keeps the input's exact nanoseconds."""
    lines = []
    for p in poses:
        sec, ns = divmod(p.t_ns, 10**9)
        (x, y, z), (qw, qx, qy, qz) = p.position, p.orientation
        lines.append("%d.%09d %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" % (sec, ns, x, y, z, qx, qy, qz, qw))
    return "".join(lines)


# ======================================================================================
#  Vicon ground truth -> TUM, time-synchronised to the IMU
# ======================================================================================
def load_vicon(path: Path) -> npt.NDArray[np.float64]:
    """Vicon csv (Frame, Sub Frame, RX, RY, RZ, RW, TX[mm], TY, TZ) -> (N, 9); occluded frames are NaN."""
    rows: list[list[float]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        header = f.readline()
        if len(header.split(",")) < 9:
            raise InputError(f"{path.name}: expected 9 columns (Frame, Sub Frame, RX, RY, RZ, RW, TX, TY, TZ)")
        for line in f:
            fields = line.strip().split(",")
            if not fields or not fields[0]:
                continue
            row = [math.nan] * 9
            try:
                row[0] = float(fields[0])
                if len(fields) >= 9:
                    row[1:] = [float(x) if x.strip() else math.nan for x in fields[1:9]]
            except ValueError:
                pass  # malformed row: keep it as an occluded frame
            if math.isfinite(row[0]):
                rows.append(row)
    if len(rows) < 10:
        raise InputError(f"{path.name}: fewer than 10 usable Vicon rows")
    return np.array(rows, dtype=np.float64)


class SyncResult(NamedTuple):
    """Vicon->IMU clock offset (vicon_time + offset = imu_time) and its correlation score."""

    offset_s: float
    correlation: float


def sync_offset(imu: Sequence[ImuSample], vic: npt.NDArray[np.float64], hz: float) -> SyncResult:
    """Find the clock offset by correlating rotation speed |w| seen by Vicon and by the gyro."""
    good = [s for s in imu if s.valid]
    if len(good) < 100:
        raise InputError("Too few valid IMU rows to synchronise with Vicon")
    t = (np.array([s.t_ns for s in good], dtype=np.float64) - good[0].t_ns) * 1e-9
    wn = np.array([norm(s.gyro) for s in good])
    q = vic[:, 2:6]
    wv = np.nan_to_num(2 * np.arccos(np.abs(np.sum(q[1:] * q[:-1], 1)).clip(0, 1)) * hz)
    n = min(len(wv), int(300 * hz))  # 5 minutes is plenty to find the offset
    tv = np.arange(n) / hz + 0.5 / hz
    kernel = np.ones(5) / 5

    def lowpass(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.convolve(x, kernel, "same")

    a = lowpass(wv[:n])

    def score(off: float) -> float:
        c = float(np.corrcoef(a, lowpass(np.interp(tv + off, t, wn)))[0, 1])
        return c if math.isfinite(c) else -1.0

    best = max((score(float(o)), float(o)) for o in np.arange(-15, 15.001, 0.02))
    best = max((score(float(o)), float(o)) for o in np.arange(best[1] - 0.03, best[1] + 0.0301, 0.005))
    return SyncResult(offset_s=best[1], correlation=best[0])


def vicon_to_tum(imu: Sequence[ImuSample], vicon_path: Path, out_path: Path, hz: float) -> SyncResult:
    """Write the Vicon path as TUM (metres, IMU clock) and return the sync result."""
    vic = load_vicon(vicon_path)
    sync = sync_offset(imu, vic, hz)
    if sync.correlation < 0.5:
        log.warning("Vicon/IMU sync is weak (correlation %.2f); the comparison may be misaligned", sync.correlation)
    t0 = imu[0].t_ns * 1e-9
    tum = np.column_stack([t0 + (vic[:, 0] - vic[0, 0]) / hz + sync.offset_s, vic[:, 6:9] / 1000.0, vic[:, 2:6]])
    tum = tum[~np.isnan(tum).any(axis=1)]
    lines = ["%.6f %.6f %.6f %.6f %.8f %.8f %.8f %.8f\n" % tuple(r) for r in tum]
    atomic_write_text(out_path, "".join(lines))
    return sync


# ======================================================================================
#  CLI
# ======================================================================================
def non_negative(text: str) -> float:
    """Parse a finite float >= 0 (argparse ``type=``)."""
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(v) or v < 0:
        raise argparse.ArgumentTypeError(f"must be a finite number >= 0: {text!r}")
    return v


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imu", nargs="?", default="gyro_accel.csv", help="IMU csv (default: gyro_accel.csv)")
    ap.add_argument("--vicon", help="Vicon csv for the actual path (default: vicon_capture_quaternion.csv if present)")
    ap.add_argument("--vicon-hz", type=non_negative, default=100.0, help="Vicon frame rate (default 100)")
    ap.add_argument("--secs", type=non_negative, default=60.0, help="use the first N seconds (default 60; 0 = all)")
    ap.add_argument("--out", default="output", help="output folder (default: output)")
    ap.add_argument("--smooth", type=non_negative, default=2.0, help="drift-removal time constant [s] (default 2)")
    ap.add_argument("--no-smooth", action="store_true", help="turn drift removal off (raw dead reckoning)")
    ap.add_argument("--no-plot", action="store_true", help="skip the picture")
    ap.add_argument("--kp", type=non_negative)
    ap.add_argument("--ki", type=non_negative)
    ap.add_argument("--vleak", type=non_negative)
    ap.add_argument("--dead-decay", type=non_negative, dest="dead_decay")
    ap.add_argument("--no-zupt", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return ap


def config_from_args(a: argparse.Namespace) -> Config:
    """Build a Config from CLI options, keeping defaults for anything not given."""
    overrides: dict[str, float | bool] = {k: getattr(a, k) for k in ("kp", "ki", "vleak", "dead_decay") if getattr(a, k) is not None}
    if a.no_zupt:
        overrides["zupt"] = False
    return Config(**overrides)  # type: ignore[arg-type]  # keys are a fixed subset of Config fields


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 ok, 2 input error)."""
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(message)s", stream=sys.stdout)
    try:
        cfg = config_from_args(a)
        imu_path = resolve_input_file(a.imu, "IMU")
        vicon_arg = a.vicon
        if vicon_arg is None and Path("vicon_capture_quaternion.csv").is_file():
            vicon_arg = "vicon_capture_quaternion.csv"
            log.info("using vicon_capture_quaternion.csv as the actual path")
        vicon_path = resolve_input_file(vicon_arg, "Vicon") if vicon_arg else None
        if vicon_path is not None and a.vicon_hz <= 0:
            raise InputError("--vicon-hz must be > 0")
        out_dir = Path(a.out).expanduser().resolve()
        if out_dir.exists() and not out_dir.is_dir():
            raise InputError(f"--out exists and is not a folder: {out_dir}")

        samples = list(read_imu(imu_path, a.secs))
        if not samples:
            raise InputError(f"{imu_path.name}: no IMU rows found (expected: timestamp_ns,gx,gy,gz,ax,ay,az)")
        n_bad = sum(not s.valid for s in samples)
        if n_bad:
            log.info("%d of %d IMU rows have missing/invalid numbers (treated as sensor faults)", n_bad, len(samples))

        ins = run_estimator(samples, cfg)
        poses = ins.poses if a.no_smooth else remove_drift(ins.poses, a.smooth)
        if len(poses) != len(samples):  # contract: exactly one pose per IMU row
            raise RuntimeError(f"internal error: {len(poses)} poses for {len(samples)} rows")
        if ins.dead_since_ns is not None:
            log.info("IMU declared dead at t=%.2f s; dead-reckoning from the last good state", (ins.dead_since_ns - samples[0].t_ns) * 1e-9)

        est = out_dir / "estimated_trajectory.txt"
        atomic_write_text(est, format_tum(poses))
        log.info("wrote %s  (%d poses, one per IMU row)", est, len(poses))

        gt: Path | None = None
        if vicon_path is not None:
            gt = out_dir / "gt_tum.txt"
            # Sync on the whole IMU file, not just the --secs window: more motion -> a sharper peak.
            full = samples if a.secs == 0 else list(read_imu(imu_path))
            sync = vicon_to_tum(full, vicon_path, gt, a.vicon_hz)
            log.info("Vicon<->IMU time offset %.3f s (correlation %.2f)", sync.offset_s, sync.correlation)

        if not a.no_plot:
            png = out_dir / "paths.png"
            plot_paths.render(est, png, gt=gt)
            log.info("\nDone. Open %s", png)
    except InputError as e:
        log.error("error: %s", e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
