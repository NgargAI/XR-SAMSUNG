"""Tests for demo_python.py: golden parity, failure handling, validation, sync, drift removal."""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

import demo_python as dp
import plot_paths

FIX = Path(__file__).parent / "fixtures"
NS = 1_000_000_000


Row = tuple[int, float, float, float, float, float, float]


def write_csv(path: Path, rows: Sequence[Row | str]) -> Path:
    lines = ["#timestamp [ns],wx,wy,wz,ax,ay,az"]
    for r in rows:
        lines.append(r if isinstance(r, str) else ",".join(str(x) for x in r))
    path.write_text("\n".join(lines) + "\n")
    return path


def still_rows(n: int, t0: int = NS, dt_ns: int = 5_000_000) -> list[tuple[int, float, float, float, float, float, float]]:
    """A level IMU at rest, with tiny deterministic noise so rows are never bit-identical."""
    return [(t0 + i * dt_ns, 1e-4 * math.sin(i), 1e-4 * math.cos(i), 0.0, 0.001 * math.sin(0.7 * i), 0.0, 9.81) for i in range(n)]


def run(path: Path, **cfg: float | bool) -> dp.StrapdownEstimator:
    return dp.run_estimator(dp.read_imu(path), dp.Config(**cfg))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- golden parity
@pytest.mark.parametrize("name", ["sim_degraded_nan", "sim_degraded_freeze"])
@pytest.mark.parametrize("smooth", [True, False])
def test_output_matches_previous_release_exactly(tmp_path: Path, name: str, smooth: bool) -> None:
    # Golden files come from the previous demo_python.py (itself verified against pathfinder.cpp).
    out = tmp_path / "o"
    args = [str(FIX / f"{name}.csv"), "--secs", "0", "--out", str(out), "--no-plot"]
    if not smooth:
        args.append("--no-smooth")
    assert dp.main(args) == 0
    golden = FIX / f"{name}.golden{'' if smooth else '_nosmooth'}.txt"
    got_lines = (out / "estimated_trajectory.txt").read_text().splitlines()
    want_lines = golden.read_text().splitlines()
    # Report the first differing line ourselves: pytest's own diff of two ~4000-line strings is
    # quadratic and stalled for >10 minutes when this test was mutation-checked.
    first_diff = next((i for i, (g, w) in enumerate(zip(got_lines, want_lines, strict=False)) if g != w), None)
    assert first_diff is None, f"line {first_diff + 1}: got {got_lines[first_diff]!r}, want {want_lines[first_diff]!r}"
    assert len(got_lines) == len(want_lines)


# --------------------------------------------------------------------------- contract
def test_one_pose_per_row_with_identical_timestamps_even_with_bad_rows(tmp_path: Path) -> None:
    base = still_rows(200)
    rows: list[Row | str] = list(base)
    rows.insert(150, f"{base[149][0] + 1},nan,0,0,0,0,9.81")  # NaN row
    rows.insert(160, f"{base[158][0] + 1},0.1,0.1")  # missing fields
    rows.insert(170, base[167])  # duplicate timestamp, right after the original row
    p = write_csv(tmp_path / "imu.csv", rows)
    samples = list(dp.read_imu(p))
    ins = dp.run_estimator(samples, dp.Config())
    assert [x.t_ns for x in ins.poses] == [s.t_ns for s in samples]
    assert len(samples) == 203


def test_tum_timestamps_keep_full_nanosecond_precision() -> None:
    t = 1403636580838555648  # EuRoC-style: too many digits for a float
    text = dp.format_tum([dp.Pose(t, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))])
    assert text.split()[0] == "1403636580.838555648"


# --------------------------------------------------------------------------- behaviour
def test_stationary_imu_stays_at_origin(tmp_path: Path) -> None:
    ins = run(write_csv(tmp_path / "s.csv", still_rows(2000)))
    assert max(dp.norm(p.position) for p in ins.poses) < 0.05
    assert ins.dead_since_ns is None


def test_dead_imu_is_detected_and_coasts_at_constant_velocity(tmp_path: Path) -> None:
    rows = still_rows(400)
    t_dead = rows[-1][0] + 5_000_000
    rows += [(t_dead + i * 5_000_000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for i in range(200)]
    ins = run(write_csv(tmp_path / "d.csv", rows))
    assert ins.dead_since_ns == t_dead  # reported from the first bad row, not the 20th
    tail = ins.poses[-100:]
    steps = [dp.norm(dp.sub(b.position, a.position)) for a, b in itertools.pairwise(tail)]
    assert max(steps) - min(steps) < 1e-9  # straight-line coasting
    assert len({p.orientation for p in tail}) == 1  # attitude held


def test_frozen_sensor_is_declared_dead_and_frozen_rows_are_not_integrated(tmp_path: Path) -> None:
    rows = still_rows(400)
    frozen = rows[-1]
    rows += [(frozen[0] + (i + 1) * 5_000_000, *frozen[1:]) for i in range(100)]
    ins = run(write_csv(tmp_path / "f.csv", rows))
    assert ins.dead_since_ns == rows[400][0]
    assert len(ins.poses) == len(rows)


def test_short_identical_run_is_integrated_normally(tmp_path: Path) -> None:
    rows = still_rows(400)
    rep = rows[-1]
    rows += [(rep[0] + (i + 1) * 5_000_000, *rep[1:]) for i in range(5)]  # < stuck_n
    rows += still_rows(50, t0=rows[-1][0] + 5_000_000)
    ins = run(write_csv(tmp_path / "r.csv", rows))
    assert ins.dead_since_ns is None
    assert len(ins.poses) == len(rows)


def test_dropped_frames_are_bridged_without_a_jump(tmp_path: Path) -> None:
    rows = still_rows(400)
    del rows[200:260]  # 0.3 s gap
    ins = run(write_csv(tmp_path / "g.csv", rows))
    assert len(ins.poses) == len(rows)
    assert max(dp.norm(p.position) for p in ins.poses) < 0.05


# --------------------------------------------------------------------------- parsing
@pytest.mark.parametrize("line", ["", "   ", "#comment", "timestamp,wx,wy,wz,ax,ay,az", "abc,1,2,3,4,5,6"])
def test_parse_skips_blank_comment_and_header_rows(line: str) -> None:
    assert dp.parse_imu_line(line) is None


@pytest.mark.parametrize("line", ["1,nan,0,0,0,0,9.8", "1,0,0,0,0,0", "1,0,0,0,0,0,", "1,0,0,x,0,0,9.8", "1,inf,0,0,0,0,9.8"])
def test_parse_marks_missing_or_non_finite_numbers_invalid(line: str) -> None:
    s = dp.parse_imu_line(line)
    assert s is not None and s.valid is False


def test_parse_accepts_crlf_and_float_timestamps() -> None:
    s = dp.parse_imu_line("1.0e9,0.1,0.2,0.3,0.4,0.5,9.8\r\n")
    assert s == dp.ImuSample(1_000_000_000, (0.1, 0.2, 0.3), (0.4, 0.5, 9.8), True)


def test_read_imu_secs_limits_duration(tmp_path: Path) -> None:
    p = write_csv(tmp_path / "s.csv", still_rows(1000))  # 5 s at 200 Hz
    got = list(dp.read_imu(p, secs=1.0))
    assert (got[-1].t_ns - got[0].t_ns) <= NS and len(got) == 201


# --------------------------------------------------------------------------- maths
def test_qexp_quarter_turn_about_z_maps_x_to_y() -> None:
    q = dp.qexp((0.0, 0.0, math.pi / 2))
    assert math.isclose(sum(c * c for c in q), 1.0)
    assert np.allclose(dp.qrot(q, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0), atol=1e-12)
    assert np.allclose(dp.qrot_inv(q, (0.0, 1.0, 0.0)), (1.0, 0.0, 0.0), atol=1e-12)


def test_remove_drift_off_returns_input_unchanged() -> None:
    poses = [dp.Pose(i * 5_000_000, (float(i), 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)) for i in range(10)]
    assert dp.remove_drift(poses, 0.0) == poses


def test_remove_drift_removes_slow_ramp_but_keeps_fast_motion() -> None:
    # The filter needs several time constants to settle at each end of the recording (with
    # tau = 2 s a ramp leaves ~exp(-t/tau) * tau * slope there), so the signal is 60 s long and only
    # the middle 20 s is checked. A 20 s signal measured 0.028 m here purely from edge transients.
    n, dt = 12000, 0.005
    t = np.arange(n) * dt
    fast = 0.1 * np.sin(2 * math.pi * 2.0 * t)  # 2 Hz motion: keep
    ramp = 0.5 * t  # slow drift: remove
    poses = [dp.Pose(round(ti * 1e9), (float(f + r), 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)) for ti, f, r in zip(t, fast, ramp, strict=True)]
    out = np.array([p.position[0] for p in dp.remove_drift(poses, 2.0)])
    mid = slice(4000, 8000)  # 20-40 s: 10 time constants from either edge
    assert np.sqrt(np.mean((out[mid] - fast[mid]) ** 2)) < 0.01
    assert [p.t_ns for p in dp.remove_drift(poses, 2.0)] == [p.t_ns for p in poses]


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize("bad", [{"kp": -1.0}, {"vleak": math.nan}, {"acc_tol": 0.0}, {"stuck_n": 1}])
def test_config_rejects_invalid_values(bad: dict[str, float]) -> None:
    with pytest.raises(dp.InputError):
        dp.Config(**bad)  # type: ignore[arg-type]  # deliberately invalid values


def test_missing_and_directory_inputs_are_reported(tmp_path: Path) -> None:
    with pytest.raises(dp.InputError, match="Cannot find"):
        dp.resolve_input_file(tmp_path / "nope.csv", "IMU")
    with pytest.raises(dp.InputError, match="not a file"):
        dp.resolve_input_file(tmp_path, "IMU")


def test_main_exits_2_with_a_message_for_missing_file(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert dp.main([str(tmp_path / "missing.csv"), "--no-plot", "--out", str(tmp_path / "o")]) == 2
    assert "Cannot find" in caplog.text


def test_main_exits_2_for_file_without_rows(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("timestamp,wx,wy,wz,ax,ay,az\n")
    assert dp.main([str(p), "--no-plot", "--out", str(tmp_path / "o")]) == 2


def test_negative_option_is_rejected_by_the_parser(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        dp.main([str(tmp_path / "x.csv"), "--secs", "-5"])


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "out.txt"
    dp.atomic_write_text(target, "hello\n")
    assert target.read_text() == "hello\n"
    assert [p.name for p in target.parent.iterdir()] == ["out.txt"]


# --------------------------------------------------------------------------- vicon
def test_load_vicon_keeps_occluded_frames_as_nan(tmp_path: Path) -> None:
    lines = ["Frame,Sub Frame,RX,RY,RZ,RW,TX,TY,TZ"] + [f"{i},0,0,0,0,1,{i},0,0" for i in range(1, 12)] + ["12,0", "13,0,,,,,,,"]
    p = tmp_path / "v.csv"
    p.write_text("\n".join(lines) + "\n")
    v = dp.load_vicon(p)
    assert v.shape == (13, 9)
    assert np.isnan(v[-2:, 2:]).all() and np.isfinite(v[:11]).all()


def test_load_vicon_rejects_wrong_layout(tmp_path: Path) -> None:
    p = tmp_path / "v.csv"
    p.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(dp.InputError, match="9 columns"):
        dp.load_vicon(p)


def test_sync_offset_recovers_a_known_clock_offset() -> None:
    hz, true_off = 100.0, 0.37
    rng = np.random.default_rng(0)
    rate = np.convolve(np.abs(rng.normal(0, 1, 6000)), np.ones(50) / 50, "same")  # random yaw-rate profile
    # Vicon: integrate the rate into a yaw angle -> quaternion (x y z w) per frame.
    yaw = np.cumsum(rate) / hz
    vic = np.zeros((len(yaw), 9))
    vic[:, 0] = np.arange(len(yaw))
    vic[:, 4], vic[:, 5] = np.sin(yaw / 2), np.cos(yaw / 2)
    # IMU at 200 Hz on its own clock: imu_time = vicon_time + offset.
    t_imu = np.arange(0, len(yaw) / hz - 1, 1 / 200)
    w = np.interp(t_imu - true_off, np.arange(len(yaw)) / hz, rate)
    imu = [dp.ImuSample(round(t * 1e9), (0.0, 0.0, float(wz)), (0.0, 0.0, 9.81), True) for t, wz in zip(t_imu, w, strict=True)]
    res = dp.sync_offset(imu, vic, hz)
    assert abs(res.offset_s - true_off) < 0.015 and res.correlation > 0.9


# --------------------------------------------------------------------------- real data (optional)
REAL = os.environ.get("PATHFINDER_REAL_DIR")


@pytest.mark.skipif(not REAL, reason="set PATHFINDER_REAL_DIR to a folder with gyro_accel.csv + vicon_capture_quaternion.csv")
def test_real_recording_first_60s_scores_as_before(tmp_path: Path) -> None:
    d = Path(REAL or ".")
    out = tmp_path / "o"
    assert dp.main([str(d / "gyro_accel.csv"), "--vicon", str(d / "vicon_capture_quaternion.csv"), "--out", str(out), "--no-plot"]) == 0
    _, _, res = plot_paths.align(plot_paths.load_tum(out / "estimated_trajectory.txt"), plot_paths.load_tum(out / "gt_tum.txt"))
    assert res["rmse_m"] == pytest.approx(1.341, abs=0.002)
