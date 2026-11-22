"""Tests for plot_paths.py: alignment maths, input checks, and that pictures are produced."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import plot_paths as pp


def write_tum(path: Path, xyz: np.ndarray, t0: float = 100.0, hz: float = 50.0) -> Path:
    t = t0 + np.arange(len(xyz)) / hz
    q = np.tile([0.0, 0.0, 0.0, 1.0], (len(xyz), 1))
    np.savetxt(path, np.column_stack([t, xyz, q]), fmt="%.9f")
    return path


def circle(n: int = 300) -> np.ndarray:
    a = np.linspace(0, 2 * math.pi, n)
    return np.column_stack([np.cos(a), np.sin(a), 0.1 * a])


def test_umeyama_recovers_a_known_rotation_and_shift() -> None:
    p = circle()
    c, s = math.cos(0.7), math.sin(0.7)
    r_true = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    q = p @ r_true.T + np.array([1.0, -2.0, 0.5])
    r, t = pp.umeyama(p, q)
    assert np.allclose(r, r_true) and np.allclose(t, [1.0, -2.0, 0.5])


def test_align_gives_zero_error_for_a_rigidly_moved_copy() -> None:
    p = circle()
    est = np.column_stack([np.arange(len(p)) / 50.0, p, np.tile([0, 0, 0, 1.0], (len(p), 1))])
    gt = est.copy()
    gt[:, 1:4] = p[:, [1, 0, 2]] * [1, -1, 1] + 3.0  # rotated 90 deg about z, shifted
    _, _, res = pp.align(est, gt)
    assert res["rmse_m"] < 1e-9 and res["n_matched"] == len(p)


def test_align_rejects_non_overlapping_ground_truth() -> None:
    est = np.column_stack([np.arange(10.0), np.zeros((10, 3)), np.tile([0, 0, 0, 1.0], (10, 1))])
    gt = est.copy()
    gt[:, 0] += 1000
    with pytest.raises(ValueError, match="overlap"):
        pp.align(est, gt)


@pytest.mark.parametrize(
    ("content", "msg"),
    [
        ("1 2 3\n4 5 6\n", "8 columns"),
        ("1 0 0 0 0 0 0 1\n", "at least 2"),
        ("1 nan 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n", "NaN"),
        ("2 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n", "increasing"),
    ],
)
def test_load_tum_rejects_malformed_files(tmp_path: Path, content: str, msg: str) -> None:
    p = tmp_path / "x.txt"
    p.write_text(content)
    with pytest.raises(ValueError, match=msg):
        pp.load_tum(p)


def test_load_tum_missing_file_says_to_run_the_demo_first(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"demo_python\.py"):
        pp.load_tum(tmp_path / "nope.txt")


@pytest.mark.parametrize("with_gt", [True, False])
def test_render_writes_a_png(tmp_path: Path, with_gt: bool) -> None:
    est = write_tum(tmp_path / "est.txt", circle())
    gt = write_tum(tmp_path / "gt.txt", circle() * 1.1) if with_gt else None
    out = tmp_path / "pic" / "paths.png"
    res = pp.render(est, out, gt=gt)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (res is not None) == with_gt


def test_main_returns_2_when_estimate_is_missing(tmp_path: Path) -> None:
    assert pp.main(["--est", str(tmp_path / "none.txt"), "--out", str(tmp_path / "p.png")]) == 2


def test_main_rejects_non_positive_secs_with_a_message(tmp_path: Path) -> None:
    est = write_tum(tmp_path / "est.txt", circle())
    with pytest.raises(SystemExit) as e:
        pp.main(["--est", str(est), "--out", str(tmp_path / "p.png"), "--secs", "-1"])
    assert e.value.code == 2
