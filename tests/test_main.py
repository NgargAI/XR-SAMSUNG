"""Tests for the standalone Python entry point src/main.py."""

from __future__ import annotations

from pathlib import Path

import demo_python as dp
import main as entry

FIX = Path(__file__).parent / "fixtures"


def test_main_matches_the_demo_estimator_row_for_row(tmp_path: Path) -> None:
    out = tmp_path / "estimated_trajectory.txt"
    assert entry.main([str(FIX / "sim_degraded_nan.csv"), str(out)]) == 0
    golden = (FIX / "sim_degraded_nan.golden.txt").read_text().splitlines()
    assert out.read_text().splitlines() == golden


def test_main_writes_one_pose_per_imu_row(tmp_path: Path) -> None:
    out = tmp_path / "t.txt"
    assert entry.main([str(FIX / "sim_degraded_freeze.csv"), str(out), "--no-smooth"]) == 0
    rows = [s for s in dp.read_imu(FIX / "sim_degraded_freeze.csv")]
    assert len(out.read_text().splitlines()) == len(rows)


def test_main_reports_missing_input_with_exit_code_2(tmp_path: Path) -> None:
    assert entry.main([str(tmp_path / "missing.csv"), str(tmp_path / "o.txt")]) == 2
