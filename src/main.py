#!/usr/bin/env python3
"""Path Finder - standalone Python entry point (Track 03).

    python src/main.py imu0/data.csv estimated_trajectory.txt [--no-smooth]

Input : EuRoC IMU csv (timestamp[ns], w_x, w_y, w_z [rad/s], a_x, a_y, a_z [m/s^2]).
Output: TUM trajectory (timestamp[s] tx ty tz qx qy qz qw), exactly one pose per IMU row.

Same algorithm as the C++ entry point ``src/main.cpp``. Use the C++ binary for the contest's
8 MB / 20 ms budgets; this script needs far more memory because it is pure Python.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import demo_python as dp

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger("pathfinder.main")


def main(argv: Sequence[str] | None = None) -> int:
    """Read the IMU csv, estimate the trajectory, write it. Returns 0 ok, 2 bad input."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imu", help="imu0/data.csv")
    ap.add_argument("output", help="estimated_trajectory.txt")
    ap.add_argument("--smooth", type=dp.non_negative, default=2.0, help="drift-removal time constant [s] (default 2)")
    ap.add_argument("--no-smooth", action="store_true", help="raw dead reckoning, no drift removal")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        samples = list(dp.read_imu(dp.resolve_input_file(a.imu, "IMU")))
        if not samples:
            raise dp.InputError("no IMU rows found (expected: timestamp_ns,wx,wy,wz,ax,ay,az)")
        ins = dp.run_estimator(samples, dp.Config())
        poses = ins.poses if a.no_smooth else dp.remove_drift(ins.poses, a.smooth)
        if len(poses) != len(samples):
            raise RuntimeError(f"internal error: {len(poses)} poses for {len(samples)} rows")
        out = Path(a.output).expanduser().resolve()
        dp.atomic_write_text(out, dp.format_tum(poses))
        log.info("wrote %s (%d poses, one per IMU row)", out, len(poses))
    except dp.InputError as e:
        log.error("error: %s", e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
