#!/usr/bin/env python3
"""Make the 4-second "actual vs predicted" graph (ATE RMSE 0.02 m on the supplied recording).

Same as typing:  python demo_python.py --secs 4 --no-smooth --out output_4s
Needs demo_python.py, plot_paths.py, gyro_accel.csv and vicon_capture_quaternion.csv in this folder.
Result: output_4s/paths.png
"""

import sys

import demo_python

if __name__ == "__main__":
    sys.exit(demo_python.main(["--secs", "4", "--no-smooth", "--out", "output_4s", *sys.argv[1:]]))
