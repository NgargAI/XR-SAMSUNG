# Path Finder source: how to compile and run

Last updated: 2026-11-22 05:26 PM IST

## C++ entry point (`main.cpp`)

Single file, C++17, no dependencies.

```bash
g++ -O2 -std=c++17 -static -o pathfinder main.cpp      # or, from the repo root: make build
./pathfinder imu0/data.csv estimated_trajectory.txt
```

| Option | Default | Meaning |
|---|---|---|
| `--smooth SEC` / `--no-smooth` | 2 | offline drift removal (zero-phase high-pass time constant) |
| `--kp`, `--ki` | 4, 1 | gravity-correction gains (roll/pitch, gyro bias) |
| `--vleak` | 0.2 | velocity leak while the IMU is healthy (bounded-motion prior) |
| `--zupt-speed`, `--zupt-time`, `--no-zupt` | 3, 0.15 | zero-velocity update gate |
| `--dead-decay` | 0 | velocity decay after the IMU dies (0 = constant velocity) |
| `--stuck-n`, `--bad-n` | 20, 20 | frozen / invalid rows before the IMU is declared dead |
| `--report` | | print peak memory (VmHWM) to stderr |

Input: EuRoC `imu0/data.csv` (`timestamp[ns], w_x, w_y, w_z, a_x, a_y, a_z`; header lines are
skipped). Output: TUM `timestamp[s] tx ty tz qx qy qz qw`, exactly one pose per IMU row, with the
input timestamp copied to the nanosecond.

## Python entry point (`main.py`)

```bash
pip install numpy matplotlib
python main.py imu0/data.csv estimated_trajectory.txt [--no-smooth]
```

Same algorithm as `main.cpp`, in pure Python (`demo_python.py`); slower and uses far more memory,
so use the C++ binary for the contest budgets.

## Other scripts

| Script | Purpose |
|---|---|
| `demo_python.py` | estimate + compare with Vicon + draw `output/paths.png` |
| `plot_paths.py` | draw a TUM trajectory against ground truth; error metrics |
| `graph_4s.py` | one-click 4-second graph (`output_4s/paths.png`) |
| `train_and_test.py` | learned speed model, train on the first 90 %, test on the rest (needs `scikit-learn`) |

Every script prints its options with `--help`. Exit code 2 means bad input, with a message.
