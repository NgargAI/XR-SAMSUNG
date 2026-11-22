#!/usr/bin/env bash
# Regenerate every figure in docs/figures from the code (needs: pip install -r requirements.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT
imu=data/gyro_accel.csv
vic=data/vicon_capture_quaternion.csv
python src/demo_python.py "$imu" --vicon "$vic" --secs 4 --no-smooth --out "$work/4s"
python src/demo_python.py "$imu" --vicon "$vic" --secs 60 --out "$work/60s"
python src/train_and_test.py "$imu" --vicon "$vic" --out "$work/learned"
cp -- "$work/4s/paths.png"                         docs/figures/01_first_4s_actual_vs_predicted.png
cp -- "$work/60s/paths.png"                        docs/figures/02_first_60s_drift.png
cp -- "$work/learned/learned_test.png"             docs/figures/03_learned_model_test_10pct.png
cp -- "$work/learned/speed_true_vs_predicted.png"  docs/figures/04_speed_true_vs_predicted.png
echo "figures updated in docs/figures"
