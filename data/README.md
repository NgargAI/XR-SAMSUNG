# Data

| File | Contents |
|---|---|
| `gyro_accel.csv` | phone IMU, 200 Hz, 255,029 rows (about 21 minutes): `timestamp[ns], gx, gy, gz [rad/s], ax, ay, az [m/s^2]` |
| `vicon_capture_quaternion.csv` | Vicon motion capture of the same session, 100 Hz: `Frame, Sub Frame, RX, RY, RZ, RW, TX, TY, TZ` (mm). Frames with occlusions have empty fields |

The Vicon file has frame numbers but no timestamps. `src/demo_python.py` assumes 100 Hz and finds
the clock offset by correlating rotation speed seen by Vicon and by the gyro (0.265 s,
correlation 0.97 for this pair).
