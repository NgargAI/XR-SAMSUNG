# Demo video

Last updated: 2026-11-22 06:20 PM IST

**Pipeline walkthrough video (2:45):** [watch on Google Drive](https://drive.google.com/file/d/1KbKA4LGNumPrQeiG0KWNj3ujcqOtw9zX/view?usp=drivesdk)

1080p, narrated (synthetic voice, see [`AI_DISCLOSURE.md`](../AI_DISCLOSURE.md)), with captions. It
walks through the problem, the solution and architecture, a real run on the supplied recording
(255,029 rows in, 255,029 poses out, 2.1 MB peak memory), a dead-IMU run, and the results.

## Script the video follows

| Time | Show | Say |
|---|---|---|
| 0:00–0:20 | Title slide | Track 03, Path Finder: 6-DoF trajectory from a degrading IMU, within 8 MB and 20 ms per second |
| 0:20–0:50 | Architecture slide | Streaming strapdown INS, gravity aiding, zero-velocity updates, failure detection, offline drift removal |
| 0:50–1:20 | Terminal: `make build` then `./build/pathfinder data/gyro_accel.csv out.txt --report` | 255,029 rows in, 255,029 poses out, 2.1 MB peak memory, about 1 second for 21 minutes of data |
| 1:20–1:50 | `python src/graph_4s.py data/gyro_accel.csv --vicon data/vicon_capture_quaternion.csv`, open the graph | Over 4 seconds the prediction follows Vicon to 2.3 cm |
| 1:50–2:20 | The 60-second graph and the learned-model graph | IMU-only drift over minutes; the learned speed model halves the error on unseen data |
| 2:20–2:50 | A degraded file (NaN rows at the end) | The IMU is declared dead and the estimator dead-reckons the last 5 seconds |
| 2:50–3:00 | Repository page | Code, tests, Docker and README are on GitHub |
