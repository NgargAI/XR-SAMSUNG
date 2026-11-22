# Changelog

All notable changes to Path Finder. Format: [Keep a Changelog](https://keepachangelog.com).

## 2026-11-22

### Added
- Demo video link (Google Drive) in the README, `video/README.md` and the deck (slides 7 and 10,
  as clickable links).

### Changed
- Team details (Trinity Stack, TIET) in the README and presentation; `make bundle` writes
  `TIET_TrinityStack.zip`. The public deck lists member names without phone numbers.
- Brownie slides show measured numbers instead of trajectory plots.

### Added
- Repository in the Track 03 submission layout: `src/main.cpp` (C++ entry point), `src/main.py`
  (standalone Python entry point), `submission/` (trajectory + `metadata.json`), `data/`,
  `docs/figures/`, `presentation/`, `video/`, `AI_DISCLOSURE.md`.
- Top-level `Makefile` (`build`, `run`, `test`, `lint`, `figures`, `bundle`), `Dockerfile`
  (non-root runtime), GitHub Actions CI, `scripts/make_bundle.sh`, `scripts/make_figures.sh`.
- `train_and_test.py` also draws the true-vs-predicted speed graph.
- Tests for `src/main.py`.

### Changed
- README states the measured C++/Python agreement precisely (8 quaternions differ by 1e-9; up to
  4.7 cm with drift removal) instead of "byte-identical".

## 2026-11-21

### Added
- `train_and_test.py`: learned speed model (MLP on 134 IMU features, Vicon as teacher), trained on
  the first 80 % of a recording and scored on the unseen last 20 %. On the supplied recording the
  whole-test-part error drops from 0.98 m (current code) to 0.39 m. Deterministic (fixed seed).
- Default split is 90 % train / 10 % test (`--train-frac 0.8` for 80/20). On the last 10 % the
  error is 0.54 m vs 1.30 m for the current code.
- `tests/test_learned.py` (9 tests); `scikit-learn==1.9.1` in `requirements.txt`.

## 2026-11-20

### Changed
- Rewrote `demo_python.py` and `plot_paths.py`: typed (`mypy --strict` clean), structured into small
  testable functions, `logging` instead of `print`. Estimator output is byte-identical to the
  previous release on the supplied recording and on the degraded test files.
- `demo_python.py` now calls the plotter directly instead of starting a second Python process.
- IMU death is now reported from the first bad row (was the 20th for NaN/zero faults); output unchanged.

### Added
- Input validation with clear messages and exit code 2: missing/directory paths, files without IMU
  rows, wrong Vicon layout, negative or non-finite options, non-overlapping ground truth.
- Crash-safe (temp file + rename) output writes.
- Log lines for invalid IMU rows and the time the IMU was declared dead.
- `--vicon-hz`, `--no-plot`, `-v/--verbose` options.
- 50 tests (golden parity, fault handling, parsing, maths, drift removal, Vicon sync, plotting) plus
  an optional real-data test; `pyproject.toml` with ruff/mypy settings; pinned `requirements*.txt`.

## 2026-11-19

### Added
- Offline drift removal (`--smooth`, default 2 s; `--no-smooth` to disable).
- Pure-Python port of `pathfinder.cpp` with Vicon comparison plot.
