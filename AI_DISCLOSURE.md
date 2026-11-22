# AI Disclosure

Last updated: 2026-11-22 05:26 PM IST

This project was built with substantial help from an AI assistant. We disclose it in full, as the
hackathon requires.

## Tool used

- **Claude** (Anthropic), used through the Claude app as a coding and writing assistant.

## What the AI produced

- **Code:** the C++ estimator (`src/main.cpp`), the Python port and entry point
  (`src/demo_python.py`, `src/main.py`), the plotting and evaluation code (`src/plot_paths.py`,
  `src/graph_4s.py`), the learned speed model (`src/train_and_test.py`), the development tools in
  `tools/`, and the tests in `tests/`.
- **Analysis:** tuning the estimator on the supplied recording, the Vicon time synchronisation,
  the train/test evaluation, and the diagnosis of where the predictions fail.
- **Documentation and slides:** this repository's README files, the figures in `docs/figures/`,
  and the draft content of the presentation.

## What the team did

- Chose the track and defined what the system had to do.
- Supplied the IMU and Vicon recordings in `data/`.
- Directed the work: asked for the approach, reviewed results, requested changes (for example the
  train/test split and the 4-second graph), and ran the code locally in VS Code.
- Is responsible for the submission, the team details in the presentation, and the demo video.

## How the AI output was checked

- 62 automated tests, including golden-output tests and tests for every IMU-failure mode.
- Static checks: `ruff`, `mypy --strict`, `bandit`, `pip-audit`.
- Every number in the README and presentation was measured by running the code; the figures are
  regenerated from the code by `scripts/make_figures.sh`.
- The learned model was tested only on data it never saw during training or model selection.

The team should review and edit this file so it states their own contribution accurately before
submitting.
