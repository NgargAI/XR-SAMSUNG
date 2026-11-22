# Path Finder - Track 03. Run `make help` for targets.
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
BIN      := build/pathfinder
IMU      ?= data/gyro_accel.csv
VICON    ?= data/vicon_capture_quaternion.csv
OUT      ?= submission/estimated_trajectory.txt
TEAM     ?= CollegeName_TeamName

.PHONY: help build run test lint figures bundle clean

help:
	@echo "make build                 compile src/main.cpp -> $(BIN)"
	@echo "make run IMU=... OUT=...   estimate a trajectory with the C++ binary"
	@echo "make test                  python tests + C++ smoke test"
	@echo "make lint                  ruff + mypy --strict + bandit"
	@echo "make figures               regenerate docs/figures from the code"
	@echo "make bundle TEAM=Name      build $(TEAM).zip in the contest layout"

$(BIN): src/main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -static -o $@ $< || $(CXX) $(CXXFLAGS) -o $@ $<

build: $(BIN)

run: $(BIN)
	$(BIN) $(IMU) $(OUT) --report

test: $(BIN)
	python -m pytest -q
	$(BIN) tests/fixtures/sim_degraded_nan.csv /tmp/pf_smoke.txt
	@test "$$(wc -l < /tmp/pf_smoke.txt)" = "$$(grep -c '^[0-9]' tests/fixtures/sim_degraded_nan.csv)" && echo "C++ smoke test: one pose per IMU row"

lint:
	python -m ruff check . && python -m ruff format --check .
	python -m mypy --strict src tests
	python -m bandit -q -r src

figures:
	bash scripts/make_figures.sh

bundle: $(BIN)
	bash scripts/make_bundle.sh $(TEAM)

clean:
	rm -rf build output output_4s *.zip
