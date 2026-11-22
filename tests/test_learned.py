"""Tests for train_and_test.py helpers (the full train/test run is covered by the real-data run)."""

from __future__ import annotations

import argparse
import math

import numpy as np
import pytest

import train_and_test as tt


def test_quat_to_rot_identity_and_quarter_turn() -> None:
    r = tt.quat_to_rot(np.array([[1.0, 0, 0, 0], [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]]))
    assert np.allclose(r[0], np.eye(3))
    assert np.allclose(r[1] @ [1, 0, 0], [0, 1, 0])


def test_window_features_has_134_values_and_removes_gravity() -> None:
    n = int(tt.WINDOW_S * tt.IMU_HZ)
    still = tt.window_features(np.zeros((n, 3)), np.tile([0, 0, tt.GRAVITY], (n, 1)), np.array([0.0, 0, 1]))
    assert still.shape == (134,)
    assert np.allclose(still[60:120], 0)  # motion-acceleration bins are zero when only gravity is felt


def test_integrate_path_rotates_body_velocity_into_world() -> None:
    n = 40  # 2 s at 20 Hz
    r = np.repeat(tt.quat_to_rot(np.array([[math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]])), n, 0)
    p = tt.integrate_path(np.tile([1.0, 0, 0], (n, 1)), r)  # 1 m/s forward, phone yawed 90 deg
    assert np.allclose(p[-1], [0, 2, 0])


class ConstModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value)


def test_predict_velocity_zeroes_speeds_below_the_gate() -> None:
    slow = tt.predict_velocity([ConstModel(0.01)] * 3, np.zeros((5, 134)))
    fast = tt.predict_velocity([ConstModel(0.5)] * 3, np.zeros((5, 134)))
    assert np.all(slow == 0) and np.allclose(fast, 0.5)


def test_aligned_error_is_zero_for_a_rigidly_moved_copy() -> None:
    q = np.random.default_rng(0).normal(size=(50, 3))
    assert tt.aligned_error(q @ tt.quat_to_rot(np.array([[0.6, 0, 0.8, 0]]))[0].T + 2.0, q) < 1e-9


@pytest.mark.parametrize("bad", ["0", "1", "-0.5", "abc"])
def test_train_fraction_must_be_sensible(bad: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        tt.fraction(bad)
