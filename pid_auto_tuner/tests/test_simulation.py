import math

import numpy as np
import pytest

from simulate_motor import Motor, simulate_motor


@pytest.fixture
def motor():
    return Motor(1.2, 0.01, 0.001, 0.0002, 0.05, 0.05)


def test_voltage_is_clipped(motor):
    result = simulate_motor(motor, (100, 100, 1))

    assert np.all(np.isfinite(result.voltage))
    assert np.max(np.abs(result.voltage)) <= 12.0 + 1e-9


def test_normal_simulation_metrics_are_finite(motor):
    result = simulate_motor(motor, (0.3, 5.0, 0.003))
    metrics = result.metrics

    for name in (
        "score",
        "final_speed_rad_s",
        "overshoot_pct",
        "final_error_pct",
        "settling_time_s",
        "peak_current_A",
        "peak_voltage_V",
        "iae",
    ):
        assert math.isfinite(metrics[name]), f"{name} should be finite"

    assert metrics["score"] >= 0
    assert metrics["peak_current_A"] >= 0
    assert metrics["peak_voltage_V"] >= 0
    assert metrics["diverged"] is False


@pytest.mark.parametrize(
    "setting,value",
    [
        ("target_speed", float("inf")),
        ("target_speed", float("nan")),
        ("target_speed", 0),
        ("target_speed", -100),
        ("duration", float("inf")),
        ("duration", float("nan")),
        ("duration", 0),
        ("dt", float("inf")),
        ("dt", float("nan")),
        ("dt", 0),
        ("voltage_limit", float("inf")),
        ("voltage_limit", float("nan")),
        ("voltage_limit", 0),
        ("current_limit", float("inf")),
        ("current_limit", float("nan")),
        ("current_limit", 0),
        ("current_limit", -1),
    ],
)
def test_invalid_simulation_settings_are_rejected(motor, setting, value):
    with pytest.raises(ValueError):
        simulate_motor(motor, (0.3, 5.0, 0.003), **{setting: value})


def test_current_limit_violation_makes_controller_invalid(motor):
    result = simulate_motor(
        motor,
        (0.3, 5.0, 0.003),
        current_limit=0.001,
    )

    assert result.metrics["peak_current_A"] > 0.001
    assert result.metrics["valid"] is False


def test_custom_voltage_limit_is_respected(motor):
    result = simulate_motor(motor, (100, 100, 1), voltage_limit=3.0)

    assert np.max(np.abs(result.voltage)) <= 3.0 + 1e-9


def test_time_array_matches_requested_step_when_divisible(motor):
    result = simulate_motor(
        motor,
        (0.3, 5.0, 0.003),
        duration=1.0,
        dt=0.001,
    )

    assert len(result.time) == 1001
    assert np.allclose(np.diff(result.time), 0.001)
