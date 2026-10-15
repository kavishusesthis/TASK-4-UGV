"""DC motor + discrete PID simulation used by the PID auto-tuner project."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np

@dataclass(frozen=True)
class Motor:
    R: float
    L: float
    J: float
    b: float
    Kt: float
    Ke: float

    def validate(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Motor parameter {name} must be finite and > 0")

@dataclass
class SimulationResult:
    time: np.ndarray
    speed: np.ndarray
    current: np.ndarray
    voltage: np.ndarray
    error: np.ndarray
    metrics: dict

def simulate_motor(motor: Motor, gains, target_speed=100.0, duration=1.0, dt=0.001,
                   voltage_limit=12.0, current_limit=12.0):
    """Simulate a motor with voltage-limited PID control.

    Electrical current is advanced using the exact RL solution for a constant
    speed during the step; mechanical speed uses a semi-implicit Euler update.
    Derivative is taken on measured speed to avoid derivative kick.
    """
    motor.validate()
    Kp, Ki, Kd = map(float, gains)
    if any((not math.isfinite(g) or g < 0) for g in (Kp, Ki, Kd)):
        raise ValueError("PID gains must be finite and non-negative")
    settings = {
        "target_speed": target_speed,
        "duration": duration,
        "dt": dt,
        "voltage_limit": voltage_limit,
        "current_limit": current_limit,
    }
    for name, value in settings.items():
        try:
            valid = math.isfinite(value) and value > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"{name} must be finite and > 0")

    n = int(round(duration / dt)) + 1
    time = np.linspace(0.0, duration, n)
    speed = np.zeros(n); current = np.zeros(n); voltage = np.zeros(n); error = np.zeros(n)
    integral = 0.0
    previous_speed = 0.0
    diverged_at = None
    for k in range(n - 1):
        e = target_speed - speed[k]
        derivative_on_measurement = (speed[k] - previous_speed) / dt
        raw_v = Kp * e + Ki * integral - Kd * derivative_on_measurement
        v = float(np.clip(raw_v, -voltage_limit, voltage_limit))
        # Conditional integration: integrate unless saturation and error push further into it.
        saturated_high = raw_v > voltage_limit and e > 0
        saturated_low = raw_v < -voltage_limit and e < 0
        if not (saturated_high or saturated_low):
            integral += e * dt
        # Recalculate after integration for the logged applied voltage.
        raw_v = Kp * e + Ki * integral - Kd * derivative_on_measurement
        v = float(np.clip(raw_v, -voltage_limit, voltage_limit))
        voltage[k] = v
        # Exact solution of L di/dt = V - R i - Ke*w for fixed w during dt.
        decay = math.exp(-motor.R * dt / motor.L)
        equilibrium_i = (v - motor.Ke * speed[k]) / motor.R
        current[k + 1] = equilibrium_i + (current[k] - equilibrium_i) * decay
        # Semi-implicit mechanical update.
        acceleration = (motor.Kt * current[k + 1] - motor.b * speed[k]) / motor.J
        speed[k + 1] = speed[k] + dt * acceleration
        error[k] = e
        previous_speed = speed[k]
        if not (math.isfinite(speed[k + 1]) and math.isfinite(current[k + 1])):
            diverged_at = k + 1
            speed[k + 1:] = np.nan; current[k + 1:] = np.nan
            voltage[k + 1:] = np.nan; error[k + 1:] = np.nan
            break
    voltage[-1] = voltage[-2] if n > 1 else 0.0
    error[-1] = target_speed - speed[-1] if math.isfinite(speed[-1]) else np.nan
    metrics = calculate_metrics(time, speed, current, voltage, target_speed,
                                current_limit=current_limit, voltage_limit=voltage_limit,
                                diverged=diverged_at is not None)
    return SimulationResult(time, speed, current, voltage, error, metrics)

def calculate_metrics(time, speed, current, voltage, target_speed, current_limit=12.0,
                      voltage_limit=12.0, diverged=False):
    finite = np.isfinite(speed) & np.isfinite(current) & np.isfinite(voltage)
    if not finite.all() or diverged:
        return {"valid": False, "score": 1e9, "overshoot_pct": float("inf"),
                "final_error_pct": float("inf"), "settling_time_s": float("inf"),
                "peak_current_A": float("inf"), "peak_voltage_V": float("inf"),
                "iae": float("inf"), "diverged": True}
    final_speed = float(speed[-1])
    overshoot_pct = max(0.0, (float(np.max(speed)) - target_speed) / target_speed * 100.0)
    final_error_pct = abs(final_speed - target_speed) / target_speed * 100.0
    band = 0.02 * target_speed
    outside = np.where(np.abs(speed - target_speed) > band)[0]
    settling_time = 0.0 if len(outside) == 0 else (float(time[outside[-1] + 1]) if outside[-1] + 1 < len(time) else float("inf"))
    peak_current = float(np.max(np.abs(current)))
    peak_voltage = float(np.max(np.abs(voltage)))
    integrate = getattr(np, "trapezoid", np.trapz)
    iae = float(integrate(np.abs(target_speed - speed), time))
    valid = (overshoot_pct <= 20.0 and final_error_pct <= 2.0 and
             peak_current <= current_limit + 1e-9 and peak_voltage <= voltage_limit + 1e-9)
    # Normalized, interpretable objective; penalties make invalid controllers unattractive.
    score = (iae / max(target_speed * time[-1], 1e-9)
             + 0.02 * overshoot_pct
             + 0.25 * final_error_pct
             + 0.2 * (settling_time if math.isfinite(settling_time) else 10.0))
    if peak_current > current_limit:
        score += 10.0 + (peak_current - current_limit)
    if not valid:
        score += 100.0
    return {"valid": bool(valid), "score": float(score), "final_speed_rad_s": final_speed,
            "overshoot_pct": overshoot_pct, "final_error_pct": final_error_pct,
            "settling_time_s": settling_time, "peak_current_A": peak_current,
            "peak_voltage_V": peak_voltage, "iae": iae, "diverged": False}
