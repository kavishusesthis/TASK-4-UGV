"""Scoring and validity rules shared by search and evaluation."""
from simulate_motor import calculate_metrics

CONSTRAINTS = {
    "max_overshoot_pct": 20.0,
    "max_final_error_pct": 2.0,
    "max_peak_current_A": 12.0,
    "max_peak_voltage_V": 12.0,
}

def score_response(result, target_speed=100.0, current_limit=12.0, voltage_limit=12.0):
    """Return the standard metrics dictionary for a SimulationResult."""
    return calculate_metrics(result.time, result.speed, result.current, result.voltage,
                             target_speed, current_limit=current_limit,
                             voltage_limit=voltage_limit)
