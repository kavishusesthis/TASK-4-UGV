"""Evaluate v2 ML-predicted PID gains on held-out simulated motors."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from simulate_motor import Motor, simulate_motor

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "dataset" / "pid_dataset_5s_v2.csv"
LOG_PATH = ROOT / "dataset" / "experiment_log_5s_v2.csv"
MODEL_PATH = ROOT / "models" / "pid_random_forest_5s_v2.joblib"

COMPARISON_PATH = ROOT / "results" / "controller_comparison_5s_v2.csv"
SUMMARY_PATH = ROOT / "results" / "controller_summary_5s_v2.csv"

FEATURES = ["R", "L", "J", "b", "Kt", "Ke"]
TARGETS = ["Kp", "Ki", "Kd"]

DURATION = 5.0
DT = 0.001


def evaluate_one(motor, gains):
    """Run one closed-loop simulation and return its metrics."""
    sim = simulate_motor(
        motor,
        gains,
        target_speed=100.0,
        duration=DURATION,
        dt=DT,
        voltage_limit=12.0,
        current_limit=12.0,
    )

    metrics = dict(sim.metrics)

    return {
        "valid": bool(metrics.get("valid", False)),
        "score": float(metrics.get("score", np.nan)),
        "final_speed_rad_s": float(
            metrics.get("final_speed_rad_s", np.nan)
        ),
        "overshoot_pct": float(metrics.get("overshoot_pct", np.nan)),
        "final_error_pct": float(metrics.get("final_error_pct", np.nan)),
        "settling_time_s": float(metrics.get("settling_time_s", np.nan)),
        "peak_current_A": float(metrics.get("peak_current_A", np.nan)),
        "peak_voltage_V": float(metrics.get("peak_voltage_V", np.nan)),
        "diverged": bool(metrics.get("diverged", False)),
    }


def main():
    data = pd.read_csv(DATA_PATH)
    log = pd.read_csv(LOG_PATH)
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    training_ids = bundle["training_motor_ids"]
    test_ids = bundle["test_motor_ids"]

    # Baseline gains are calculated using training motors only.
    training_data = data[data["motor_id"].isin(training_ids)]
    baseline_log = np.log10(training_data[TARGETS]).median().to_numpy()
    baseline_gains = 10 ** baseline_log

    # Get each motor's parameters from the experiment log.
    motor_table = (
        log.sort_values(["motor_id", "trial_id"])
        .drop_duplicates("motor_id")
        .set_index("motor_id")
    )

    best_table = data.set_index("motor_id")
    records = []

    for motor_id in test_ids:
        row = motor_table.loc[motor_id]

        motor = Motor(
            R=float(row["R"]),
            L=float(row["L"]),
            J=float(row["J"]),
            b=float(row["b"]),
            Kt=float(row["Kt"]),
            Ke=float(row["Ke"]),
        )

        X = pd.DataFrame(
            [{feature: float(row[feature]) for feature in FEATURES}],
            columns=FEATURES,
        )

        predicted_log = model.predict(X)[0]
        ml_gains = 10 ** predicted_log

        controllers = [
            ("ML prediction", ml_gains),
            ("Median baseline", baseline_gains),
        ]

        # Reference only: the search-derived best gains for this motor.
        if motor_id in best_table.index:
            best_row = best_table.loc[motor_id]
            controllers.append((
                "Best recorded",
                best_row[TARGETS].to_numpy(dtype=float),
            ))

        for name, gains in controllers:
            try:
                metrics = evaluate_one(motor, gains)
            except Exception as exc:
                print(f"WARNING: motor {motor_id}, {name}: {exc}")
                metrics = {
                    "valid": False,
                    "score": np.nan,
                    "final_speed_rad_s": np.nan,
                    "overshoot_pct": np.nan,
                    "final_error_pct": np.nan,
                    "settling_time_s": np.nan,
                    "peak_current_A": np.nan,
                    "peak_voltage_V": np.nan,
                    "diverged": True,
                }

            records.append({
                "motor_id": int(motor_id),
                "controller": name,
                "Kp": float(gains[0]),
                "Ki": float(gains[1]),
                "Kd": float(gains[2]),
                **metrics,
            })

        print(f"Evaluated motor {motor_id}")

    comparison = pd.DataFrame(records)
    comparison.to_csv(COMPARISON_PATH, index=False)

    summary = (
        comparison.groupby("controller")
        .agg(
            motors_tested=("motor_id", "nunique"),
            valid_controllers=("valid", "sum"),
            mean_score=("score", "mean"),
            mean_final_error_pct=("final_error_pct", "mean"),
            mean_overshoot_pct=("overshoot_pct", "mean"),
            mean_settling_time_s=("settling_time_s", "mean"),
            max_peak_current_A=("peak_current_A", "max"),
            max_peak_voltage_V=("peak_voltage_V", "max"),
        )
        .reset_index()
    )

    summary["valid_rate_pct"] = (
        100 * summary["valid_controllers"] / summary["motors_tested"]
    )

    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n========== V2 CLOSED-LOOP SUMMARY ==========")
    print(summary.round(4).to_string(index=False))

    print("\n========== PER-MOTOR RESULTS ==========")
    print(
        comparison[
            [
                "motor_id", "controller", "valid", "score",
                "final_error_pct", "overshoot_pct",
                "settling_time_s", "peak_current_A",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nSaved detailed results:", COMPARISON_PATH)
    print("Saved summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
