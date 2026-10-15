"""Evaluate ML-predicted PID gains on five held-out motors."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from simulate_motor import Motor, simulate_motor

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "dataset" / "pid_dataset_5s.csv"
PREDICTIONS_PATH = ROOT / "results" / "gain_predictions_5s.csv"
MODEL_PATH = ROOT / "models" / "pid_random_forest_5s.joblib"

OUTPUT_PATH = ROOT / "results" / "controller_comparison_5s.csv"
SUMMARY_PATH = ROOT / "results" / "controller_summary_5s.csv"

FEATURES = ["R", "L", "J", "b", "Kt", "Ke"]
TARGETS = ["Kp", "Ki", "Kd"]
MOTOR_PARAMETERS = FEATURES

DURATION = 5.0
DT = 0.001
TARGET_SPEED = 100.0


def main():
    for path in [DATA_PATH, PREDICTIONS_PATH, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    dataset = pd.read_csv(DATA_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    features = bundle["features"]
    targets = bundle["targets"]

    # The prediction file identifies the five motors held out during training.
    test_ids = predictions["motor_id"].astype(int).tolist()

    if len(test_ids) != 5 or len(set(test_ids)) != 5:
        raise ValueError("Expected five unique held-out motor IDs.")

    dataset = dataset.set_index("motor_id", drop=False)

    missing_ids = [mid for mid in test_ids if mid not in dataset.index]
    if missing_ids:
        raise ValueError(f"Test motors missing from dataset: {missing_ids}")

    rows = []

    for motor_id in test_ids:
        motor_row = dataset.loc[motor_id]
        pred_row = predictions.loc[
            predictions["motor_id"] == motor_id
        ].iloc[0]

        motor = Motor(**{
            name: float(motor_row[name])
            for name in MOTOR_PARAMETERS
        })

        # Predict again using the saved model.
        X = pd.DataFrame(
            [{name: float(motor_row[name]) for name in features}],
            columns=features,
        )
        predicted_log = model.predict(X)[0]
        ml_gains = np.power(10.0, predicted_log)

        controllers = {
            "ML prediction": ml_gains,
            "Median baseline": np.array([
                pred_row[f"baseline_{name}"] for name in TARGETS
            ], dtype=float),
            "Best recorded": np.array([
                motor_row[name] for name in TARGETS
            ], dtype=float),
        }

        for controller_name, gains in controllers.items():
            result = simulate_motor(
                motor=motor,
                gains=gains,
                target_speed=TARGET_SPEED,
                duration=DURATION,
                dt=DT,
                voltage_limit=12.0,
                current_limit=12.0,
            )

            metrics = result.metrics

            rows.append({
                "motor_id": motor_id,
                "controller": controller_name,
                "Kp": float(gains[0]),
                "Ki": float(gains[1]),
                "Kd": float(gains[2]),
                "valid": metrics["valid"],
                "score": metrics["score"],
                "final_speed_rad_s": metrics.get(
                    "final_speed_rad_s", np.nan
                ),
                "overshoot_pct": metrics["overshoot_pct"],
                "final_error_pct": metrics["final_error_pct"],
                "settling_time_s": metrics["settling_time_s"],
                "peak_current_A": metrics["peak_current_A"],
                "peak_voltage_V": metrics["peak_voltage_V"],
                "diverged": metrics["diverged"],
            })

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    results.to_csv(OUTPUT_PATH, index=False)

    summary = (
        results.groupby("controller")
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
        summary["valid_controllers"] / summary["motors_tested"] * 100
    )

    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n========== HELD-OUT MOTOR EVALUATION ==========")
    print(f"Simulation duration: {DURATION} seconds")
    print(f"Target speed: {TARGET_SPEED} rad/s")
    print(f"Test motors: {test_ids}")

    print("\n--- Results for each motor ---")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- Controller summary ---")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nDetailed results saved to: {OUTPUT_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print("\nNo original evaluation files were overwritten.")


if __name__ == "__main__":
    main()
