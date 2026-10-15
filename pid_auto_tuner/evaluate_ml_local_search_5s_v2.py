"""Compare direct ML predictions with ML-assisted local PID search."""

from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from simulate_motor import Motor, simulate_motor

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "dataset" / "pid_dataset_5s_v2.csv"
LOG_PATH = ROOT / "dataset" / "experiment_log_5s_v2.csv"
MODEL_PATH = ROOT / "models" / "pid_random_forest_5s_v2.joblib"

COMPARISON_PATH = ROOT / "results" / "ml_local_search_comparison_5s_v2.csv"
SUMMARY_PATH = ROOT / "results" / "ml_local_search_summary_5s_v2.csv"

FEATURES = ["R", "L", "J", "b", "Kt", "Ke"]
TARGETS = ["Kp", "Ki", "Kd"]

DURATION = 5.0
DT = 0.001
SEARCH_FACTORS = [0.50, 0.75, 1.00, 1.25, 1.50]


def evaluate(motor, gains):
    result = simulate_motor(
        motor,
        gains,
        target_speed=100.0,
        duration=DURATION,
        dt=DT,
        voltage_limit=12.0,
        current_limit=12.0,
    )
    return result.metrics


def main():
    data = pd.read_csv(DATA_PATH)
    log = pd.read_csv(LOG_PATH)
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    training_ids = bundle["training_motor_ids"]
    test_ids = bundle["test_motor_ids"]

    # Calculate the baseline from training motors only.
    training_data = data[data["motor_id"].isin(training_ids)]
    baseline_gains = 10 ** np.log10(
        training_data[TARGETS]
    ).median().to_numpy()

    motor_table = (
        log.sort_values(["motor_id", "trial_id"])
        .drop_duplicates("motor_id")
        .set_index("motor_id")
    )
    best_table = data.set_index("motor_id")

    records = []
    number_of_candidates = len(SEARCH_FACTORS) ** 3

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
            [{name: float(row[name]) for name in FEATURES}],
            columns=FEATURES,
        )
        ml_gains = 10 ** model.predict(X)[0]

        # Evaluate direct ML prediction and training-only median baseline.
        for name, gains in [
            ("ML prediction", ml_gains),
            ("Median baseline", baseline_gains),
        ]:
            metrics = evaluate(motor, gains)
            records.append({
                "motor_id": int(motor_id),
                "controller": name,
                "Kp": float(gains[0]),
                "Ki": float(gains[1]),
                "Kd": float(gains[2]),
                "search_candidates": 1,
                **metrics,
            })

        # Search a local grid around the ML-predicted gains.
        candidates = []
        for factors in product(SEARCH_FACTORS, repeat=3):
            gains = ml_gains * np.asarray(factors)
            metrics = evaluate(motor, gains)

            candidates.append({
                "gains": gains,
                "metrics": metrics,
            })

        valid_candidates = [
            item for item in candidates if item["metrics"]["valid"]
        ]

        if valid_candidates:
            selected = min(
                valid_candidates,
                key=lambda item: item["metrics"]["score"],
            )
        else:
            # Keep the least-bad result, but do not label it valid.
            selected = min(
                candidates,
                key=lambda item: item["metrics"]["score"],
            )

        gains = selected["gains"]
        metrics = selected["metrics"]

        records.append({
            "motor_id": int(motor_id),
            "controller": "ML local search",
            "Kp": float(gains[0]),
            "Ki": float(gains[1]),
            "Kd": float(gains[2]),
            "search_candidates": number_of_candidates,
            **metrics,
        })

        # Include the search-derived best controller as a reference.
        if motor_id in best_table.index:
            best = best_table.loc[motor_id]
            gains = best[TARGETS].to_numpy(dtype=float)
            metrics = evaluate(motor, gains)

            records.append({
                "motor_id": int(motor_id),
                "controller": "Best recorded reference",
                "Kp": float(gains[0]),
                "Ki": float(gains[1]),
                "Kd": float(gains[2]),
                "search_candidates": 0,
                **metrics,
            })

        print(
            f"Motor {motor_id}: "
            f"{len(valid_candidates)}/{number_of_candidates} "
            "local candidates valid"
        )

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
            mean_search_candidates=("search_candidates", "mean"),
        )
        .reset_index()
    )

    summary["valid_rate_pct"] = (
        100 * summary["valid_controllers"] / summary["motors_tested"]
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n========== ML LOCAL-SEARCH SUMMARY ==========")
    print(summary.round(4).to_string(index=False))

    print("\n========== RESULTS FOR PREVIOUS ML FAILURES ==========")
    failed_ids = [96, 15]
    print(
        comparison[comparison["motor_id"].isin(failed_ids)][
            [
                "motor_id", "controller", "valid", "Kp", "Ki", "Kd",
                "score", "overshoot_pct", "final_error_pct",
                "peak_current_A", "search_candidates",
            ]
        ].round(5).to_string(index=False)
    )

    print("\nDetailed results saved to:", COMPARISON_PATH)
    print("Summary saved to:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
