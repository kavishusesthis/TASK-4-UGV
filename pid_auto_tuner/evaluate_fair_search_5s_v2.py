"""Fair comparison of ML-guided local search and random search.

Uses the saved model and its existing 25 test motors.
Each search method evaluates 125 candidates per motor.
Existing datasets, models, and result files are not overwritten.
"""

from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from simulate_motor import Motor, simulate_motor


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
MODEL_PATH = ROOT / "models" / "pid_random_forest_5s_v2.joblib"
RESULT_DIR = ROOT / "results"

LOG_PATH = DATA_DIR / "experiment_log_5s_v2.csv"
DURATION = 5.0
DT = 0.001
SEARCH_BUDGET = 125

FEATURES = ["R", "L", "J", "b", "Kt", "Ke"]
TARGETS = ["Kp", "Ki", "Kd"]

# Match the gain ranges used by generate_dataset_5s_v2.py.
GAIN_BOUNDS = np.array([
    [0.001, 2.0],    # Kp
    [0.001, 20.0],   # Ki
    [1e-5, 0.05],    # Kd
])

LOCAL_FACTORS = [0.50, 0.75, 1.00, 1.25, 1.50]

METRIC_COLUMNS = [
    "score",
    "final_speed_rad_s",
    "overshoot_pct",
    "final_error_pct",
    "settling_time_s",
    "peak_current_A",
    "peak_voltage_V",
    "iae",
]


def random_gains(rng):
    """Sample gains log-uniformly, matching the dataset generator."""
    return np.exp(
        rng.uniform(
            np.log(GAIN_BOUNDS[:, 0]),
            np.log(GAIN_BOUNDS[:, 1]),
        )
    )


def evaluate_candidate(motor, gains):
    """Run one simulation and return its metrics."""
    try:
        result = simulate_motor(
            motor,
            tuple(float(x) for x in gains),
            duration=DURATION,
            dt=DT,
        )

        metrics = dict(result.metrics)
        metrics["valid"] = bool(metrics.get("valid", False))
        metrics["score"] = float(metrics.get("score", 1e9))

        if not np.isfinite(metrics["score"]):
            metrics["score"] = 1e9
            metrics["valid"] = False

        return metrics

    except (ValueError, FloatingPointError, OverflowError) as exc:
        return {
            "valid": False,
            "score": 1e9,
            "error": str(exc),
        }


def choose_best(candidates):
    """Prefer valid candidates; within each group, prefer lower scores."""
    return min(
        candidates,
        key=lambda item: (
            not item["metrics"]["valid"],
            item["metrics"]["score"],
        ),
    )


def record_result(motor_id, method, selected, evaluations, unique_candidates):
    row = {
        "motor_id": int(motor_id),
        "method": method,
        "candidate_evaluations": int(evaluations),
        "unique_candidates": int(unique_candidates),
        "Kp": float(selected["gains"][0]),
        "Ki": float(selected["gains"][1]),
        "Kd": float(selected["gains"][2]),
    }

    metrics = selected["metrics"]
    row["valid"] = bool(metrics["valid"])

    for name in METRIC_COLUMNS:
        row[name] = metrics.get(name, np.nan)

    return row


def main():
    RESULT_DIR.mkdir(exist_ok=True)

    print("Loading the existing trained model...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle.get("features", FEATURES)
    test_ids = [int(x) for x in bundle["test_motor_ids"]]

    print("Loading existing experiment log...")
    log = pd.read_csv(LOG_PATH)

    # Each motor has many trials but the physical parameters are constant.
    motor_table = log.groupby("motor_id", sort=False).first()

    print(f"Saved test motors: {len(test_ids)}")
    print(f"Search budget per method per motor: {SEARCH_BUDGET}")
    print(f"Simulation duration: {DURATION}s")
    print(f"Time step: {DT}s")
    print()

    rows = []

    for position, motor_id in enumerate(test_ids, start=1):
        if motor_id not in motor_table.index:
            raise ValueError(f"Motor {motor_id} is missing from the experiment log.")

        motor_row = motor_table.loc[motor_id]

        motor = Motor(
            R=float(motor_row["R"]),
            L=float(motor_row["L"]),
            J=float(motor_row["J"]),
            b=float(motor_row["b"]),
            Kt=float(motor_row["Kt"]),
            Ke=float(motor_row["Ke"]),
        )

        X = pd.DataFrame(
            [{name: float(motor_row[name]) for name in features}],
            columns=features,
        )

        # The saved model predicts log10(Kp), log10(Ki), log10(Kd).
        predicted = np.power(10.0, model.predict(X)[0])
        predicted = np.clip(
            predicted,
            GAIN_BOUNDS[:, 0],
            GAIN_BOUNDS[:, 1],
        )

        # Method 1: ML local search, exactly 5^3 = 125 candidates.
        local_candidates = []
        for factors in product(LOCAL_FACTORS, repeat=3):
            gains = predicted * np.asarray(factors, dtype=float)
            gains = np.clip(
                gains,
                GAIN_BOUNDS[:, 0],
                GAIN_BOUNDS[:, 1],
            )
            local_candidates.append(gains)

        local_evaluations = [
            {"gains": gains, "metrics": evaluate_candidate(motor, gains)}
            for gains in local_candidates
        ]
        local_best = choose_best(local_evaluations)

        rows.append(
            record_result(
                motor_id,
                "ML local search",
                local_best,
                evaluations=len(local_evaluations),
                unique_candidates=len({
                    tuple(np.round(item["gains"], 12))
                    for item in local_evaluations
                }),
            )
        )

        # Method 2: Random search, exactly 125 independently sampled candidates.
        rng = np.random.default_rng(20261015 + motor_id)
        random_candidates = [
            random_gains(rng) for _ in range(SEARCH_BUDGET)
        ]

        random_evaluations = [
            {"gains": gains, "metrics": evaluate_candidate(motor, gains)}
            for gains in random_candidates
        ]
        random_best = choose_best(random_evaluations)

        rows.append(
            record_result(
                motor_id,
                "Random search",
                random_best,
                evaluations=len(random_evaluations),
                unique_candidates=len({
                    tuple(np.round(item["gains"], 12))
                    for item in random_evaluations
                }),
            )
        )

        # Diagnostic baseline: evaluate the ML prediction itself once.
        # This is a one-evaluation baseline, not a 125-evaluation search.
        prediction_metrics = evaluate_candidate(motor, predicted)
        prediction_item = {
            "gains": predicted,
            "metrics": prediction_metrics,
        }

        rows.append(
            record_result(
                motor_id,
                "Direct ML prediction",
                prediction_item,
                evaluations=1,
                unique_candidates=1,
            )
        )

        print(
            f"[{position:02d}/{len(test_ids)}] Motor {motor_id}: "
            f"ML search valid={local_best['metrics']['valid']}, "
            f"random valid={random_best['metrics']['valid']}, "
            f"direct ML valid={prediction_metrics['valid']}"
        )

    detail = pd.DataFrame(rows)

    detail_path = RESULT_DIR / "fair_search_comparison_5s_v2.csv"
    summary_path = RESULT_DIR / "fair_search_summary_5s_v2.csv"

    detail.to_csv(detail_path, index=False)

    summary_rows = []

    for method, group in detail.groupby("method", sort=False):
        valid = group[group["valid"]]

        summary = {
            "method": method,
            "test_motors": len(group),
            "valid_motors": int(group["valid"].sum()),
            "valid_rate_pct": 100.0 * group["valid"].mean(),
            "mean_score_all_motors": group["score"].mean(),
            "mean_score_valid_only": (
                valid["score"].mean() if len(valid) else np.nan
            ),
            "mean_candidate_evaluations": group[
                "candidate_evaluations"
            ].mean(),
        }

        for column in [
            "final_error_pct",
            "overshoot_pct",
            "settling_time_s",
            "peak_current_A",
            "peak_voltage_V",
        ]:
            summary[f"mean_{column}_valid_only"] = (
                valid[column].mean() if len(valid) else np.nan
            )

        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    print("\n========== FAIR SEARCH SUMMARY ==========")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n========== FILES CREATED ==========")
    print(detail_path)
    print(summary_path)

    print("\nNo model training, dataset generation, Git staging, or commits were performed.")


if __name__ == "__main__":
    main()
