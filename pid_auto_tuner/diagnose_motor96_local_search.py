"""Inspect all 125 ML-assisted local-search candidates for motor 96."""

from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from simulate_motor import Motor, simulate_motor

ROOT = Path(__file__).resolve().parent

LOG_PATH = ROOT / "dataset" / "experiment_log_5s_v2.csv"
MODEL_PATH = ROOT / "models" / "pid_random_forest_5s_v2.joblib"
OUTPUT_PATH = ROOT / "results" / "motor96_local_candidates_diagnostic.csv"

MOTOR_ID = 96
FEATURES = ["R", "L", "J", "b", "Kt", "Ke"]
TARGETS = ["Kp", "Ki", "Kd"]
SEARCH_FACTORS = [0.50, 0.75, 1.00, 1.25, 1.50]


def main():
    log = pd.read_csv(LOG_PATH)
    bundle = joblib.load(MODEL_PATH)

    # Get this motor's physical characteristics.
    motor_rows = (
        log.sort_values(["motor_id", "trial_id"])
        .drop_duplicates("motor_id")
        .set_index("motor_id")
    )

    if MOTOR_ID not in motor_rows.index:
        raise ValueError(f"Motor {MOTOR_ID} not found in {LOG_PATH}")

    row = motor_rows.loc[MOTOR_ID]

    motor = Motor(
        R=float(row["R"]),
        L=float(row["L"]),
        J=float(row["J"]),
        b=float(row["b"]),
        Kt=float(row["Kt"]),
        Ke=float(row["Ke"]),
    )

    # Predict the starting gains using the trained model.
    X = pd.DataFrame(
        [{name: float(row[name]) for name in FEATURES}],
        columns=FEATURES,
    )

    model = bundle["model"]
    predicted_gains = 10 ** model.predict(X)[0]

    print(f"Diagnosing motor {MOTOR_ID}")
    print("Motor characteristics:")
    for feature in FEATURES:
        print(f"  {feature} = {float(row[feature]):.8g}")

    print("\nML-predicted starting gains:")
    for name, value in zip(TARGETS, predicted_gains):
        print(f"  {name} = {value:.8g}")

    records = []

    # Evaluate every combination of local-search multipliers.
    for factors in product(SEARCH_FACTORS, repeat=3):
        gains = predicted_gains * np.asarray(factors, dtype=float)

        result = simulate_motor(
            motor,
            gains,
            target_speed=100.0,
            duration=5.0,
            dt=0.001,
            voltage_limit=12.0,
            current_limit=12.0,
        )

        metrics = result.metrics

        records.append({
            "motor_id": MOTOR_ID,
            "Kp_factor": factors[0],
            "Ki_factor": factors[1],
            "Kd_factor": factors[2],
            "Kp": float(gains[0]),
            "Ki": float(gains[1]),
            "Kd": float(gains[2]),
            **metrics,
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)

    valid = df[df["valid"]].sort_values("score")
    lowest_score = df.sort_values("score").head(1)
    lowest_current = df.sort_values("peak_current_A").head(5)

    print(f"\nCandidates evaluated: {len(df)}")
    print(f"Valid candidates: {int(df['valid'].sum())}/{len(df)}")
    print(f"Saved all candidates to:\n{OUTPUT_PATH}")

    columns = [
        "Kp_factor", "Ki_factor", "Kd_factor",
        "Kp", "Ki", "Kd", "valid", "score",
        "overshoot_pct", "final_error_pct",
        "peak_current_A", "peak_voltage_V",
    ]

    if not valid.empty:
        print("\nBest valid candidates:")
        print(valid[columns].head(10).round(6).to_string(index=False))
    else:
        print("\nNo valid candidate was found in this local grid.")

    print("\nLowest-score candidate:")
    print(lowest_score[columns].round(6).to_string(index=False))

    print("\nFive candidates with the lowest peak current:")
    print(lowest_current[columns].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
