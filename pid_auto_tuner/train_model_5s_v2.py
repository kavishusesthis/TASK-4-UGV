"""Train a PID gain prediction model using the v2 five-second dataset."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "dataset" / "pid_dataset_5s_v2.csv"
MODEL_PATH = ROOT / "models" / "pid_random_forest_5s_v2.joblib"
METRICS_PATH = ROOT / "results" / "training_metrics_5s_v2.json"
PREDICTIONS_PATH = ROOT / "results" / "gain_predictions_5s_v2.csv"

FEATURES = ["R", "L", "J", "b", "Kt", "Ke"]
TARGETS = ["Kp", "Ki", "Kd"]


def main():
    data = pd.read_csv(DATA_PATH)

    # Only train on controllers that passed all validity checks.
    data = data[data["valid"].astype(str).str.lower() == "true"].copy()

    if data.empty:
        raise ValueError("No valid training examples found.")

    if not np.isfinite(data[FEATURES + TARGETS].to_numpy()).all():
        raise ValueError("Dataset contains non-finite features or targets.")

    if (data[TARGETS] <= 0).any().any():
        raise ValueError("PID gains must be positive for log-space training.")

    X = data[FEATURES]
    y_log = np.log10(data[TARGETS])

    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=42,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y_log.iloc[train_idx]
    y_test = y_log.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predicted_log = model.predict(X_test)

    # Simple baseline: predict the training-set median log gain for every motor.
    baseline_log = np.tile(
        np.median(y_train.to_numpy(), axis=0),
        (len(test_idx), 1),
    )

    overall_model_mae = mean_absolute_error(
        y_test.to_numpy().ravel(),
        predicted_log.ravel(),
    )
    overall_baseline_mae = mean_absolute_error(
        y_test.to_numpy().ravel(),
        baseline_log.ravel(),
    )

    per_gain = {}
    for i, gain in enumerate(TARGETS):
        per_gain[gain] = {
            "model_log10_mae": float(
                mean_absolute_error(y_test.iloc[:, i], predicted_log[:, i])
            ),
            "baseline_log10_mae": float(
                mean_absolute_error(y_test.iloc[:, i], baseline_log[:, i])
            ),
        }

    predictions = pd.DataFrame({
        "motor_id": data["motor_id"].iloc[test_idx].to_numpy(),
    })

    actual_gains = 10 ** y_test.to_numpy()
    predicted_gains = 10 ** predicted_log
    baseline_gains = 10 ** baseline_log

    for i, gain in enumerate(TARGETS):
        predictions[f"actual_{gain}"] = actual_gains[:, i]
        predictions[f"predicted_{gain}"] = predicted_gains[:, i]
        predictions[f"baseline_{gain}"] = baseline_gains[:, i]

    MODEL_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "targets": TARGETS,
            "target_transform": "log10",
            "training_motor_ids": data["motor_id"].iloc[train_idx].tolist(),
            "test_motor_ids": data["motor_id"].iloc[test_idx].tolist(),
        },
        MODEL_PATH,
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    metrics = {
        "dataset": "dataset/pid_dataset_5s_v2.csv",
        "simulation_duration_seconds": 5.0,
        "total_labelled_motors": int(len(data)),
        "training_motors": int(len(train_idx)),
        "test_motors": int(len(test_idx)),
        "test_motor_ids": data["motor_id"].iloc[test_idx].tolist(),
        "model_log10_mae": float(overall_model_mae),
        "training_median_log10_mae": float(overall_baseline_mae),
        "per_gain": per_gain,
        "note": (
            "Exploratory metrics on held-out synthetic motors. "
            "Gain prediction accuracy does not guarantee valid closed-loop control."
        ),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("\n========== V2 TRAINING RESULTS ==========")
    print(f"Labelled motors: {len(data)}")
    print(f"Training motors: {len(train_idx)}")
    print(f"Test motors: {len(test_idx)}")
    print(f"Model overall log10 MAE: {overall_model_mae:.4f}")
    print(f"Median baseline log10 MAE: {overall_baseline_mae:.4f}")

    print("\nPer-gain errors (log10 scale):")
    for gain, values in per_gain.items():
        print(
            f"{gain}: model={values['model_log10_mae']:.4f}, "
            f"baseline={values['baseline_log10_mae']:.4f}"
        )

    print("\nTest motor IDs:", metrics["test_motor_ids"])
    print("\nSaved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved predictions:", PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
