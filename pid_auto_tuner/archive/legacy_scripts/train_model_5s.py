"""Train a separate Random Forest model using five-second PID labels."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "pid_dataset_5s.csv"
MODEL_PATH = ROOT / "models" / "pid_random_forest_5s.joblib"
RESULTS_DIR = ROOT / "results"

FEATURES = ["R", "L", "J", "b", "Kt", "Ke"]
TARGETS = ["Kp", "Ki", "Kd"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + TARGETS)

    if "valid" in df.columns:
        df = df[df["valid"].astype(str).str.lower().isin(["true", "1"])]

    for target in TARGETS:
        df = df[df[target] > 0]

    if len(df) < 8:
        raise ValueError(
            f"Only {len(df)} valid motor examples. At least 8 are required."
        )

    if df["motor_id"].duplicated().any():
        raise ValueError("Expected one labelled row per motor.")

    # Split whole motors, not individual experiment rows.
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.25, random_state=42
    )

    X_train = df.loc[train_idx, FEATURES]
    X_test = df.loc[test_idx, FEATURES]

    y_train = np.log10(df.loc[train_idx, TARGETS])
    y_test = np.log10(df.loc[test_idx, TARGETS])

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predicted_log = model.predict(X_test)
    median_log = np.tile(
        np.median(y_train.to_numpy(), axis=0),
        (len(y_test), 1),
    )

    model_log_mae = float(
        mean_absolute_error(y_test.to_numpy(), predicted_log)
    )
    baseline_log_mae = float(
        mean_absolute_error(y_test.to_numpy(), median_log)
    )

    actual = 10 ** y_test.to_numpy()
    predicted = 10 ** predicted_log
    baseline = 10 ** median_log

    metrics = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "simulation_duration_seconds": 5.0,
        "total_motors": int(len(df)),
        "training_motors": int(len(train_idx)),
        "test_motors": int(len(test_idx)),
        "model_log10_mae": model_log_mae,
        "training_median_log10_mae": baseline_log_mae,
        "per_gain_mae": {},
        "note": (
            "Exploratory metrics on five held-out synthetic motors; "
            "not evidence of hardware performance."
        ),
    }

    for j, name in enumerate(TARGETS):
        model_mae = float(mean_absolute_error(actual[:, j], predicted[:, j]))
        baseline_mae = float(mean_absolute_error(actual[:, j], baseline[:, j]))

        metrics["per_gain_mae"][name] = {
            "model": model_mae,
            "median_baseline": baseline_mae,
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "targets": TARGETS,
            "target_transform": "log10",
            "random_state": 42,
            "simulation_duration_seconds": 5.0,
        },
        MODEL_PATH,
    )

    (RESULTS_DIR / "training_metrics_5s.json").write_text(
        json.dumps(metrics, indent=2)
    )

    prediction_rows = []
    for idx, actual_gains, predicted_gains, baseline_gains in zip(
        test_idx, actual, predicted, baseline
    ):
        row = {
            "motor_id": int(df.loc[idx, "motor_id"]),
        }

        for j, name in enumerate(TARGETS):
            row[f"actual_{name}"] = float(actual_gains[j])
            row[f"predicted_{name}"] = float(predicted_gains[j])
            row[f"baseline_{name}"] = float(baseline_gains[j])

        prediction_rows.append(row)

    pd.DataFrame(prediction_rows).to_csv(
        RESULTS_DIR / "gain_predictions_5s.csv", index=False
    )

    print("========== FIVE-SECOND MODEL TRAINING ==========")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {RESULTS_DIR / 'training_metrics_5s.json'}")
    print(f"Saved predictions: {RESULTS_DIR / 'gain_predictions_5s.csv'}")
    print("\nOriginal model and results were not overwritten.")


if __name__ == "__main__":
    main()
