"""Train Random Forest regression on best valid controllers, holding out whole motors."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "pid_dataset.csv"
MODEL_DIR = ROOT / "models"; MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR = ROOT / "results"; RESULTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "pid_random_forest.joblib"
FEATURES = ["R", "L", "J", "b", "Kt", "Ke"]
TARGETS = ["Kp", "Ki", "Kd"]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run python generate_dataset.py first.")
    df = pd.read_csv(DATA_PATH)
    if "valid" in df:
        df = df[df["valid"].astype(str).str.lower().isin(["true", "1"])].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + TARGETS)
    for col in TARGETS:
        df = df[df[col] > 0]
    if len(df) < 8:
        raise ValueError(f"Only {len(df)} valid motor examples. Generate more motors before training.")
    # One row per motor is the dataset contract; splitting rows keeps each motor isolated.
    train_idx, test_idx = train_test_split(df.index, test_size=0.25, random_state=42)
    X_train, X_test = df.loc[train_idx, FEATURES], df.loc[test_idx, FEATURES]
    y_train, y_test = np.log10(df.loc[train_idx, TARGETS]), np.log10(df.loc[test_idx, TARGETS])
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred_log = model.predict(X_test)
    median_log = np.tile(np.median(y_train.to_numpy(), axis=0), (len(y_test), 1))
    model_mae = float(mean_absolute_error(y_test.to_numpy(), pred_log))
    median_mae = float(mean_absolute_error(y_test.to_numpy(), median_log))
    metrics = {"training_motors": int(len(train_idx)), "test_motors": int(len(test_idx)),
               "model_log10_mae": model_mae, "training_median_log10_mae": median_mae,
               "note": "Small synthetic dataset; metrics are exploratory, not evidence of hardware performance."}
    print(json.dumps(metrics, indent=2))
    joblib.dump({"model": model, "features": FEATURES, "targets": TARGETS,
                 "target_transform": "log10", "random_state": 42}, MODEL_PATH)
    (RESULTS_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    pred_rows = []
    true = 10 ** y_test.to_numpy(); pred = 10 ** pred_log; baseline = 10 ** median_log
    for j, name in enumerate(TARGETS):
        print(f"{name}: model MAE={mean_absolute_error(true[:, j], pred[:, j]):.6g}; "
              f"median baseline MAE={mean_absolute_error(true[:, j], baseline[:, j]):.6g}")
    for idx, actual, estimate in zip(test_idx, true, pred):
        pred_rows.append({"dataset_index": int(idx), **{f"actual_{n}": float(v) for n,v in zip(TARGETS, actual)},
                          **{f"predicted_{n}": float(v) for n,v in zip(TARGETS, estimate)}})
    pd.DataFrame(pred_rows).to_csv(RESULTS_DIR / "gain_predictions.csv", index=False)
    print(f"Saved model: {MODEL_PATH}")
if __name__ == "__main__":
    main()
