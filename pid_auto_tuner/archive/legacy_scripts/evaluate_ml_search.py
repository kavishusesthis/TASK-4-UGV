"""Compare direct ML prediction, random search, and ML-assisted local search."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from simulate_motor import Motor, simulate_motor
from generate_dataset import sample_gains

ROOT = Path(__file__).resolve().parent
FEATURES = ["R", "L", "J", "b", "Kt", "Ke"]
TARGETS = ["Kp", "Ki", "Kd"]

# Same gain ranges used by generate_dataset.py.
GAIN_LIMITS = np.array([
    [0.001, 2.0],
    [0.001, 20.0],
    [0.00001, 0.05],
], dtype=float)


def evaluate(motor, gains, duration=1.0, dt=0.001):
    return simulate_motor(
        motor, gains, duration=duration, dt=dt
    ).metrics


def random_search(motor, rng, budget=100):
    best_gains, best_metrics = None, None

    for _ in range(budget):
        gains = sample_gains(rng)
        metrics = evaluate(motor, gains)

        if metrics["valid"] and (
            best_metrics is None
            or metrics["score"] < best_metrics["score"]
        ):
            best_gains, best_metrics = tuple(gains), metrics

    return best_gains, best_metrics


def ml_local_search(motor, initial_gains, rng, budget=100):
    """Search around ML gains in log space, retaining the best valid result."""
    initial = np.clip(
        np.asarray(initial_gains, dtype=float),
        GAIN_LIMITS[:, 0],
        GAIN_LIMITS[:, 1],
    )

    initial_metrics = evaluate(motor, initial)
    best_gains = tuple(initial)
    best_metrics = initial_metrics if initial_metrics["valid"] else None

    log_initial = np.log(initial)
    log_limits = np.log(GAIN_LIMITS)

    for trial in range(budget):
        # Gradually broaden the search around the ML prediction.
        # Each candidate is clipped to the generator's supported ranges.
        spread = 0.25 + 1.25 * (trial / max(budget - 1, 1))
        candidate_log = log_initial + rng.normal(
            loc=0.0, scale=spread, size=3
        )
        candidate = np.exp(
            np.clip(candidate_log, log_limits[:, 0], log_limits[:, 1])
        )

        metrics = evaluate(motor, candidate)

        if metrics["valid"] and (
            best_metrics is None
            or metrics["score"] < best_metrics["score"]
        ):
            best_gains = tuple(candidate)
            best_metrics = metrics

    # If neither the initial ML gains nor local candidates were valid,
    # report the original ML controller's metrics as the fallback.
    if best_metrics is None:
        best_metrics = initial_metrics

    return best_gains, best_metrics


def main():
    data = pd.read_csv(ROOT / "dataset" / "pid_dataset.csv")
    bundle = joblib.load(ROOT / "models" / "pid_random_forest.joblib")
    model = bundle["model"]

    usable = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURES + TARGETS
    )
    usable = usable[(usable[TARGETS] > 0).all(axis=1)]

    train_idx, test_idx = train_test_split(
        usable.index, test_size=0.25, random_state=42
    )
    test = usable.loc[test_idx]

    rng = np.random.default_rng(20261014)
    rows = []
    budget = 100

    for _, row in test.iterrows():
        motor = Motor(*(float(row[name]) for name in FEATURES))

        predicted = 10 ** model.predict(
            pd.DataFrame([row[FEATURES]], columns=FEATURES)
        )[0]

        random_gains, _ = random_search(motor, rng, budget)
        local_gains, _ = ml_local_search(
            motor, predicted, rng, budget
        )

        methods = {
            "ML prediction": tuple(predicted),
            "Random search": random_gains,
            "ML + local search": local_gains,
        }

        for method, gains in methods.items():
            if gains is None:
                # No valid random-search candidate was found.
                # Keep an explicit failed result rather than hiding it.
                metrics = evaluate(motor, predicted)
                metrics = dict(metrics)
                metrics["valid"] = False
            else:
                metrics = evaluate(motor, gains)

            rows.append({
                "motor_id": int(row.get("motor_id", row.name)),
                "method": method,
                **dict(zip(TARGETS, map(float, gains or predicted))),
                **metrics,
            })

        print(f"Evaluated held-out motor {int(row.get('motor_id', row.name))}")

    out = pd.DataFrame(rows)
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    detail_path = results_dir / "ml_search_comparison.csv"
    summary_path = results_dir / "ml_search_summary.csv"
    out.to_csv(detail_path, index=False)

    summary = out.groupby("method").agg(
        motors=("motor_id", "count"),
        valid_rate=("valid", "mean"),
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        mean_overshoot_pct=("overshoot_pct", "mean"),
        mean_final_error_pct=("final_error_pct", "mean"),
        mean_settling_time_s=("settling_time_s", "mean"),
    ).reset_index()

    summary.to_csv(summary_path, index=False)

    print("\nComparison summary:")
    print(summary.to_string(index=False))
    print(f"\nDetailed results: {detail_path}")
    print(f"Summary: {summary_path}")
    print(
        "\nCaution: only a few held-out synthetic motors are available. "
        "Treat this as an experiment, not proof of general performance."
    )


if __name__ == "__main__":
    main()
