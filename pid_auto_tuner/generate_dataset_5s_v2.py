"""Generate a larger five-second PID dataset without overwriting v1 files."""
from pathlib import Path

import numpy as np
import pandas as pd

from simulate_motor import Motor, simulate_motor

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"

MOTOR_FEATURES = ["R", "L", "J", "b", "Kt", "Ke"]
TARGETS = ["Kp", "Ki", "Kd"]


def sample_motor(rng):
    """Sample motor parameters using the original project's ranges."""
    R = float(rng.uniform(0.5, 4.0))
    L = float(np.exp(rng.uniform(np.log(0.001), np.log(0.020))))
    J = float(np.exp(rng.uniform(np.log(0.0001), np.log(0.005))))
    b = float(np.exp(rng.uniform(np.log(1e-5), np.log(1e-3))))
    Kt = float(rng.uniform(0.02, 0.10))

    return Motor(R=R, L=L, J=J, b=b, Kt=Kt, Ke=Kt)


def sample_gains(rng):
    """Sample gains logarithmically across a broad range."""
    return (
        float(np.exp(rng.uniform(np.log(0.001), np.log(2.0)))),
        float(np.exp(rng.uniform(np.log(0.001), np.log(20.0)))),
        float(np.exp(rng.uniform(np.log(1e-5), np.log(0.05)))),
    )


def candidate_gains(rng, random_trials):
    """Combine a complete coarse grid with reproducible random exploration."""
    kp_values = (0.003, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 1.8)
    ki_values = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0, 20.0)
    kd_values = (0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03)

    coarse = [
        (kp, ki, kd)
        for kp in kp_values
        for ki in ki_values
        for kd in kd_values
    ]

    # Keep a reproducible subset of the grid to bound runtime.
    rng.shuffle(coarse)
    selected_coarse = coarse[:60]

    random_candidates = [
        sample_gains(rng) for _ in range(random_trials)
    ]

    return selected_coarse + random_candidates


def generate(n_motors=100, random_trials=200, seed=2026,
             duration=5.0, dt=0.001):
    rng = np.random.default_rng(seed)
    logs = []
    best_rows = []

    for motor_id in range(n_motors):
        motor = sample_motor(rng)
        best = None

        candidates = candidate_gains(rng, random_trials)

        for trial_id, gains in enumerate(candidates):
            try:
                sim = simulate_motor(
                    motor,
                    gains,
                    duration=duration,
                    dt=dt,
                )
                metrics = sim.metrics
            except (ValueError, FloatingPointError, OverflowError) as exc:
                metrics = {
                    "valid": False,
                    "score": 1e9,
                    "error": str(exc),
                }

            row = {
                "motor_id": motor_id,
                **motor.__dict__,
                "trial_id": trial_id,
                "Kp": gains[0],
                "Ki": gains[1],
                "Kd": gains[2],
                "valid": bool(metrics.get("valid", False)),
                "score": float(metrics.get("score", 1e9)),
                **{
                    key: value
                    for key, value in metrics.items()
                    if key not in ("valid", "score")
                },
            }

            logs.append(row)

            if row["valid"] and (
                best is None or row["score"] < best["score"]
            ):
                best = row.copy()

        if best is not None:
            best_rows.append({
                key: best[key]
                for key in [
                    "motor_id",
                    *MOTOR_FEATURES,
                    *TARGETS,
                    "valid",
                    "score",
                    "final_speed_rad_s",
                    "overshoot_pct",
                    "final_error_pct",
                    "settling_time_s",
                    "peak_current_A",
                    "peak_voltage_V",
                    "iae",
                ]
            })

        status = (
            f"best valid score {best['score']:.4f}"
            if best is not None
            else "NO valid controller found"
        )
        print(f"Motor {motor_id + 1}/{n_motors}: {status}")

    return pd.DataFrame(logs), pd.DataFrame(best_rows)


def main():
    DATA_DIR.mkdir(exist_ok=True)

    log, best = generate(
        n_motors=100,
        random_trials=200,
        seed=2026,
        duration=5.0,
        dt=0.001,
    )

    log_path = DATA_DIR / "experiment_log_5s_v2.csv"
    best_path = DATA_DIR / "pid_dataset_5s_v2.csv"

    log.to_csv(log_path, index=False)
    best.to_csv(best_path, index=False)

    valid_trials = int(log["valid"].sum())

    print("\n========== V2 DATASET SUMMARY ==========")
    print(f"Motors: 100")
    print(f"Trials: {len(log)}")
    print(f"Valid trials: {valid_trials}")
    print(f"Invalid trials: {len(log) - valid_trials}")
    print(f"Motors with valid labels: {len(best)}/100")
    print(f"Valid trial rate: {100 * valid_trials / len(log):.2f}%")
    print(f"Experiment log: {log_path}")
    print(f"Best-controller dataset: {best_path}")


if __name__ == "__main__":
    main()
