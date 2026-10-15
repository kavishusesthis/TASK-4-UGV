"""Generate synthetic motors and search PID gains; outputs CSV experiment log and labels."""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from simulate_motor import Motor, simulate_motor

ROOT = Path(__file__).resolve().parent
MOTOR_FEATURES = ["R", "L", "J", "b", "Kt", "Ke"]
TARGETS = ["Kp", "Ki", "Kd"]

def sample_motor(rng):
    R = float(rng.uniform(0.5, 4.0))
    L = float(np.exp(rng.uniform(np.log(0.001), np.log(0.020))))
    J = float(np.exp(rng.uniform(np.log(0.0001), np.log(0.005))))
    b = float(np.exp(rng.uniform(np.log(1e-5), np.log(1e-3))))
    Kt = float(rng.uniform(0.02, 0.10))
    return Motor(R, L, J, b, Kt, Kt)

def sample_gains(rng):
    return (float(np.exp(rng.uniform(np.log(0.001), np.log(2.0)))),
            float(np.exp(rng.uniform(np.log(0.001), np.log(20.0)))),
            float(np.exp(rng.uniform(np.log(1e-5), np.log(0.05)))))

def candidate_gains(motor, rng, random_trials):
    # Broad reproducible log-random exploration plus a small coarse grid.
    coarse = []
    for kp in (0.03, 0.1, 0.3, 0.6, 1.0):
        for ki in (0.1, 1.0, 5.0, 10.0):
            for kd in (0.0001, 0.001, 0.005):
                coarse.append((kp, ki, kd))
    rng.shuffle(coarse)
    candidates = coarse[:min(25, len(coarse))]
    candidates.extend(sample_gains(rng) for _ in range(random_trials))
    return candidates

def generate(n_motors=20, random_trials=100, seed=42, duration=1.0, dt=0.001):
    rng = np.random.default_rng(seed)
    logs, best_rows = [], []
    for motor_id in range(n_motors):
        motor = sample_motor(rng)
        best = None
        for trial_id, gains in enumerate(candidate_gains(motor, rng, random_trials)):
            try:
                sim = simulate_motor(motor, gains, duration=duration, dt=dt)
                m = sim.metrics
            except (ValueError, FloatingPointError, OverflowError) as exc:
                m = {"valid": False, "score": 1e9, "error": str(exc)}
            row = {"motor_id": motor_id, **motor.__dict__, "trial_id": trial_id,
                   "Kp": gains[0], "Ki": gains[1], "Kd": gains[2],
                   "valid": bool(m.get("valid", False)), "score": float(m.get("score", 1e9)),
                   **{k: v for k, v in m.items() if k not in ("valid", "score")}}
            logs.append(row)
            if row["valid"] and (best is None or row["score"] < best["score"]):
                best = row.copy()
        if best is not None:
            best_rows.append({k: best[k] for k in ["motor_id"] + MOTOR_FEATURES + TARGETS +
                              ["valid", "score", "final_speed_rad_s", "overshoot_pct",
                               "final_error_pct", "settling_time_s", "peak_current_A",
                               "peak_voltage_V", "iae"]})
        print(f"Motor {motor_id+1}/{n_motors}: " +
              ("best valid score %.4f" % best["score"] if best else "NO valid controller found"))
    return pd.DataFrame(logs), pd.DataFrame(best_rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motors", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100, help="random candidates per motor, in addition to coarse candidates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args()
    out = ROOT / "dataset"; out.mkdir(exist_ok=True)
    log, best = generate(args.motors, args.trials, args.seed, args.duration, args.dt)
    log_path, best_path = out / "experiment_log.csv", out / "pid_dataset.csv"
    log.to_csv(log_path, index=False); best.to_csv(best_path, index=False)
    valid_trials = int(log["valid"].sum()) if len(log) else 0
    print(f"\nTrials: {len(log)} | valid trials: {valid_trials} | invalid trials: {len(log)-valid_trials}")
    print(f"Motors with valid labels: {len(best)}/{args.motors}")
    print(f"Saved: {log_path}\nSaved: {best_path}")
    if len(best) < 8:
        print("WARNING: fewer than 8 labelled motors; this is a pipeline smoke test, not enough for reliable ML.")
if __name__ == "__main__":
    main()
