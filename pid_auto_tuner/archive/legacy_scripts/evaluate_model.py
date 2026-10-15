"""Closed-loop comparison on held-out motors: ML, training median, random search, Z-N and oracle."""
from pathlib import Path
import argparse, json, time
import joblib
import numpy as np
import pandas as pd
from simulate_motor import Motor, simulate_motor
from generate_dataset import sample_gains
from scoring import score_response

ROOT = Path(__file__).resolve().parent
FEATURES = ["R", "L", "J", "b", "Kt", "Ke"]
TARGETS = ["Kp", "Ki", "Kd"]

def random_search(motor, rng, trials=100, duration=1.0, dt=0.001):
    best_gains, best_metrics = None, None
    for gains in (sample_gains(rng) for _ in range(trials)):
        metrics = simulate_motor(motor, gains, duration=duration, dt=dt).metrics
        if metrics["valid"] and (best_metrics is None or metrics["score"] < best_metrics["score"]):
            best_gains, best_metrics = gains, metrics
    return best_gains, best_metrics

def zn_reaction_curve(motor, target_speed=100.0, duration=1.0, dt=0.001):
    """Approximate open-loop Ziegler-Nichols reaction-curve tuning.

    Apply a 12 V step, estimate the tangent at maximum slope, then use classic
    PID reaction-curve rules. This is a comparison baseline, not guaranteed safe.
    """
    n = int(round(duration/dt)) + 1
    t = np.linspace(0, duration, n)
    i = np.zeros(n); w = np.zeros(n); V = 12.0
    for k in range(n-1):
        decay = np.exp(-motor.R*dt/motor.L)
        eq = (V - motor.Ke*w[k])/motor.R
        i[k+1] = eq + (i[k]-eq)*decay
        w[k+1] = w[k] + dt*(motor.Kt*i[k+1]-motor.b*w[k])/motor.J
    slope = np.gradient(w, dt)
    idx = int(np.argmax(slope))
    max_slope = float(slope[idx])
    final = float(w[-1])
    if max_slope <= 1e-9 or final <= 1e-9:
        return (0.1, 0.1, 0.001)
    process_gain = final / V
    t_intercept = t[idx] - w[idx]/max_slope
    t_end = t[idx] + (final-w[idx])/max_slope
    dead_time = max(dt, t_intercept)
    time_constant = max(dt, t_end - t_intercept)
    # Classic Z-N open-loop reaction-curve PID settings.
    kp = max(1e-5, 1.2*time_constant/(process_gain*dead_time))
    ti = max(dt, 2.0*dead_time)
    td = max(0.0, 0.5*dead_time)
    ki = kp/ti
    kd = kp*td
    return (float(np.clip(kp, 1e-5, 2.0)), float(np.clip(ki, 1e-5, 20.0)),
            float(np.clip(kd, 1e-7, 0.05)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--search-trials", type=int, default=100)
    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.001)
    args = p.parse_args()
    data = pd.read_csv(ROOT / "dataset" / "pid_dataset.csv")
    bundle = joblib.load(ROOT / "models" / "pid_random_forest.joblib")
    model, features, targets = bundle["model"], bundle["features"], bundle["targets"]
    if len(data) < 8:
        raise ValueError("Need at least 8 labelled motors to evaluate.")
    # Recreate the same deterministic split used by train_model.py.
    from sklearn.model_selection import train_test_split
    usable = data.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES+TARGETS)
    usable = usable[(usable[TARGETS] > 0).all(axis=1)]
    train_idx, test_idx = train_test_split(usable.index, test_size=0.25, random_state=42)
    train = usable.loc[train_idx]; test = usable.loc[test_idx]
    median_gains = np.median(train[TARGETS].to_numpy(), axis=0)
    rng = np.random.default_rng(2026)
    rows = []
    for _, row in test.iterrows():
        motor = Motor(*(float(row[k]) for k in FEATURES))
        predicted = 10 ** model.predict(pd.DataFrame([row[FEATURES]], columns=FEATURES))[0]
        zn_gains = zn_reaction_curve(motor, duration=args.duration, dt=args.dt)
        methods = {
            "ML prediction": tuple(predicted),
            "Training median": tuple(median_gains),
            "Ziegler-Nichols reaction curve": zn_gains,
        }
        # Random search is a deployable baseline with an explicit fixed tuning budget.
        rs_gains, rs_metrics = random_search(motor, rng, args.search_trials, args.duration, args.dt)
        if rs_gains is not None:
            methods["Random search"] = rs_gains
        # Oracle is explicitly labelled as an upper-bound reference, not a fair baseline.
        methods["Test-motor oracle (reference only)"] = tuple(row[TARGETS])
        for method, gains in methods.items():
            result = simulate_motor(motor, gains, duration=args.duration, dt=args.dt)
            m = result.metrics
            rows.append({"motor_id": int(row.get("motor_id", row.name)), "method": method,
                         **dict(zip(TARGETS, map(float, gains))),
                         **{k: v for k,v in m.items() if isinstance(v, (int,float,bool))}})
    out = pd.DataFrame(rows)
    ROOT.joinpath("results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "controller_comparison.csv", index=False)
    summary = out.groupby("method").agg(mean_score=("score","mean"),
        median_score=("score","median"), valid_rate=("valid","mean"),
        mean_overshoot_pct=("overshoot_pct","mean"),
        mean_final_error_pct=("final_error_pct","mean"),
        mean_settling_time_s=("settling_time_s","mean"),
        mean_peak_current_A=("peak_current_A","mean")).reset_index()
    summary.to_csv(ROOT / "results" / "controller_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nDetailed results: results/controller_comparison.csv")
    print("Summary: results/controller_summary.csv")
    print("Note: test-motor oracle uses each test motor's stored best gains and is not deployable.")
if __name__ == "__main__":
    main()
