"""Classical tuning baselines on the same 25 held-out motors (5 s sim)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from simulate_motor import Motor, simulate_motor

ROOT = Path(__file__).resolve().parent
ids = json.loads((ROOT / "results" / "training_metrics_5s_v2.json").read_text())["test_motor_ids"]
data = pd.read_csv(ROOT / "dataset" / "pid_dataset_5s_v2.csv").set_index("motor_id")
LO = np.array([0.001, 0.001, 1e-5]); HI = np.array([2.0, 20.0, 0.05])  # same range as the search

def zn_reaction_curve(m, duration=5.0, dt=0.001, V=12.0):
    n = int(round(duration / dt)) + 1
    t = np.linspace(0, duration, n); i = np.zeros(n); w = np.zeros(n)
    decay = np.exp(-m.R * dt / m.L)
    for k in range(n - 1):
        eq = (V - m.Ke * w[k]) / m.R
        i[k + 1] = eq + (i[k] - eq) * decay
        w[k + 1] = w[k] + dt * (m.Kt * i[k + 1] - m.b * w[k]) / m.J
    slope = np.gradient(w, dt); idx = int(np.argmax(slope))
    ms, final = float(slope[idx]), float(w[-1])
    if ms <= 1e-9 or final <= 1e-9:
        return np.array([0.1, 0.1, 0.001])
    K = final / V
    t0 = t[idx] - w[idx] / ms
    L = max(dt, t0); T = max(dt, t[idx] + (final - w[idx]) / ms - t0)
    kp = 1.2 * T / (K * L); ti = 2.0 * L; td = 0.5 * L
    return np.clip([kp, kp / ti, kp * td], LO, HI)

def imc_pi(m, lam_factor=0.3):
    """IMC/lambda tuning on the first-order model (inductance neglected)."""
    D = m.R * m.b + m.Kt * m.Ke
    K, tau = m.Kt / D, m.J * m.R / D
    lam = lam_factor * tau
    kp = tau / (K * lam)
    return np.clip([kp, kp / tau, 0.0], [LO[0], LO[1], 0.0], HI)

rows = []
for mid in ids:
    r = data.loc[mid]
    motor = Motor(r.R, r.L, r.J, r.b, r.Kt, r.Ke)
    for name, g in [("Ziegler-Nichols", zn_reaction_curve(motor)), ("IMC PI", imc_pi(motor))]:
        mt = simulate_motor(motor, g, duration=5.0).metrics
        rows.append({"motor_id": mid, "method": name, "Kp": g[0], "Ki": g[1], "Kd": g[2],
                     "valid": mt["valid"], "score": mt["score"],
                     "overshoot_pct": mt["overshoot_pct"], "peak_current_A": mt["peak_current_A"]})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "classical_comparison_5s_v2.csv", index=False)
summary = df.groupby("method").agg(motors=("valid", "size"), valid=("valid", "sum"),
                                   mean_score_all=("score", "mean"))
summary["valid_rate_pct"] = 100 * summary["valid"] / summary["motors"]
summary["mean_score_valid_only"] = df[df.valid].groupby("method")["score"].mean()
print(summary)
summary.to_csv(ROOT / "results" / "classical_summary_5s_v2.csv")