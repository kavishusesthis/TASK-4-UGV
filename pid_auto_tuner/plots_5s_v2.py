from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simulate_motor import Motor, simulate_motor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "plots"; OUT.mkdir(parents=True, exist_ok=True)
R = ROOT / "results"

# 1. predicted vs actual gains
p = pd.read_csv(R / "gain_predictions_5s_v2.csv")
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, g in zip(ax, ["Kp", "Ki", "Kd"]):
    x, y = p[f"actual_{g}"], p[f"predicted_{g}"]
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    a.loglog(x, y, "o"); a.loglog([lo, hi], [lo, hi], "--")
    a.set_title(g); a.set_xlabel("actual"); a.set_ylabel("predicted")
fig.tight_layout(); fig.savefig(OUT / "gain_pred_vs_actual.png", dpi=150)

# 2. valid rate and score per method
s = pd.read_csv(R / "fair_search_summary_5s_v2.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(s["method"], s["valid_rate_pct"]); ax[0].set_ylabel("valid rate (%)")
ax[1].bar(s["method"], s["mean_score_valid_only"]); ax[1].set_ylabel("mean score (valid only, lower better)")
for a in ax: a.tick_params(axis="x", rotation=20)
fig.tight_layout(); fig.savefig(OUT / "method_comparison.png", dpi=150)

# 3. step responses for chosen motors
data = pd.read_csv(ROOT / "dataset" / "pid_dataset_5s_v2.csv").set_index("motor_id")
c = pd.read_csv(R / "fair_search_comparison_5s_v2.csv")
for mid in (96, 62):
    r = data.loc[mid]; motor = Motor(r.R, r.L, r.J, r.b, r.Kt, r.Ke)
    plt.figure(figsize=(7, 4))
    for _, row in c[c.motor_id == mid].iterrows():
        if pd.isna(row.Kp): continue
        sim = simulate_motor(motor, (row.Kp, row.Ki, row.Kd), duration=5.0)
        plt.plot(sim.time, sim.speed, label=f"{row.method} (valid={row.valid})")
    plt.axhline(100, color="k", ls=":"); plt.xlabel("time (s)"); plt.ylabel("speed (rad/s)")
    plt.title(f"Motor {mid}"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / f"step_motor_{mid}.png", dpi=150)
print("saved to", OUT)