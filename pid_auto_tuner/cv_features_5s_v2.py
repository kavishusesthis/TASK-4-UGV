from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT / "dataset" / "pid_dataset_5s_v2.csv")
d = d[d["valid"].astype(str).str.lower() == "true"].reset_index(drop=True)
BASE = ["R", "L", "J", "b", "Kt", "Ke"]
TARGETS = ["Kp", "Ki", "Kd"]
y = np.log10(d[TARGETS].to_numpy())


def features(df, engineered):
    X = df[BASE].copy()
    if engineered:
        D = df.R * df.b + df.Kt * df.Ke
        X["tau_e"] = df.L / df.R          # electrical time constant
        X["tau_m"] = df.J * df.R / D      # mechanical time constant
        X["dc_gain"] = df.Kt / D          # speed per volt
    return X


for engineered in (False, True):
    X = features(d, engineered)
    res = {"model": [], "median": []}
    for seed in range(5):  # 5 repeats of 5-fold CV
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
            m = RandomForestRegressor(300, min_samples_leaf=2, random_state=seed, n_jobs=-1)
            m.fit(X.iloc[tr], y[tr])
            pred = m.predict(X.iloc[te])
            base = np.median(y[tr], axis=0)
            res["model"].append([mean_absolute_error(y[te][:, i], pred[:, i]) for i in range(3)])
            res["median"].append([mean_absolute_error(y[te][:, i], np.full(len(te), base[i])) for i in range(3)])
    print(f"\nengineered features = {engineered}")
    for k, v in res.items():
        v = np.array(v)
        print(k, " ".join(f"{g}: {v[:, i].mean():.3f}+-{v[:, i].std():.3f}" for i, g in enumerate(TARGETS)))