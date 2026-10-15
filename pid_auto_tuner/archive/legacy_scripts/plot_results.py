"""Create project plots from training/evaluation CSV outputs."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
from simulate_motor import Motor, simulate_motor
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "dataset"

def main():
    RESULTS.mkdir(exist_ok=True)
    pred_path = RESULTS / "gain_predictions.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        for gain in ("Kp", "Ki", "Kd"):
            plt.figure()
            plt.scatter(pred[f"actual_{gain}"], pred[f"predicted_{gain}"])
            lo = min(pred[f"actual_{gain}"].min(), pred[f"predicted_{gain}"].min())
            hi = max(pred[f"actual_{gain}"].max(), pred[f"predicted_{gain}"].max())
            plt.plot([lo, hi], [lo, hi])
            plt.xlabel(f"Actual {gain}")
            plt.ylabel(f"Predicted {gain}")
            plt.title(f"{gain}: actual vs predicted on held-out motors")
            plt.tight_layout()
            plt.savefig(RESULTS / f"gain_{gain}_actual_vs_predicted.png", dpi=150)
            plt.close()

    summary_path = RESULTS / "controller_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        plt.figure()
        plt.bar(summary["method"], summary["mean_score"])
        plt.ylabel("Mean score (lower is better)")
        plt.title("Closed-loop controller score comparison")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(RESULTS / "controller_score_comparison.png", dpi=150)
        plt.close()

        plt.figure()
        plt.bar(summary["method"], summary["valid_rate"] * 100)
        plt.ylabel("Valid controllers (%)")
        plt.ylim(0, 100)
        plt.title("Controller validity rate")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(RESULTS / "controller_validity_comparison.png", dpi=150)
        plt.close()

    comparison_path = RESULTS / "controller_comparison.csv"
    dataset_path = DATA / "pid_dataset.csv"
    if comparison_path.exists() and dataset_path.exists():
        comparison = pd.read_csv(comparison_path)
        dataset = pd.read_csv(dataset_path)
        if not comparison.empty and not dataset.empty:
            motor_id = int(comparison.iloc[0]["motor_id"])
            motor_row = dataset[dataset["motor_id"] == motor_id]
            if not motor_row.empty:
                row = motor_row.iloc[0]
                motor = Motor(*(float(row[k]) for k in ["R", "L", "J", "b", "Kt", "Ke"]))
                plt.figure()
                for method in comparison[comparison["motor_id"] == motor_id]["method"].unique():
                    r = comparison[(comparison["motor_id"] == motor_id) &
                                   (comparison["method"] == method)].iloc[0]
                    gains = (float(r["Kp"]), float(r["Ki"]), float(r["Kd"]))
                    sim = simulate_motor(motor, gains)
                    plt.plot(sim.time, sim.speed, label=method)
                plt.axhline(100.0, linestyle="--", label="Target (100 rad/s)")
                plt.xlabel("Time (s)")
                plt.ylabel("Speed (rad/s)")
                plt.title(f"Speed responses for held-out motor {motor_id}")
                plt.legend(fontsize="small")
                plt.tight_layout()
                plt.savefig(RESULTS / "held_out_motor_speed_responses.png", dpi=150)
                plt.close()
    print(f"Plots saved under {RESULTS}")

if __name__ == "__main__":
    main()
