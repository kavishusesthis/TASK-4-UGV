from simulate_motor import Motor, simulate_motor

def main():
    motor = Motor(R=1.2, L=0.01, J=0.001, b=0.0002, Kt=0.05, Ke=0.05)
    for name, gains in [
        ("A", (0.1, 1.0, 0.001)),
        ("B", (0.3, 5.0, 0.003)),
        ("C", (0.6, 10.0, 0.005)),
    ]:
        result = simulate_motor(motor, gains)
        print(f"{name}: gains={gains}; " + ", ".join(
            f"{k}={v}" for k, v in result.metrics.items()
            if k in ("final_speed_rad_s", "overshoot_pct", "final_error_pct",
                     "settling_time_s", "peak_current_A", "score", "valid")))

if __name__ == "__main__":
    main()
