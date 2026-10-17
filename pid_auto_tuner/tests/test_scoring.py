from simulate_motor import Motor, simulate_motor

def main():
    motor = Motor(R=1.2, L=0.01, J=0.001, b=0.0002, Kt=0.05, Ke=0.05)
    result = simulate_motor(motor, (0.6, 10.0, 0.005))
    assert result.metrics["score"] >= 0
    assert result.metrics["valid"] in (True, False)
    assert result.metrics["peak_voltage_V"] <= 12.0 + 1e-9
    assert result.metrics["peak_current_A"] >= 0
    print("Scoring smoke tests passed.")
    print(result.metrics)

if __name__ == "__main__":
    main()
