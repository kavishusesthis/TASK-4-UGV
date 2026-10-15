# My Contribution: ML-Based PID Auto-Tuning

## Objective
Investigate whether a machine-learning model can predict useful PID gains
for simulated DC motors and improve them through local search.

## Implementation
- Used the existing five-second motor simulation and scoring pipeline.
- Used a trained Random Forest model to predict Kp, Ki, and Kd from motor
  parameters.
- Implemented ML-guided local search with 125 candidate evaluations.
- Compared against random search with the same 125-candidate budget.
- Evaluated both approaches on 25 saved held-out test motors.
- Expanded simulator tests; 22 tests passed.

## Main Results
- ML local search: 24/25 valid controllers (96%).
- Random search: 25/25 valid controllers (100%).
- On the 24 motors where both methods succeeded, ML local search achieved
  a lower score on 21 motors.
- Mean score on those 24 motors:
  - ML local search: 0.230289
  - Random search: 0.240365
- The ML local-search mean score was 4.19% lower on the shared successful
  motors.

## Failure Investigation
Motor 96 exposed a limitation: direct ML prediction and the local search
both exceeded the current constraint. Random search found a valid controller.

An additional diagnostic search around previously recorded valid gains found
90 valid candidates out of 125, with a best score of 0.73671 and peak current
of 10.601 A. Because these gains were based on known recorded data, this
result is diagnostic rather than independent evidence of generalization.

## Conclusion
ML-guided local search showed better scores on most shared successful test
motors, but its controller-validity rate was lower than random search.
A broader fallback search and further testing are needed before claiming
improved overall reliability.

## Reproducibility
Main evaluation script:
evaluate_fair_search_5s_v2.py

Detailed results:
results/fair_search_comparison_5s_v2.csv

Summary:
results/fair_search_summary_5s_v2.csv

All reported performance results are from simulation, not physical UGV tests.
