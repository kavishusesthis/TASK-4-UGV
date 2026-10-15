# ML-Based PID Auto-Tuner for Simulated DC Motors

## 1. Project objective

This project trains a Machine Learning model to predict PID controller
gains (Kp, Ki, Kd) from simulated DC motor characteristics.

It compares:
- Direct ML-predicted PID gains
- A training-data median baseline
- ML predictions improved through local search

All results are simulation-based. The project has not been validated
on a physical motor or UGV.

## 2. Motor simulation

The simulator models simplified DC motor electrical and mechanical
dynamics.

Motor input features:
- R: armature resistance
- L: armature inductance
- J: moment of inertia
- b: viscous friction coefficient
- Kt: torque constant
- Ke: back-EMF constant

Model outputs:
- Kp: proportional gain
- Ki: integral gain
- Kd: derivative gain

The evaluation uses a target speed of 100 rad/s, a simulation duration
of 5 seconds, a 1 ms time step, a 12 V voltage limit and a 12 A current
limit.

These are simulation settings, not real hardware specifications.

## 3. Validity constraints

A controller is considered valid only if all the following conditions
are satisfied:

- Overshoot <= 20%
- Final absolute speed error <= 2% of target
- Peak absolute current <= 12 A
- Peak absolute voltage <= 12 V

A low score alone does not make a controller valid. Constraint
violations must also be checked.

## 4. Dataset

The latest experiment uses 100 synthetic motor configurations.

Files:
- dataset/experiment_log_5s_v2.csv
  Contains 26,000 candidate trials and their evaluation metrics.
- dataset/pid_dataset_5s_v2.csv
  Contains the best valid controller label for each eligible motor.

Dataset summary:
- Synthetic motors sampled: 100
- Candidate trials: 26,000
- Valid trials: 9,797
- Motors with valid controller labels: 99
- Motor ID 70 has no valid label in the sampled candidates.

A missing label is not replaced with an invented controller.

## 5. Machine Learning model

Model: Random Forest regression.

The model predicts Kp, Ki and Kd from the six motor features.
PID gains are modelled in log10 space and transformed back for
predictions.

Training and evaluation use a motor-level split:
- Labelled motors: 99
- Training motors: 74
- Held-out test motors: 25

Saved model:
models/pid_random_forest_5s_v2.joblib

Training metrics:
results/training_metrics_5s_v2.json

Overall held-out log10 MAE:
- Random Forest: 0.5288
- Training-median baseline: 0.6172

The model performs better overall on this metric. However, Kd prediction
error is worse than the median baseline, and gain prediction accuracy
does not guarantee a valid closed-loop controller.

## 6. Latest held-out controller evaluation

The latest fair-search experiment evaluated three methods on the same
25 held-out synthetic motors:

- Direct ML prediction: one controller predicted by the Random Forest.
- ML-assisted local search: 125 candidates around the predicted gains.
- Random search: 125 randomly sampled candidates.

Latest results:

| Method | Valid controllers | Valid rate |
|---|---:|---:|
| Direct ML prediction | 23/25 | 92% |
| ML-assisted local search | 24/25 | 96% |
| Random search | 25/25 | 100% |

Lower scores are better. Among the 24 motors where both ML-assisted
local search and random search found valid controllers:

- ML-assisted local search had the lower score on 21 motors.
- Random search had the lower score on 3 motors.
- Mean score was 0.2303 for ML-assisted local search and 0.2404
  for random search.
- ML-assisted local search's mean score was approximately 4.19% lower.

This comparison is limited to the 24 shared successful cases. It does
not mean ML-assisted local search was more reliable: random search
found valid controllers for all 25 motors, while ML-assisted local
search failed on one.

Files containing the latest fair-search results:

- `results/fair_search_summary_5s_v2.csv`
- `results/fair_search_comparison_5s_v2.csv`

Earlier evaluations are also retained in `results/` for reference.
The comparison is simulation-based and does not demonstrate performance
on physical hardware.

## 7. Known limitation: motor 96

The ML-assisted local search tested 125 candidates for motor 96 and
found no valid controller within its search grid.

The lowest-current candidates still reached approximately 12.34 A,
above the 12 A limit.

This does not prove that no valid PID gains exist. It only shows that
the tested local search did not find a valid solution for this motor.

Diagnostic script:
diagnose_motor96_local_search.py

Diagnostic results:
results/motor96_local_candidates_diagnostic.csv

## 8. Setup

Python 3.10 or newer is recommended.

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

## 9. Run and verify

Run the existing tests:

    python -m pytest -q

Train the v2 model using the existing labelled dataset:

    python train_model_5s_v2.py

Evaluate direct ML predictions:

    python evaluate_model_5s_v2.py

Evaluate ML-assisted local search:

    python evaluate_ml_local_search_5s_v2.py

Dataset generation is available through generate_dataset_5s_v2.py.
Run it only when you intentionally want to regenerate the v2 datasets,
because doing so can replace the current experiment files.

The scripts save their outputs under models/ and results/.

## 10. Limitations and future work

- All motor characteristics are synthetic.
- No physical motor or UGV hardware validation has been performed.
- The model was evaluated on 25 held-out motors from this synthetic setup.
- Some predicted or searched gains may violate the validity constraints.
- The local search explores a limited range around the model prediction.
- The saved best-controller reference is not a deployable baseline.
- Wider searches and additional tests may improve coverage, but no global
  optimum is guaranteed.

## 11. Handover status

The simulator, dataset generation, ML training, held-out evaluation,
and ML-assisted local search have been implemented.

The latest recorded automated test run passed all 22 tests.

The results are preliminary simulation findings. The team should review
the scripts, datasets, saved model and result files before integrating
or committing changes to the shared repository. No physical motor or
UGV validation has been demonstrated.
