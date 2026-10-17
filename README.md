Task 4- Sub-Team 1

Work Completed

1. Created a WhatsApp group for team communication and coordination.

2. Held a team discussion meeting at 9 PM to understand the task requirements

and discuss the overall approach.

3. Simulated different motors using Python.

4. Tested different combinations of PID parameters (Kp, Ki and Kd).

5. Evaluated the controller's performance and decided whether it met the required conditions.

6. Generated a dataset containing motor characteristics, tested PID combinations and performance results.

7. Trained a Machine Learning model to predict suitable PID parameters for a motor.

8. Evaluated the trained model on unseen motors.

9. Compared ML-based predictions and search methods with other PID tuning approaches.

10. Ran automated tests and performed additional analysis to check the implementation and evaluate the results.

Results 
Method	Valid rate on 25 held-out motors
Direct ML prediction	92% (23/25)
ML-assisted local search	96% (24/25)
Random search	100% (25/25)
Ziegler–Nichols	76% (19/25)
IMC PI	64% (16/25)

