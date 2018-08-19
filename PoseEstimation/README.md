# Common loss functions

1. View discretization: classification
2. Pose regression
    * regress 6d pose directly
        - angular distance for measuring the rotation error
        - L1 distance for the translation error measurement
    * regress RT matrix
        - geometric reprojection loss[1]
        - point matching loss[2]
    * regress SE(3) transformation
3. combine classification with regression
4. regress object coordinates

# reference

[1] Geometric Loss Functions for Camera Pose Regression with Deep Learning

[2] DeepIM: Deep Iterative Matching for 6D Pose Estimation

