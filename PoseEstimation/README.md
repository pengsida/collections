# Common approaches

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
    * use an encoder-decoder network to regress object coordinates
    * huber loss
5. fitting
    * fitting with sparse control points[3]

# reference

[1] Geometric Loss Functions for Camera Pose Regression with Deep Learning

[2] DeepIM: Deep Iterative Matching for 6D Pose Estimation

[3] Real-Time Seamless Single Shot 6D Object Pose Prediction
