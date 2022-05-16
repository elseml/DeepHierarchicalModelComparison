# Deep hierarchical model comparison project

## notebooks

Contains all analyses reported in the paper.

### 01_calibration_validation

First validation study that contains three sub-parts:
- calibration_fixed_sizes: Training and calibration assessment with data sets that all possess the same amount of groups and nested observations.
- calibration_variable_observations: Training and calibration assessment with data sets that all possess the same amount of groups but vary in their amount of nested observations.
- calibration_variable_sizes: Training and calibration assessment with data sets that vary in their amount of groups as well as nested observations.

The trained models can be accessed in the respective 'checkpoints' folder.

### 02_bridge_sampling_comparison

Second validation study in which the approximation performance of the networks from calibration_fixed_sizes and calibration_variable_sizes is benchmarked against bridge sampling.

### 03_levy_flight_application

Application study in which two formulations of the standard diffusion model are compared to a lévy flight model with a non-Gaussian noise distribution. Consists of five steps:
1. Simulate training and validation data
2. Pretrain the network on simulated data with a reduced amount of trials per participant
3. Fine-tune the network on simulated data that contains the same amount of trials per participant as the empirical data
4. Validate the trained networks on new simulated data sets 
5. Apply the trained networks to the empirical data set

## src

Contains custom Julia and Python functions that enable the analyses.
