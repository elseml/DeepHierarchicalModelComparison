# Deep hierarchical model comparison project

## [notebooks](notebooks)

Contains all analyses reported in the paper.

### [01_calibration_validation](notebooks/01_calibration_validation)

First validation study that contains three sub-parts:
- [calibration_fixed_sizes](notebooks/01_calibration_validation/calibration_fixed_sizes.ipynb): Training and calibration assessment with data sets that all possess the same amount of groups and nested observations.
- [calibration_variable_observations](notebooks/01_calibration_validation/calibration_variable_observations.ipynb): Training and calibration assessment with data sets that all possess the same amount of groups but vary in their amount of nested observations.
- [calibration_variable_sizes](notebooks/01_calibration_validation/calibration_variable_sizes.ipynb): Training and calibration assessment with data sets that vary in their amount of groups as well as nested observations.

The trained models can be accessed in the respective 'checkpoints' folder.

### [02_bridge_sampling_comparison](notebooks/02_bridge_sampling_comparison)

Second validation study in which the approximation performance of the networks from calibration_fixed_sizes and calibration_variable_sizes is benchmarked against bridge sampling.

### [03_levy_flight_application](notebooks/03_levy_flight_application)

Application study in which two formulations of the standard diffusion model are compared to a lévy flight model with a non-Gaussian noise distribution. Consists of five steps:
- [00_simulator](notebooks/03_levy_flight_application/00_simulator.ipynb): Simulate training and validation data
- [01_pretrain_networks](notebooks/03_levy_flight_application/01_pretrain_networks.ipynb): Pretrain the network on simulated data with a reduced amount of trials per participant
- [02_finetune_networks](notebooks/03_levy_flight_application/02_finetune_networks.ipynb): Fine-tune the network on simulated data that contains the same amount of trials per participant as the empirical data
- [03_validate_networks](notebooks/03_levy_flight_application/03_validate_networks.ipynb): Validate the trained networks on new simulated data sets 
- [04_apply_networks](notebooks/03_levy_flight_application/04_apply_networks.ipynb): Apply the trained networks to the empirical data set

## [src](src)

Contains custom Julia and Python functions that enable the analyses.
