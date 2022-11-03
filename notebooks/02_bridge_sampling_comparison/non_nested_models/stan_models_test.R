#library(tidyverse)
library(rstan)
library(posterior)
library(reticulate)
np <- import("numpy")


### Set working directory to main folder
setwd(dirname(dirname(dirname(dirname(rstudioapi::getSourceEditorContext()$path)))))

### Load data and models
test_data <- np$load("data/02_bridge_sampling_comparison/non_nested_models/test_data.npy", allow_pickle = TRUE)[[1]][[1]]
test_data_single <- test_data[1,,,]
# Data set 1-3 are simulated from SDT, 4+9-11 from MPT
sdt_model_file <-"notebooks/02_bridge_sampling_comparison/non_nested_models/sdt_model.stan"
mpt_model_file <-"notebooks/02_bridge_sampling_comparison/non_nested_models/mpt_model.stan"

### Test the SDT model
sdt_model <- stan_model(file = sdt_model_file)
sdt_data <- list(
  M = dim(test_data_single)[1],  # Number of clusters.
  N = dim(test_data_single)[2],  # Number of observations.
  N_old_new = dim(test_data_single)[2]/2, # Number of old/new items, equal as 50/50 proportion of old and new items is given
  X = test_data_single[,,2] # 2D array of observations (without condition indicator).
)
sdt_fit <- sampling(sdt_model, data = sdt_data,
                    iter = 5000, warmup = 1000, chains = 4, cores = 4)
# 30sec for 5000 samples


### Test the MPT model
mpt_model <- stan_model(file = mpt_model_file)
mpt_data <- list(
  M = dim(test_data_single)[1],  # Number of clusters.
  N = dim(test_data_single)[2],  # Number of observations.
  N_old_new = dim(test_data_single)[2]/2, # Number of old/new items, equal as 50/50 proportion of old and new items is given
  X = test_data_single[,,2] # 2D array of observations (without condition indicator).
)
mpt_fit <- sampling(mpt_model, data = mpt_data,
                    iter = 5000, warmup = 1000, chains = 4, cores = 4)
# 190sec for 5000 samples


##### Diagnostics #####

# Inspect single stanfit objects

### Inspect traceplots
traceplot(sdt_fit, pars = c("mu_h", "sigma_h", "mu_f", "sigma_f"))
traceplot(mpt_fit, pars = c("mu_d", "mu_g", "lambdas[1]", "lambdas[2]",
                            "Q[1,1]", "Q[1,2]", "Q[2,1]", "Q[2,2]"))

### Convential n_eff and Rhat statistics
print(sdt_fit) 
print(mpt_fit)

#### Improved Rhat and ess statistics

## SDT model
# mu_h
Rhat(extract_variable_matrix(sdt_fit, "mu_h"))
ess_bulk(extract_variable_matrix(sdt_fit, "mu_h"))
ess_tail(extract_variable_matrix(sdt_fit, "mu_h"))

# sigma_h
Rhat(extract_variable_matrix(sdt_fit, "sigma_h"))
ess_bulk(extract_variable_matrix(sdt_fit, "sigma_h"))
ess_tail(extract_variable_matrix(sdt_fit, "sigma_h"))

# mu_f
Rhat(extract_variable_matrix(sdt_fit, "mu_f"))
ess_bulk(extract_variable_matrix(sdt_fit, "mu_f"))
ess_tail(extract_variable_matrix(sdt_fit, "mu_f"))

# sigma_f
Rhat(extract_variable_matrix(sdt_fit, "sigma_f"))
ess_bulk(extract_variable_matrix(sdt_fit, "sigma_f"))
ess_tail(extract_variable_matrix(sdt_fit, "sigma_f"))

## MPT model
# mu_d
Rhat(extract_variable_matrix(mpt_fit, "mu_d"))
ess_bulk(extract_variable_matrix(mpt_fit, "mu_d"))
ess_tail(extract_variable_matrix(mpt_fit, "mu_d"))

# mu_g
Rhat(extract_variable_matrix(mpt_fit, "mu_g"))
ess_bulk(extract_variable_matrix(mpt_fit, "mu_g"))
ess_tail(extract_variable_matrix(mpt_fit, "mu_g"))

# lambdas[1]
Rhat(extract_variable_matrix(mpt_fit, "lambdas[1]"))
ess_bulk(extract_variable_matrix(mpt_fit, "lambdas[1]"))
ess_tail(extract_variable_matrix(mpt_fit, "lambdas[1]"))

# lambdas[2]
Rhat(extract_variable_matrix(mpt_fit, "lambdas[2]"))
ess_bulk(extract_variable_matrix(mpt_fit, "lambdas[2]"))
ess_tail(extract_variable_matrix(mpt_fit, "lambdas[2]"))

# Q[1,1]
Rhat(extract_variable_matrix(mpt_fit, "Q[1,1]"))
ess_bulk(extract_variable_matrix(mpt_fit, "Q[1,1]"))
ess_tail(extract_variable_matrix(mpt_fit, "Q[1,1]"))

# Q[1,2]
Rhat(extract_variable_matrix(mpt_fit, "Q[1,2]"))
ess_bulk(extract_variable_matrix(mpt_fit, "Q[1,2]"))
ess_tail(extract_variable_matrix(mpt_fit, "Q[1,2]"))

# Q[2,1]
Rhat(extract_variable_matrix(mpt_fit, "Q[2,1]"))
ess_bulk(extract_variable_matrix(mpt_fit, "Q[2,1]"))
ess_tail(extract_variable_matrix(mpt_fit, "Q[2,1]"))

# Q[2,2]
Rhat(extract_variable_matrix(mpt_fit, "Q[2,2]"))
ess_bulk(extract_variable_matrix(mpt_fit, "Q[2,2]"))
ess_tail(extract_variable_matrix(mpt_fit, "Q[2,2]"))
