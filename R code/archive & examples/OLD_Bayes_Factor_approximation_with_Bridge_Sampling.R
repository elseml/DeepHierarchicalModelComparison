library(tidyverse)
library(rstan)
library(bayesplot)
library(bridgesampling)
library(reticulate)
np <- import("numpy")

### Set working directory to be the folder in which the script resides
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

### Import test datasets
test_data_bf <- np$load("test_data_bf.npy")
test_data_bf_true_indices <- np$load("test_data_bf_true_indices.npy")

# Input checks
dim(test_data_bf) # Check input dimensionality
#print(test_data_bf[1,1,,]) # Show data of the first person in the first dataset

# (For the beginning: start with only 1 dataset)
test_data_bf_single <- test_data_bf[3,,,]

### Specify Stan models

# FOR TESTING: model0 with target += normal_lpdf(theta[L] | 10, sqrt(tau2)); -> 10 instead of 0 to make the models more distinguishable

model0 <- 'data {
  int<lower = 1> N;               // Number of observations.
  int<lower = 1> L;               // Number of clusters.
  matrix[L, N] y;                 // Matrix of observations.
  real<lower=0> alpha_t;          // Higher order variance prior - shape parameter.
  real<lower=0> beta_t;           // Higher order variance prior - rate parameter.
  real<lower=0> alpha_s;          // Variance prior - shape parameter.
  real<lower=0> beta_s;           // Variance prior - rate parameter.
}
parameters {
  real<lower=0> tau2; // Group-level variance
  vector[L] theta; // Participant effects
  real<lower=0> sigma2; // Unit-level variance
}
model {
  target += inv_gamma_lpdf(tau2 | alpha_t, beta_t);
  target += normal_lpdf(theta | 0, sqrt(tau2));
  target += inv_gamma_lpdf(sigma2 | alpha_s, beta_s);  
  for (l in 1:L) {
      target += normal_lpdf(y[l] | theta[l], sqrt(sigma2));
  }
}
'
model1 <- 'data {
  int<lower = 1> N;               // Number of observations.
  int<lower = 1> L;               // Number of clusters.
  matrix[L, N] y;                 // Matrix of observations.
  real mu0;                       // higher order mean prior - mean
  real<lower=0> tau20;            // higher order mean prior - variance
  real<lower=0> alpha_t;          // Higher order variance prior - shape parameter.
  real<lower=0> beta_t;           // Higher order variance prior - rate parameter.
  real<lower=0> alpha_s;          // Variance prior - shape parameter.
  real<lower=0> beta_s;           // Variance prior - rate parameter.
}
parameters {
  real mu;
  real<lower=0> tau2; // Group-level variance
  vector[L] theta; // Participant effects
  real<lower=0> sigma2; // Unit-level variance
}
model {
  target += normal_lpdf(mu | mu0, sqrt(tau20));
  target += inv_gamma_lpdf(tau2 | alpha_t, beta_t);
  target += normal_lpdf(theta | mu, sqrt(tau2));
  target += inv_gamma_lpdf(sigma2 | alpha_s, beta_s);  
  for (l in 1:L) {
      target += normal_lpdf(y[l] | theta[l], sqrt(sigma2));
  }
}
'

### Compile Models
model0 <- stan_model(model_code = model0, model_name = "stanmodel")
model1 <- stan_model(model_code = model1, model_name = "stanmodel")

### Fit Models
# Prepare data & priors for sampling
data_and_priors0 <- list(
  N = dim(test_data_bf)[3],  # Number of observations.
  L = dim(test_data_bf)[2],  # Number of clusters.
  y = test_data_bf_single,   # Matrix of observations.
  alpha_t = 1,     
  beta_t = 1,
  alpha_s = 1,     
  beta_s = 1
)

data_and_priors1 <- list(
  N = dim(test_data_bf)[3],  # Number of observations.
  L = dim(test_data_bf)[2],  # Number of clusters.
  y = test_data_bf_single,   # Matrix of observations.
  mu0 = 0,                   # Difference to model0! 
  tau20 = 1,          
  alpha_t = 1,     
  beta_t = 1,
  alpha_s = 1,     
  beta_s = 1
)

# Fit
stanfit_model0 <- sampling(model0, data = data_and_priors0,
                      iter = 50000, warmup = 1000, chains = 3, cores = 1)

stanfit_model1 <- sampling(model1, data = data_and_priors1,
                           iter = 50000, warmup = 1000, chains = 3, cores = 1)

# Check chain mixing through trace plots
# (only for first 5 thetas to keep plotting time reasonable)
# stanfit_model0 %>%
#   mcmc_trace(
#     pars = c("tau2", str_c("theta[", 1:5, "]")),
#     n_warmup = 1000,
#     facet_args = list(nrow = 5, labeller = label_parsed)
#   )
# 
# stanfit_model1 %>%
#   mcmc_trace(
#     pars = c("tau2", str_c("theta[", 1:5, "]")),
#     n_warmup = 1000,
#     facet_args = list(nrow = 5, labeller = label_parsed)
#   )

### Bridge Sampling
# Compute the (Log) Marginal Likelihoods
m0.bridge <- bridge_sampler(stanfit_model0, silent = TRUE)
m1.bridge <- bridge_sampler(stanfit_model1, silent = TRUE)

print(m0.bridge)
print(m1.bridge)

# Compute percentage errors
m0.error <- error_measures(m0.bridge)$percentage
m1.error <- error_measures(m1.bridge)$percentage

print(m0.error)
print(m1.error)

# Compute Bayes factor
BF10 <- bf(m1.bridge,m0.bridge) # Attention: order different in blog post (m0,m1)
print(BF10)

# Compute posterior model probabilities (assuming equal prior model probabilities)
post1 <- post_prob(m0.bridge, m1.bridge)
print(post1)

### Evaluation of true model recovery
# Compute MAE
mae_metric <- mean(abs(test_data_bf_true_indices[1]-post1[2])) # [1] as we are only using the first dataset so far
print(mae_metric)

# Compute Accuracy
accuracy_metric <- mean(test_data_bf_true_indices[1] == round(post1[2])) # [1] as we are only using the first dataset so far
print(accuracy_metric)
