library(rstan)
library(loo)
library(bridgesampling)
library(bayesplot)


### Set working directory to be the folder in which the script resides
setwd(dirname(rstudioapi::getSourceEditorContext()$path))


### Load data
data = read.csv('data/regression.csv', header=T, sep=' ')


### Stan models
linear_model = stan_model('stan_models/linear_regression.stan')
quadratic_model = stan_model('stan_models/quadratic_regression.stan')
cubic_model = stan_model('stan_models/cubic_regression.stan')


### Stan data
stan_data_linear = list(x = data$x, 
                        y = data$y, 
                        N = nrow(data))


stan_data_quad = list(X = cbind(data$x, data$x^2),
                      y = data$y, 
                      N = nrow(data))


stan_data_cube = list(X = cbind(data$x, data$x^2, data$x^3),
                      y = data$y, 
                      N = nrow(data))


### Sample from models
fit_linear = sampling(linear_model, data = stan_data_linear, 
                      chains=4, warmup=1000,iter=4000, cores=4, seed=42)

fit_quad = sampling(quadratic_model, data = stan_data_quad, 
                    chains=4, warmup=1000, iter=4000, cores=4, seed=42)

fit_cube = sampling(cubic_model, data = stan_data_cube, chains=4, 
                    warmup=1000,iter=4000, cores=4, seed=42)

### Check some plots and predictions

# Your code here


### Posterior predictive model comparison

loglik_linear = extract_log_lik(fit_linear, parameter_name = 'll', merge_chains = F)
loglik_quad = extract_log_lik(fit_quad, parameter_name = 'll', merge_chains=F)
loglik_cube = extract_log_lik(fit_cube, parameter_name = 'll', merge_chains=F)
r_eff_linear = relative_eff(loglik_linear)
r_eff_quad = relative_eff(loglik_quad)
r_eff_cube = relative_eff(loglik_cube)


# LOO (PSIS-LOO-CV)
loo_linear = loo(loglik_linear, r_eff=r_eff_linear)
loo_quad = loo(loglik_quad, r_eff=r_eff_quad)
loo_cube = loo(loglik_cube, r_eff=r_eff_cube)

# WAIC
waic_linear = waic(rstan::extract(fit_linear, pars='ll')$ll)
waic_quad = waic(rstan::extract(fit_quad, pars='ll')$ll)
waic_cube = waic(rstan::extract(fit_cube, pars='ll')$ll)

# Inspect comparison
loo_compare(loo_linear, loo_quad, loo_cube)
loo_compare(waic_linear, waic_quad, waic_cube)

# Conclusion?


### Model comparison with Bayes factors

# Run bridge samplers
linear_bridge = bridge_sampler(fit_linear, silent=T)
quad_bridge = bridge_sampler(fit_quad, silent=T)
cube_bridge = bridge_sampler(fit_cube, silent=T)

# Compute Bayes factors
bf(linear_bridge, quad_bridge)
bf(quad_bridge, cube_bridge)
bf(linear_bridge, cube_bridge)

# Compute model posterior probabilities
post_prob(linear_bridge, quad_bridge, cube_bridge)
