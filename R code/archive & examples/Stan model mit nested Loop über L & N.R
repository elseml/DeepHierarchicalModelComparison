model0 <- 'data {
  int<lower = 1> N;               // Number of observations.
  int<lower = 1> L;               // Number of clusters.
  matrix[L, N] y;                 // Matrix of observations.
  real<lower=0> alpha;            // Higher order variance prior - shape parameter.
  real<lower=0> beta;             // Higher order variance prior - rate parameter.
  real<lower=0> sigma2;           // Intra-cluster variance.
}
parameters {
  real<lower=0> tau2; // Group-level variance
  vector[L] theta; // Participant effects
}
model {
  target += inv_gamma_lpdf(tau2 | alpha, beta);
  target += normal_lpdf(theta[L] | 100, sqrt(tau2));
  for (l in 1:L) {
    for (n in 1:N) {
      target += normal_lpdf(y[l,n] | theta[l], sqrt(sigma2));
    }
  }
}
'
model1 <- 'data {
  int<lower = 1> N;               // Number of observations.
  int<lower = 1> L;               // Number of clusters.
  matrix[L, N] y;                 // Matrix of observations.
  real mu0;                       // higher order mean prior - mean
  real<lower=0> tau20;            // higher order mean prior - variance
  real<lower=0> alpha;            // Higher order variance prior - shape parameter.
  real<lower=0> beta;             // Higher order variance prior - rate parameter.
  real<lower=0> sigma2;           // Intra-cluster variance.
}
parameters {
  real mu;
  real<lower=0> tau2; // Group-level variance
  vector[L] theta; // Participant effects
}
model {
  target += normal_lpdf(mu | mu0, sqrt(tau20));
  target += inv_gamma_lpdf(tau2 | alpha, beta);
  target += normal_lpdf(theta[L] | mu, sqrt(tau2));
  for (l in 1:L) {
    for (n in 1:N) {
      target += normal_lpdf(y[l,n] | theta[l], sqrt(sigma2));
    }
  }
}
'