# m0 model fit
m0 <- as.matrix(stanfit_model0)
m0_means <- colMeans(m0)

mean(m0_means[2:54]) # mean(theta)
var(m0_means[2:54])  # var(theta)
m0_means['sigma2']   # sigma2
m0_means['tau2']     # tau2

# m1 model fit
m1 <- as.matrix(stanfit_model1)
m1_means <- colMeans(m1)

mean(m1_means[3:55]) # mean(theta)
var(m1_means[3:55])  # var(theta)
m1_means['sigma2']   # sigma2
m1_means['mu']       # mu
m1_means['tau2']     # tau2
