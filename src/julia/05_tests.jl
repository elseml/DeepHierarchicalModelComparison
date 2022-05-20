using Distributions
using AlphaStableDistributions
using Statistics
using Plots
using StatsPlots
using StatsFuns

include("datasets.jl")
include("diffusion.jl")
include("experiment.jl")
include("priors.jl")

# Single trial generation
generate_levy_trial(1.0, 0.5, 2.0, 0.5, 2.0, 0.3, 0.7, 0.2);

# Single participant generation
generate_levy_participant(100, 1.0, 0.5, -2.0, 2.0, 0.5, 2.0, 0.3, 0.7, 0.2);

# Single participant in two conditions generation
generate_levy_conditions(1.0, 0.5, -2.0, 2.0, 0.5, 2.0, 0.3, 0.7, 0.2);

# Hyperprior generation
generate_levy_hyperpriors();

# (lower level) Prior generation
h_test = generate_levy_hyperpriors()
generate_levy_priors(h_test...);

# Dataset generation by model
generate_levy_dataset_by_model(1, 40);
generate_levy_dataset_by_model(2, 40);
generate_levy_dataset_by_model(3, 40);
generate_levy_dataset_by_model(4, 40);

# Batch of datasets generation
generate_levy_batch(1, 2, 40, 900);
generate_levy_batch(2, 2, 40, 900);
generate_levy_batch(3, 2, 40, 900);
generate_levy_batch(4, 2, 40, 900);