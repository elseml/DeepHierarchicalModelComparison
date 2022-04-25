"Generates reaction time datasets according to the four models in Wieschen, Voss and Radev (2020)."
function generate_levy_dataset_by_model(model::Int, n_clusters::Int64, n_trials::Int64)::Array{Float64, 3}

    data = fill(0.0, (n_clusters, n_trials, 3))
    hyperpriors = generate_levy_hyperpriors()

    for k in 1:n_clusters
        a_l, zr_l, v0_l, v1_l, t0_l, alpha_l, sz, sv, st = generate_levy_priors(hyperpriors...)
 
        if model == 1 || model == 3 # Gaussian noise
            alpha_l = 2.0
        end

        if model == 1 || model == 2 # Basic instead of full diffusion model
            sz = sv = st = 0.0 
        end

        data[k, :, :] = generate_levy_conditions(n_trials, a_l, zr_l, v0_l, v1_l, t0_l, alpha_l, sz, sv, st)
    end

    return data
end


"Generates a batch of datasets simulated from a given model."
function generate_levy_batch(model::Int64, batch_size::Int64, n_clusters::Int64, n_trials::Int64)::Tuple{Vector{Int64}, Array{Float64, 4}}

    data = fill(0.0, (batch_size, n_clusters, n_trials, 3))
    index_list = fill(model, batch_size)

    Threads.@threads for b in 1:batch_size
        data[b, :, :, :] = generate_levy_dataset_by_model(model, n_clusters, n_trials)
    end

    return index_list, data
end


# half-finished drafting for a joint function that returns a single datasets from all models. 
# Separate approach seems better right now in order to split long computation times into smaller parts.

#"Generates a batch of datasets simulated from all models."
#function generate_levy_batch_all_models(batch_size::Int64, n_clusters::Int64, n_trials::Int64)::Array{Float64, 4}
#
#    if batch_size%4 != 0
#        throw(ArgumentError("batch_size needs to be dividable by 4 without remainder."))
#    end
#
#    data = fill(0.0, (batch_size, n_clusters, n_trials, 3))
#
#    datasets_per_model = Int(batch_size/4)
#    base_model_index = repeat([1,2,3,4], inner=Int(datasets_per_model))
#
#    Threads.@threads for m in 1:4
#        Threads.@threads for d in 1:datasets_per_model
#            data[d, :, :, :] = generate_levy_dataset_by_model(base_model_index, n_clusters)
#    end

#    return data
#end