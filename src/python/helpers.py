import numpy as np
import pandas as pd

from time import perf_counter


# Calibration: Hacks for BayesFlow compatibility. 

# fixed data set sizes
def n_clust_obs_f_f():
    """
    Nasty hack to make compatible with BayesFlow.
    Defines a fixed number of clusters and observations.
    """
    
    K = 50
    N = 50
    return (K, N)


# variable number of observations between data sets
def n_clust_obs_f_v(n_obs_min, n_obs_max):
    """
    Nasty hack to make compatible with BayesFlow.
    Defines a fixed number of clusters and a variable number of observations.
    """
    
    K = 50
    N = np.random.randint(n_obs_min, n_obs_max)
    return (K, N)


# apply nasty hack to validation data generation to work with BayesFlow simulator
def n_clust_obs_f_v_val(n):
    """
    Nasty hack to make compatible with BayesFlow.
    Defines a fixed number of clusters and a number of observations that is iterated through.
    """
    
    K = 50
    N = n
    return (K, N)

# variable data set sizes
def n_clust_obs_v_v(n_clust_min, n_clust_max, n_obs_min, n_obs_max):
    """
    Nasty hack to make compatible with BayesFlow.
    Defines a variable number of clusters and observations.
    """
    
    K = np.random.randint(n_clust_min, n_clust_max)
    N = np.random.randint(n_obs_min, n_obs_max)
    return (K, N)



# Bridge sampling comparison: data transformations

def get_preds_and_bfs(evidence_net, summary_net, data, training_time_start, training_time_stop, losses):
    """ 
    Writes model predictions and resulting Bayes Factors for a given 
    array of datasets into a pandas DataFrame. 
    """

    dataset = np.arange(1,data['X'].shape[0]+1)
    true_model = data['m']

    # Predict
    inference_time_start = perf_counter()
    m1_prob = np.array(evidence_net.predict(summary_net(data['X']))['m_probs'][:, 1], dtype = np.longdouble)
    inference_time_stop = perf_counter()
    m0_prob = 1 - m1_prob
    selected_model = (m1_prob > 0.5)

    # Bayes Factors
    bayes_factor = m1_prob / m0_prob
    
    # Times
    training_time = np.repeat((training_time_stop-training_time_start), 100)
    inference_time = np.repeat((inference_time_stop-inference_time_start), 100)
    
    # Final epoch mean loss
    final_epoch_loss = np.repeat(np.mean(losses[10]), 100)

    # Create DataFrame
    vals = np.c_[dataset, true_model, m0_prob, m1_prob, selected_model, bayes_factor,
                 training_time, inference_time, final_epoch_loss]
    names = ['dataset', 'true_model', 'm0_prob', 'm1_prob', 'selected_model', 'bayes_factor',
             'training_time', 'inference_time', 'final_epoch_loss']
    df = pd.DataFrame(vals, columns = names)
    df[["dataset", "true_model", "selected_model"]] = df[["dataset", "true_model", "selected_model"]].astype(int)
    
    return df


def log_with_inf_noise_addition(x):
    """ 
    Adjusts the model probabilities leading to Inf values by a minimal amount of noise, 
    recomputes the Bayes factors and then computes the log of the given array. 
    """
    
    noise = 0.000000001

    x_copy = x.copy()
    for i in range(x.shape[0]):
        if x.loc[i,'m0_prob'] == 0:
            print('Dataset with infinite BF: {}'.format(i))
            x_copy.loc[i,'m0_prob'] = x_copy.loc[i,'m0_prob'] + noise
            x_copy.loc[i,'m1_prob'] = x_copy.loc[i,'m1_prob'] - noise
            x_copy.loc[i,'bayes_factor'] = x_copy.loc[i,'m1_prob'] / x_copy.loc[i,'m0_prob']
    x_copy = np.log(x_copy['bayes_factor'])
    return x_copy


def computation_times(bridge_sampling_results, NN_fixed_results, NN_variable_results):
    """Calculates the computation times of bridge sampling and the two neural network variants.

    Parameters
    ----------
    bridge_sampling_results : pd.DataFrame
        Bridge sampling approximations
    NN_fixed_results : pd.DataFrame
        Neural network (trained on fixed sample sizes) approximations
    NN_variable_results : pd.DataFrame
        Neural network (trained on varying sample sizes) approximations
    """

    # Calculate computation times
    bridge_time = (bridge_sampling_results['compile_time'] + 
                        (bridge_sampling_results['stan_time'] + bridge_sampling_results['bridge_time']).cumsum()
                        )/60
    NN_fixed_time = (NN_fixed_results['training_time'] + NN_fixed_results['inference_time'].cumsum())/60
    NN_variable_time = (NN_variable_results['training_time'] + NN_variable_results['inference_time'].cumsum())/60

    # Adjust index to represent datasets
    bridge_time.index = bridge_time.index+1
    NN_fixed_time.index = NN_fixed_time.index+1
    NN_variable_time.index = NN_variable_time.index+1

    return bridge_time, NN_fixed_time, NN_variable_time