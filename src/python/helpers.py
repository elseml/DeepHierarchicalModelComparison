import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import truncnorm
from tensorflow.keras.utils import to_categorical
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



# Levy flight application: Load and transform data


def load_simulated_rt_data(folder, indices_filename, datasets_filename):
    """Loads and transforms simulated reaction time data.

    Parameters
    ----------
    folder : string
        Path to the folder containing the files
    indices_filename : string
    datasets_filename : string
    """

    indices = np.load(os.path.join(folder, indices_filename))
    datasets = np.load(os.path.join(folder, datasets_filename))

    # unpack indices
    indices = indices[:,0,0,0]-1

    # one-hot encode indices
    indices = to_categorical(indices, num_classes=4)

    return indices, datasets


def load_empirical_rt_data(load_dir):
    """
    Reads single subject datasets from a folder and transforms into list of 4D-arrays 
    which allows for a variable number of observations between participants.
    Assumes data files have a three-column csv format (Condition, Response, Response Time).
    ----------
    
    Arguments:
    load_dir : str -- a string indicating the directory from which to load the data
    --------
        
    Returns:
    X: list of length (n_clusters), containing tf.Tensors of shape (1, 1, n_obs, 3) 
        -- variable order now (Condition, Response Time, Response)
    """
    
    data_files = os.listdir(load_dir)
    X = []
    
    # Loop through data files and estimate
    for data_file in data_files:
        
        ### Read in and transform data
        data = pd.read_csv(os.path.join(load_dir, data_file), header=None, sep=' ')
        data = data[[0,2,1]].values # reorder columns
        X_file = tf.convert_to_tensor(data, dtype=tf.float32)[np.newaxis][np.newaxis] # get 4D tensor
        X.append(X_file)
      
    return X


def mask_inputs(data_sets, missings_mean, missings_sd):
    """Masks some training inputs with -1 so that training leads to a robust net that can handle missing data

    Parameters
    ----------
    data_sets : np.array
        simulated training data sets
    missings_mean : float
        empirical mean of missings per person
    missings_sd : float
        empirical sd of missings per person

    Returns
    -------
    data_sets : np.array
        simulated training data sets with masked inputs
    """

    n_data_sets = data_sets.shape[0]
    n_persons = data_sets.shape[1]
    n_trials = data_sets.shape[2]
    n_variables = data_sets.shape[3]

    # create truncated normal parameterization in accordance with scipy documentation
    a, b = (0 - missings_mean) / missings_sd, (n_trials - missings_mean) / missings_sd 

    for d in range(n_data_sets):
        # draw number of masked values per person from truncated normal distribution
        masks_per_person = truncnorm.rvs(a, b, loc=missings_mean, scale=missings_sd, size=n_persons).round().astype(int)
        # assign the specific trials to be masked within a person 
        mask_positions = [np.random.randint(0, n_trials, size=(n_persons, j)) for j in masks_per_person][0]
        for j in range(n_persons):
            data_sets[d,j,:,:][mask_positions[j]] = -1

    # assert that the average amount of masks per person matches the location of the truncnormal dist
    deviation = ((data_sets[:,:,:,:] == -1).sum()/(n_data_sets*n_persons*n_variables)) - missings_mean
    assert deviation < 1, f"Average amount of masks per person deviates by {deviation} from missings_mean!"

    return data_sets