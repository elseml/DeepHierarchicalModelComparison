import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score



def performance_metrics(bridge_sampling_results, NN_fixed_results, NN_variable_results, names, metrics):
    """ 
    Calculates various metrics to compare approximation performance.

    Parameters
    ----------
    bridge_sampling_results : pd.DataFrame
        Bridge Sampling approximations
    NN_fixed_results : pd.DataFrame
        Neural network (trained on fixed sample sizes) approximations
    NN_variable_results : pd.DataFrame
        Neural network (trained on varying sample sizes) approximations
    names : list
        Method names

    Returns
    -------
    df : pd.DataFrame
        Table with calculated performance metrics
    """
    
    mae = []
    rmse = []
    roc_auc = [] 
    accuracy = []
    
    for d in (bridge_sampling_results, NN_fixed_results, NN_variable_results):
        mae_temp = np.mean(abs(d['true_model']-d['m1_prob']))
        mae.append(mae_temp)
        rmse_temp = np.sqrt(((d['true_model']-d['m1_prob'])**2).mean())
        rmse.append(rmse_temp)
        roc_auc_temp = roc_auc_score(d['true_model'], d['m1_prob'])
        roc_auc.append(roc_auc_temp)
        accuracy_temp = (d['true_model'] == d['selected_model']).mean()
        accuracy.append(accuracy_temp)

    
    df = pd.DataFrame([mae, rmse, roc_auc, accuracy], index=metrics, 
                      columns = names).transpose()
    
    return df



def bootstrapped_metrics(bridge_sampling_results, NN_fixed_results, NN_variable_results, n_bootstrap, names, metrics):
    """
    Calculates bootstrapped performance metrics.

    Parameters
    ----------
    n_bootstrap : integer
        Number of bootstrap steps
    bridge_sampling_results : pd.DataFrame
        Bridge sampling approximations
    NN_fixed_results : pd.DataFrame
        Neural network (trained on fixed sample sizes) approximations
    NN_variable_results : pd.DataFrame
        Neural network (trained on varying sample sizes) approximations
    names : list
        Method names
    """

    n_test = bridge_sampling_results.shape[0]

    perf_metrics_bootstrapped = []

    for b in range(n_bootstrap):
        b_idx = np.random.choice(np.random.permutation(n_test), size=n_test, replace=True)
        perf_metrics = performance_metrics(
                            bridge_sampling_results.iloc[b_idx,:], 
                            NN_fixed_results.iloc[b_idx,:], 
                            NN_variable_results.iloc[b_idx,:], 
                            names, metrics
                            )
        perf_metrics_bootstrapped.append(perf_metrics)

    bootstrapped_mean = pd.DataFrame(np.mean(perf_metrics_bootstrapped, axis=0), index=names, columns=metrics)
    bootstrapped_sds = pd.DataFrame(np.std(perf_metrics_bootstrapped, axis=0), index=names, columns=metrics)

    return bootstrapped_mean, bootstrapped_sds