from .helpers import log_with_inf_noise_addition

import numpy as np
np.set_printoptions(suppress=True)
import tensorflow as tf
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = 'w'
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error



# Calibration: Plotting for training with fixed numbers of clusters and observations

def plot_confusion_matrix(m_true, m_pred, model_names, ax, normalize=True, 
                          cmap=plt.cm.Blues, annotate=True):
    """
    Helper function to print and plot the confusion matrix. 
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(m_true, m_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=model_names, yticklabels=model_names,
           ylabel='True Model',
           xlabel='Predicted Model')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if annotate:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    ax.set_title('Confusion Matrix')

    
def plot_calibration_curve(m_true, m_pred, n_bins, pub_style, ax):
    """Helper function to plot calibration curve and ece."""
    
    prob_true, prob_pred = calibration_curve(m_true, m_pred, n_bins=n_bins)
    cal_err = np.mean(np.abs(prob_true - prob_pred))
    if pub_style == True:
        ax.plot(prob_true, prob_pred, color='black')
        ax.plot(ax.get_xlim(), ax.get_xlim(), '--', color='darkgrey')
        print('ECE = {0:.3f}'.format(cal_err))

    elif pub_style == False:
        ax.plot(prob_true, prob_pred)
        ax.plot(ax.get_xlim(), ax.get_xlim(), '--')
        ax.text(0.1, 0.9,  'ECE = {0:.3f}'.format(cal_err),
                    horizontalalignment='left',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    size=12)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Confidence')
    sns.despine(ax=ax)
    
    
def plot_bootstrap_accuracy(m_true, m_pred, n_bootstrap, ax):
    """Helper function to plot the bootstrap accuracy of recovery."""
    
    n_test = m_true.shape[0]
    accs = []
    for bi in range(n_bootstrap):
        b_idx = np.random.choice(np.random.permutation(n_test), size=n_test, replace=True)
        m_true_b, m_pred_b = m_true[b_idx], m_pred[b_idx]
        accs.append(accuracy_score(m_true_b, m_pred_b))
    
    sns.histplot(accs, ax=ax, stat='probability')
    sns.despine(ax=ax)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('')
    ax.yaxis.set_ticks([])
    ax.text(0.05, 0.9,  'Bootstrap accuracy = {0:.3f}'.format(np.mean(accs)),
                        horizontalalignment='left',
                        verticalalignment='center',
                        transform=ax.transAxes,
                        size=12)
    ax.text(0.05, 0.8,  'SD = {0:.3f}'.format(np.std(accs)),
                        horizontalalignment='left',
                        verticalalignment='center',
                        transform=ax.transAxes,
                        size=12)
    
    
def plot_bootstrap_mae(m_true, m_pred, n_bootstrap, ax):
    """Helper function to plot the bootstrap accuracy of recovery."""
    
    n_test = m_true.shape[0]
    maes = []
    for bi in range(n_bootstrap):
        b_idx = np.random.choice(np.random.permutation(n_test), size=n_test, replace=True)
        m_true_b, m_pred_b = m_true[b_idx], m_pred[b_idx]
        maes.append(mean_absolute_error(m_true_b, m_pred_b))
    
    sns.histplot(maes, ax=ax, stat='probability')
    sns.despine(ax=ax)
    ax.set_xlabel('MAE')
    ax.set_ylabel('')
    ax.yaxis.set_ticks([])
    ax.text(0.05, 0.9,  'Mean MAE = {0:.3f}'.format(np.mean(maes)),
                        horizontalalignment='left',
                        verticalalignment='center',
                        transform=ax.transAxes,
                        size=12)
    ax.text(0.05, 0.8,  'SD = {0:.3f}'.format(np.std(maes)),
                        horizontalalignment='left',
                        verticalalignment='center',
                        transform=ax.transAxes,
                        size=12)
    

def perf_tester(evidence_net, summary_net, val_data, n_bootstrap=100, n_cal_bins=15, pub_style=False, save=False):
    """Helper function to test the performance of the model."""
    
    # Compute model predictions in chunks so GPU memory does not blow-up
    m_soft = tf.concat([evidence_net.predict(summary_net(x_chunk))['m_probs'][:, 1] for x_chunk in tf.split(val_data['X'], 20)], axis=0).numpy()
    m_hard = (m_soft > 0.5).astype(np.int32)
    m_true = val_data['m'][:, 1]
    
    if pub_style == False:
        # Prepare figures
        fig, axarr = plt.subplots(2, 2, figsize=(16, 8))
        
        # Plot stuff
        plot_calibration_curve(m_true, m_soft, n_cal_bins, pub_style, axarr[0, 0])
        plot_confusion_matrix(m_true, m_hard, ['M1', 'M2'], axarr[0, 1])
        plot_bootstrap_accuracy(m_true, m_hard, n_bootstrap, axarr[1, 0])
        plot_bootstrap_mae(m_true, m_hard, n_bootstrap, axarr[1, 1])
        fig.tight_layout()

    elif pub_style == True:
        fig, ax = plt.subplots(figsize=(5,5))
        plot_calibration_curve(m_true, m_soft, n_cal_bins, pub_style, ax)
        if save == True:
            fig.savefig('Calibration_fixed_sizes.png', dpi=300, bbox_inches='tight')



# Calibration: Plotting for training with fixed numbers of clusters and variable number of observations

def plot_eces_over_obs(m_true, m_pred, n_obs_min, n_obs_max, n_bins, pub_style, save):
    """Helper function to plot ece as a function of N."""

    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    n_obs_points = np.arange(n_obs_min, n_obs_max+1)
    prob_true, prob_pred, cal_err = ([] for i in range(3))
    
    for i in range(n_obs_max-(n_obs_min-1)):
        prob_true_i, prob_pred_i = calibration_curve(m_true[i], m_pred[i], n_bins=n_bins)
        prob_true.append(prob_true_i)
        prob_pred.append(prob_pred_i)
        cal_err_i = np.mean(np.abs(prob_true[i] - prob_pred[i]))
        cal_err.append(cal_err_i)
    mean_ece = np.mean(cal_err)
    sd_ece = np.std(cal_err)

    if pub_style == False:
        ax.plot(n_obs_points, cal_err)
        plt.axhline(y=mean_ece, color='tab:red')
        plt.fill_between(n_obs_points, mean_ece-3*sd_ece, mean_ece+3*sd_ece, 
                     color='tab:red', alpha=0.1)
        ax.set_ylim([0, 1])
        ax.text(0.1, 0.9,  'Mean ECE = {0:.3f}'.format(mean_ece),
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes,
                size=12)
        ax.text(0.1, 0.85,  'Shaded region: Mean ECE +/- 3SD ',
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes,
                size=12)    
        ax.set_title('Expected Calibration Error (ECE)')

    elif pub_style == True:
        ax.plot(n_obs_points, cal_err, color='black')
        plt.axhline(y=mean_ece, linestyle='--', color='darkgrey')
        ax.set_ylim([0, 0.4])
        print(cal_err[:5])

    ax.set_xlim([n_obs_min, n_obs_max])
    ax.set_xlabel('$N$')
    ax.set_ylabel('ECE')

    if save == True:
        f.savefig('Calibration_variable_observations.png', dpi=300, bbox_inches='tight')

    
def perf_tester_over_obs(evidence_net, summary_net, val_data, n_obs_min, n_obs_max, n_cal_bins=15, pub_style=False, save=False):
    """Utility function to test the performance of the model."""
    
    # Compute model predictions in chunks so GPU memory does not blow-up
    m_soft, m_hard, m_true = ([] for i in range(3))
    
    for i in range(n_obs_max-(n_obs_min-1)):
        m_soft_i = tf.concat([evidence_net.predict(summary_net(x_chunk))['m_probs'][:, 1] for x_chunk in tf.split(val_data['X'][i], 5)], axis=0).numpy()
        m_soft.append(m_soft_i)
        m_hard_i = (m_soft[i] > 0.5).astype(np.int32)
        m_hard.append(m_hard_i)
        m_true_i = val_data['m'][i][:, 1]
        m_true.append(m_true_i)

    # Plot/save stuff
    plot_eces_over_obs(m_true, m_soft, n_obs_min, n_obs_max, n_cal_bins, pub_style, save)



# Calibration: Plotting for training with variable numbers of clusters and variable number of observations

def compute_eces_variable(evidence_net, summary_net, simulator, n_val_per_setting, n_clust_min, n_clust_max, n_obs_min, n_obs_max, n_cal_bins=15):
    """
    Simulates validation data per setting and computes the expected calibration error of the model.
    --------

    Returns:
    2 lists of shape((n_clust_max+1 - n_clust_min)*(n_obs_max+1 - n_obs_min)) 
    - containing the mean (1st list) / sd (2nd list) eces of all possible combinations on L and N.
    """
    
    def n_clust_obs_f_v_val(l, n):
        """
        Nasty hack to make compatible with BayesFlow.
        Defines a fixed number of clusters and a number of observations that is iterated through.
        """
        
        K = l
        N = n
        return (K, N)
    
    # Create lists
    ece_means = []
    
    with tqdm(total=(n_clust_max+1 - n_clust_min), desc='Loop through clusters progress') as p_bar: 
        with tqdm(total=(n_obs_max+1 - n_obs_min), desc='Loop through nested observations progress') as p_bar_within:
            for l in range(n_clust_min, n_clust_max+1): # Loop through clusters
                
                p_bar_within.reset((n_obs_max+1 - n_obs_min)) # reuse 2nd bar so that screen doesn't explode
                for n in range(n_obs_min, n_obs_max+1): # Loop through nested observations

                    # Simulate validation data
                    m_val_sim, _, x_val_sim = simulator(n_val_per_setting, n_clust_obs_f_v_val(l, n))

                    # Predict model probabilities
                    #m_soft = evidence_net.predict(summary_net(x_val_sim))['m_probs'][:, 1]
                    m_soft = tf.concat([evidence_net.predict(summary_net(x_chunk))['m_probs'][:, 1] for x_chunk in tf.split(x_val_sim, 20)], axis=0).numpy()      
                    m_true = m_val_sim[:, 1]  

                    # Compute calibration error
                    prob_true, prob_pred = calibration_curve(m_true, m_soft, n_bins=n_cal_bins)
                    cal_err = np.abs(prob_true - prob_pred)

                    mean_ece = np.mean(cal_err)
                    ece_means.append(mean_ece)

                    # Update inner progress bar
                    p_bar_within.set_postfix_str("Cluster {0}, Observation {1}".format(l, n + 1))
                    p_bar_within.update()

                # Refresh inner + update outer progress bar
                p_bar_within.refresh() 
                p_bar.set_postfix_str("Finished clusters: {}".format(l))
                p_bar.update()
    
    return(ece_means)


def plot_eces_variable(ece_means, n_clust_min, n_clust_max, n_obs_min, n_obs_max, save=False):
    """ 
    Takes the ECE results from compute_eces_variable() and 
    projects them onto a 3D-plot.
    """
    
    # Prepare objects
    #f = plt.figure(figsize=(16, 16))
    f = plt.figure(figsize=(6, 6))
    ax = plt.axes(projection='3d')

    n_clust_points = np.arange(n_clust_min, n_clust_max+1)
    n_obs_points = np.arange(n_obs_min, n_obs_max+1)

    n_clust_grid, n_obs_grid = np.meshgrid(n_clust_points, n_obs_points)
    cal_err_grid = np.reshape(ece_means, (-1, n_clust_max)) # reshape into (#clusters, #observations)
    ax.plot_surface(n_clust_grid, n_obs_grid, cal_err_grid, cmap='coolwarm', edgecolor='none')

    ax.elev = 15
    ax.set_xlim([n_clust_min, n_clust_max]) 
    ax.set_ylim([n_obs_max, n_obs_min])
    ax.set_zlim([0, 0.3])
    ax.set_xlabel('$J$')
    ax.set_ylabel('$N$')    
    ax.set_zlabel('ECE')
    ax.text2D(0.05, 0.95, 'Mean ECE = {0:.3f}'.format(np.mean(ece_means)), transform=ax.transAxes)
    ax.text2D(0.05, 0.90, 'SD around mean ECE = {0:.3f}'.format(np.std(ece_means)), transform=ax.transAxes)
    #ax.set_title('Expected Calibration Error (ECE)')
    if save == True:
        f.savefig('Calibration_variable_sizes.png', dpi=300, bbox_inches='tight')
    

# TODO: MAKE FUNCTION GENERAL TO WORK FOR EITHER OBS OR CLUSTERS
def plot_eces_means(ece_means, n_clust_min, n_clust_max, n_obs_min, n_obs_max, x_axis):
    """Helper function to plot ece means over clusters (1 = inspect observations) or observations (0 = inspect clusters)."""

    if x_axis == 0:
        xlabel='$J$'
        axis = 0
        n_min, n_max = n_clust_min, n_clust_max

    elif x_axis == 1:
        xlabel='$N$'
        axis = 1
        n_min, n_max = n_obs_min, n_obs_max

    # Average marginalized dimension
    ece_means_reshaped = np.reshape(ece_means, (-1, n_clust_max)) # reshape into (#clusters, #observations)
    ece_means_marginalized = np.mean(ece_means_reshaped, axis = axis)
    print(ece_means_marginalized[:5]) # inspect ECEs for small number of clusters/observations

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    n_points = np.arange(n_min, n_max+1)
    
    ax.plot(n_points, ece_means_marginalized, color='black')
    grand_mean_ece = np.mean(ece_means_marginalized)
    plt.axhline(y=grand_mean_ece, linestyle='--', color='darkgrey')
    ax.set_xlim([n_min, n_max])
    ax.set_ylim([0, 0.4])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('ECE')


# Bridge sampling comparison: Plot approximations

def plot_approximations(bridge_sampling_results, NN_results, approximated_outcome, NN_name, 
                        figsize=(5, 5), colors = {0:'darkgrey', 1:'black'}):
    """Plots and contrasts the approximations generated by bridge sampling and a neural network.

    Parameters
    ----------
    bridge_sampling_results : pd.DataFrame
        Bridge sampling approximations
    NN_results : pd.DataFrame
        Neural network approximations
    approximated_outcome : boolean
        Indicates whether posterior model probabilities (0) or Log Bayes factors (1) shall be plotted
    NN_name : string
        Indicates the type of neural network (fixed/variable)
    """

    f, ax = plt.subplots(1, 1, figsize=figsize)

    if approximated_outcome == 0: # PMPs
        bridge_data = bridge_sampling_results['m1_prob']
        NN_data = NN_results['m1_prob']
        label_outcome = '$p$($M_{2}$|$D$)'
        
    elif approximated_outcome == 1: # Log BFs
        bridge_data = log_with_inf_noise_addition(bridge_sampling_results)
        NN_data = log_with_inf_noise_addition(NN_results)
        label_outcome = 'Log $BF_{21}$'

    label_NN = NN_name

    ax.scatter(bridge_data, NN_data, c=bridge_sampling_results['true_model'].map(colors))
    ax.plot(ax.get_xlim(), ax.get_xlim(), '--', color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(label_outcome + ', ' + 'bridge sampling')
    ax.set_ylabel(label_outcome + ', ' + label_NN)

