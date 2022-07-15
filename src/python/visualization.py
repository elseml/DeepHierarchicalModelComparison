from .helpers import log_with_inf_noise_addition
from .settings import plotting_settings

import numpy as np
np.set_printoptions(suppress=True)
import tensorflow as tf
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams.update({'mathtext.fontset': 'cm'}) # Enable mathcal for model labels
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error

# Calibration: Plotting for training with fixed numbers of clusters and observations

def plot_confusion_matrix(m_true, m_pred, model_names, ax, normalize=True, 
                          cmap='viridis', annotate=True, title=True):
    """
    Helper function to print and plot the confusion matrix. 
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(m_true, m_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, shrink=0.7)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]))

    ax.set_xticklabels(model_names, fontsize=plotting_settings['fontsize_labels'])
    ax.set_yticklabels(model_names, fontsize=plotting_settings['fontsize_labels'])
    ax.set_xlabel('Predicted model', fontsize=plotting_settings['fontsize_labels'])
    ax.set_ylabel('True model', fontsize=plotting_settings['fontsize_labels']) 

    # Loop over data dimensions and create text annotations.
    if annotate:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] < thresh else "black")
    if title:
        ax.set_title('Confusion Matrix')

    
def plot_calibration_curve(m_true, m_pred, n_bins, ax, xlabel=True, ylabel=True, title=None, show_ece=True):
    """Helper function to plot the calibration curve and ece."""
    
    prob_true, prob_pred = calibration_curve(m_true, m_pred, n_bins=n_bins)
    cal_err = np.mean(np.abs(prob_true - prob_pred))

    ax.plot((0,1), (0,1), '--', color='darkgrey')
    ax.plot(prob_true, prob_pred, color=plotting_settings['colors_discrete'][0])
    print('ECE = {0:.3f}'.format(cal_err))

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    if xlabel:
        ax.set_xlabel('Accuracy')
    if ylabel:
        ax.set_ylabel('Confidence')
    ax.set_title(title, fontsize=plotting_settings['fontsize_labels'])
    ax.grid(alpha=.3)
    if show_ece:
        ax.text(0.1, 0.9, r'$\widehat{{\mathrm{{ECE}}}} = {0:.3f}$'.format(cal_err),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes, size=12)
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
    """Helper function to plot the bootstrap mean absolute error."""
    
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
        plot_calibration_curve(m_true, m_soft, n_cal_bins, axarr[0, 0])
        plot_confusion_matrix(m_true, m_hard, ['M1', 'M2'], axarr[0, 1])
        plot_bootstrap_accuracy(m_true, m_hard, n_bootstrap, axarr[1, 0])
        plot_bootstrap_mae(m_true, m_hard, n_bootstrap, axarr[1, 1])
        fig.tight_layout()

    elif pub_style == True:
        fig, ax = plt.subplots(figsize=(5,5))
        plot_calibration_curve(m_true, m_soft, n_cal_bins, ax)
        if save == True:
            fig.savefig('calibration_fixed_sizes.png', dpi=300, bbox_inches='tight')


def plot_calibration_curve_uncertainty(m_true, pm_samples_model, narrow_ci, wide_ci, n_bins, ax, 
                                       xlabel=True, ylabel=True, title=None, show_ece=True, show_legend=True):
    """ Plots a calibration curve for a single model with uncertainty (median line and two credible intervals).
        Caution: Can throw a ValueError if one of the samples produces an empty bin (i.e. less than 15 bins) -> repeat.

    Parameters
    ----------
    m_true : np.array
            One-dimensional array that indicates whether the model is true or not. 
    pm_samples_model : np.array
            Two-dimensional array containing the posterior samples of the model's probability for each data set.
    narrow_ci : list
            The quantiles of the narrow credible interval.
    wide_ci : list
            The quantiles of the wide credible interval.
    n_bins : int
            Number of calibration bins.
    ax : matplotlib Axes
            Matplot axes object specifying the plot that should be used.
    xlabel : boolean
            Controls if the x-axis label is shown.
    ylabel : boolean
            Controls if the y-axis label is shown.
    title : str
            An optional title that can be provided.
    show_ece : boolean
            Controls if the Expected Calibration Error is shown.
    show_legend : boolean
            Controls if a legend is shown.
    """

    n_samples = pm_samples_model.shape[1]

    # Get bins & ECE for each sample from the learned dirichlet distribution
    probs_true = np.zeros((n_samples, n_bins))
    probs_pred = np.zeros((n_samples, n_bins))

    for n in range(n_samples):
        m_soft = pm_samples_model[:, n]
        probs_true[n, :], probs_pred[n, :] = calibration_curve(m_true, m_soft, n_bins=15)
    cal_err = np.mean(np.abs(probs_true - probs_pred))

    # Get median for each bin 
    probs_true_median = np.squeeze(np.quantile(probs_true, q=[0.5], axis=0))
    probs_pred_median = np.squeeze(np.quantile(probs_pred, q=[0.5], axis=0))

    # Get quantiles for each bin 
    # Narrow credible interval
    probs_true_narrow = np.quantile(probs_true, q=narrow_ci, axis=0)
    probs_pred_narrow = np.quantile(probs_pred, q=narrow_ci, axis=0)
    # Wide credible interval
    probs_true_wide = np.quantile(probs_true, q=wide_ci, axis=0)
    probs_pred_wide = np.quantile(probs_pred, q=wide_ci, axis=0)

    # Plot median curve and diagonal
    ax.plot((0,1), (0,1), '--', color='darkgrey')
    ax.plot(probs_true_median, probs_pred_median, color=plotting_settings['colors_discrete'][0])

    # Plot credible intervals
    ax.fill(np.append(probs_true_narrow[0,:], probs_true_narrow[1,:][::-1]),
            np.append(probs_pred_narrow[0,:], probs_pred_narrow[1,:][::-1]),
            color=plotting_settings['colors_discrete'][0], alpha=0.3, label='{:.0%} CI'.format(narrow_ci[1]-narrow_ci[0]))
    ax.fill(np.append(probs_true_wide[0,:], probs_true_wide[1,:][::-1]),
            np.append(probs_pred_wide[0,:], probs_pred_wide[1,:][::-1]),
            color=plotting_settings['colors_discrete'][0], alpha=0.2, label='{:.0%} CI'.format(wide_ci[1]-wide_ci[0]))   

    # Format plot
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    if xlabel:
            ax.set_xlabel('Accuracy')
    if ylabel:
            ax.set_ylabel('Confidence')
    ax.set_title(title, fontsize=plotting_settings['fontsize_labels'])
    ax.grid(alpha=.3)
    if show_ece:
            ax.text(0.1, 0.9, r'$\widehat{{\mathrm{{ECE}}}} = {0:.3f}$'.format(cal_err),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes, size=12)
    if show_legend:
            ax.legend(loc="lower right", fontsize=12)
    sns.despine(ax=ax)

# Calibration: Plotting for training with fixed numbers of clusters and variable number of observations

def plot_eces_over_obs(m_true, m_pred, n_obs_min, n_obs_max, n_bins, pub_style, save=False):
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
        ax.plot(n_obs_points, cal_err, color=plotting_settings['colors_discrete'][0])
        plt.axhline(y=mean_ece, linestyle='--', color='darkgrey')
        ax.set_ylim([0, 0.3])

    ax.set_xlim([n_obs_min, n_obs_max])
    ax.set_xlabel('Number of observations ($N$)')
    ax.set_ylabel('ECE')
    sns.despine(ax=ax)
    
    print(f'Mean ECE = {mean_ece}')

    if save == True:
        f.savefig('calibration_variable_observations.png', dpi=300, bbox_inches='tight')

    
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

def compute_eces_variable(evidence_net, summary_net, simulator, n_val_per_setting, n_clust_min, n_clust_max, 
                          n_obs_min, n_obs_max, n_cal_bins=15, add_accuracy=False):
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
    if add_accuracy:
        accuracies = []
    
    with tqdm(total=(n_clust_max+1 - n_clust_min), desc='Loop through clusters progress') as p_bar: 
        with tqdm(total=(n_obs_max+1 - n_obs_min), desc='Loop through nested observations progress') as p_bar_within:
            for l in range(n_clust_min, n_clust_max+1): # Loop through clusters
                
                p_bar_within.reset((n_obs_max+1 - n_obs_min)) # reuse 2nd bar so that screen doesn't explode
                for n in range(n_obs_min, n_obs_max+1): # Loop through nested observations

                    # Simulate validation data
                    m_val_sim, _, x_val_sim = simulator(n_val_per_setting, n_clust_obs_f_v_val(l, n))

                    # Predict model probabilities
                    m_soft = tf.concat([evidence_net.predict(summary_net(x_chunk))['m_probs'][:, 1] for x_chunk in tf.split(x_val_sim, 20)], axis=0).numpy()      
                    m_hard = (m_soft > 0.5).astype(np.int32)
                    m_true = m_val_sim[:, 1]  

                    # Compute calibration error
                    prob_true, prob_pred = calibration_curve(m_true, m_soft, n_bins=n_cal_bins)
                    cal_err = np.abs(prob_true - prob_pred)

                    mean_ece = np.mean(cal_err)
                    ece_means.append(mean_ece)

                    if add_accuracy:
                        accuracy = accuracy_score(m_true, m_hard)
                        accuracies.append(accuracy)

                    # Update inner progress bar
                    p_bar_within.set_postfix_str("Cluster {0}, Observation {1}".format(l, n + 1))
                    p_bar_within.update()

                # Refresh inner + update outer progress bar
                p_bar_within.refresh() 
                p_bar.set_postfix_str("Finished clusters: {}".format(l))
                p_bar.update()
    
    if add_accuracy:
        return ece_means, accuracies

    return ece_means


def plot_eces_variable(ece_means, n_clust_min, n_clust_max, n_obs_min, n_obs_max, save=False, zlims=[0, 0.3], zlabel='ECE'):
    """ 
    Takes the ECE results from compute_eces_variable() and 
    projects them onto a 3D-plot.
    """
    
    # Prepare objects
    f = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')

    n_clust_points = np.arange(n_clust_min, n_clust_max+1)
    n_obs_points = np.arange(n_obs_min, n_obs_max+1)

    n_clust_grid, n_obs_grid = np.meshgrid(n_clust_points, n_obs_points)
    cal_err_grid = np.reshape(ece_means, (-1, n_clust_max)) # reshape into (#clusters, #observations)
    ax.plot_surface(n_clust_grid, n_obs_grid, cal_err_grid, cmap='viridis', edgecolor='none')

    ax.elev = 15
    ax.set_xlim([n_clust_min, n_clust_max]) 
    ax.set_ylim([n_obs_max, n_obs_min])
    ax.set_zlim(zlims)
    ax.set_xlabel('Number of groups ($M$)')
    ax.yaxis.set_rotate_label(False) # disable automatic label rotation
    ax.set_ylabel('Number of observations ($N$)', rotation=35) # not parallel to axis with automatic rotation    
    ax.set_zlabel(zlabel)

    print(f'Mean ECE = {np.mean(ece_means)}')
    #ax.text2D(0.05, 0.95, 'Mean ECE = {0:.3f}'.format(np.mean(ece_means)), transform=ax.transAxes)
    #ax.text2D(0.05, 0.90, 'SD around mean ECE = {0:.3f}'.format(np.std(ece_means)), transform=ax.transAxes)
    #ax.set_title('Expected Calibration Error (ECE)')

    if save == True:
        f.savefig('calibration_variable_sizes.png', dpi=300, bbox_inches='tight')
    

def plot_ece_means(ece_means, n_clust_min, n_clust_max, n_obs_min, n_obs_max, x_axis):
    """Helper function to plot ece means over clusters (1 = inspect observations) or observations (0 = inspect clusters)."""

    if x_axis == 0:
        xlabel='$M$'
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
    ax.grid(alpha=.3)


# Bridge sampling comparison: Plot approximations

def plot_approximations(bridge_sampling_results, NN_results, approximated_outcome, NN_name, save=False, ax=None):
    """Plots and contrasts the approximations generated by bridge sampling and a neural network.

    Parameters
    ----------
    bridge_sampling_results : pd.DataFrame
        Bridge sampling approximations.
    NN_results : pd.DataFrame
        Neural network approximations.
    approximated_outcome : string
        Indicates whether posterior model probabilities ('PMPs') or Log Bayes factors ('Log BFs') shall be plotted.
    NN_name : string
        Indicates the type of neural network (fixed/variable).
    """

    if not ax:
        f, ax = plt.subplots(1, 1, figsize=plotting_settings['figsize'])

    if approximated_outcome == 'PMPs': # PMPs
        bridge_data = bridge_sampling_results['m1_prob']
        NN_data = NN_results['m1_prob']
        label_outcome = r'$p(\mathcal{M}_2\,|\,\it{x})$'
        
    elif approximated_outcome == 'Log BFs': # Log BFs
        bridge_data = log_with_inf_noise_addition(bridge_sampling_results)
        NN_data = log_with_inf_noise_addition(NN_results)
        label_outcome = r'$\log \/ \mathrm{BF}_{21}$' #'$Log\/BF_{21}$'

    colors = {0:plotting_settings['colors_discrete'][3], 1:plotting_settings['colors_discrete'][0]}

    ax.scatter(bridge_data, NN_data, c=bridge_sampling_results['true_model'].map(colors), alpha=.8)
    helperlist = [plt.plot([], marker="o", ls="", color=color, alpha=.8)[0] for color in colors.values()] # hack for legend
    ax.legend(helperlist, [r'Simulated from $\mathcal{M}_1$', 'Simulated from $\mathcal{M}_2$'], loc='upper left', fontsize=12)
    ax.plot(ax.get_xlim(), ax.get_xlim(), '--', color='darkgrey')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(label_outcome + ' - ' + 'bridge sampling', fontsize=plotting_settings['fontsize_labels'])
    ax.set_ylabel(label_outcome + ' - ' + NN_name, fontsize=plotting_settings['fontsize_labels'])
    ax.grid(alpha=.3)
    
    if save == True:
        if approximated_outcome == 'PMPs': 
            plt.savefig('BS_vs_NN_PMPs_.png', dpi=300, bbox_inches='tight')
        if approximated_outcome == 'Log BFs': 
            plt.savefig('BS_vs_NN_BFs_.png', dpi=300, bbox_inches='tight')


# Bridge sampling comparison: Plot computation times

def plot_computation_times(bridge_time, NN_fixed_time, NN_variable_time, save=False):
    """ Plots computation times for bridge sampling and the neural networks."""

    f, ax = plt.subplots(1, 1, figsize=plotting_settings['figsize'])

    ax.plot(bridge_time, label='Bridge sampling', lw=2, color=plotting_settings['colors_discrete'][0])
    ax.plot(NN_fixed_time, label='Fixed network', lw=2, linestyle='dotted', color=plotting_settings['colors_discrete'][1])
    ax.plot(NN_variable_time, label='Variable network', lw=2, linestyle='dashed', color=plotting_settings['colors_discrete'][2])
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Data set', fontsize=plotting_settings['fontsize_labels']) # add (cumulative)? 
    ax.set_ylabel('Computation time in minutes', fontsize=plotting_settings['fontsize_labels'])
    ax.set_xlim(xmin=1)
    ax.set_ylim(ymin=0)
    ax.legend(loc='upper left', fontsize=12)

    # Add a grid for every 10 minutes / datasets
    minor_xticks = np.arange(10, 101, 10)
    minor_yticks = np.arange(0, 121, 10)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.grid(which='both', alpha=.3)

    if save:
        plt.savefig('bs_comparison_comp_times.png', dpi=300, bbox_inches='tight')



# Lévy flight application: Plot validation results

def plot_validation_results(true_models, preds, pm_samples, narrow_ci, wide_ci, labels, save=False):
    """ Plots all validation results in one plot. """

    true_models_flat = tf.argmax(true_models, axis=1)
    preds_flat = tf.argmax(preds, axis=1)

    # Create subfigures
    fig = plt.figure(constrained_layout=True, figsize=(10, 5))
    subfigs = fig.subfigures(nrows=1, ncols=2, wspace=0.07, hspace=0.07)
    
    # Confusion matrix
    ax_0 = subfigs[0].subplots(nrows=1, ncols=1)
    plot_confusion_matrix(true_models_flat, preds_flat, labels, ax=ax_0, title=False)

    # Calibration curves
    ax_1 = subfigs[1].subplots(nrows=2, ncols=2)
    pos1 = [0,0,1,1]
    pos2 = [0,1,0,1]
    xlabels = [False, False, True, True]
    ylabels = [True, False, True, False]

    for m in range(true_models.shape[1]):
        m_true = true_models[:, m]
        pm_samples_model = pm_samples[:, :, m]
        plot_calibration_curve_uncertainty(m_true, pm_samples_model, narrow_ci=narrow_ci, wide_ci=wide_ci, 
                                           n_bins=15, ax=ax_1[pos1[m], pos2[m]], xlabel=xlabels[m], 
                                           ylabel=ylabels[m], title=labels[m], show_legend=False)
    
    if save:
        fig.savefig('levy_validation.png', dpi=300, bbox_inches='tight')


# Lévy flight application: Visualize PMPs

def plot_model_posteriors(dirichlet_samples, labels, title=None, save=False, ax=None):
    """ Plots the model posteriors."""

    # Prepare dirichlet samples
    dirichlet_samples = np.squeeze(dirichlet_samples)

    # Plot
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=plotting_settings['figsize'])

    violin_plot = ax.violinplot(dirichlet_samples, showmedians=True)
    colors = plotting_settings['colors_discrete']

    for c, vp in zip(colors, violin_plot['bodies']):
        vp.set_facecolor(c)
        vp.set_edgecolor('black')
        vp.set_alpha(plotting_settings['alpha'])

    # Make all the violin statistics marks black:
    for partname in ['cbars','cmins','cmaxes','cmedians']:
        violin_plot[partname].set_edgecolor('black')

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(labels, fontsize=plotting_settings['fontsize_labels'])
    ax.set_ylabel('Posterior model probability', fontsize=plotting_settings['fontsize_labels'])
    ax.set_ylim([0, 1])

    if title:
        ax.set_title(title, fontsize=plotting_settings['fontsize_title'])

    if save:
        plt.savefig('levy_model_posteriors.png', dpi=300, bbox_inches='tight')


# Lévy flight application: Robustness against additional noise

def plot_noise_robustness(noise_proportions, mean_probs, mean_variabilities, labels, save=False):
    """ Plots the robustness of the model to additional noise (missing values).

    Parameters
    ----------
    mean_noise_proportion_list : list
        Mean noise proportions in the data sets for each noise step. 
    mean_probs : np.array
        Mean predictions output by the evidential network for each noise step.
    mean_variabilities : np.array
        Standard deviations of the mean predictions over the runs for each noise step.
    labels : list
        Model names to be displayed in the legend.
    save : bool, optional
        Save plot to current directory.
    """
    
    n_models = mean_probs.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=plotting_settings['figsize'])

    for m in range(n_models):
        ax.plot(noise_proportions, 
                mean_probs[:, m], 
                color=plotting_settings['colors_discrete'][m], 
                alpha=plotting_settings['alpha'])

    ax.legend(labels) # between loops to correctly display lines and not shaded area

    for m in range(n_models):
        ax.fill_between(noise_proportions, 
                        mean_probs[:, m] - mean_variabilities[:, m], 
                        mean_probs[:, m] + mean_variabilities[:, m], 
                        color=plotting_settings['colors_discrete'][m], 
                        alpha=0.2)
                    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Percentage of missing values', fontsize=plotting_settings['fontsize_labels'])
    ax.set_ylabel('Posterior model probability', fontsize=plotting_settings['fontsize_labels'])

    if save:
        fig.savefig('levy_noise_robustness.png', dpi=300, bbox_inches='tight')