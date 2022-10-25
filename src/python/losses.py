import os, sys
sys.path.append(os.path.abspath(os.path.join('../../..'))) # access sibling directories
sys.path.append("C:\\Users\\lasse\\Documents\\GitHub\\BayesFlow")

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy

from bayesflow.losses import kl_dirichlet




# Custom loss for multi-task learning

def multi_task_loss(network, model_indices, sim_data, kl_weight=0.01):
    """ Computes the combined logloss given alphas, softmax probs and true model indices m_true.
        Requires the evidential network to possess an evidential as well as a softmax output layer.
    Parameters
    ----------
    network       : tf.keras.Model
        An evidential network (with real outputs in ``[1, +inf]``)
    model_indices : tf.Tensor of shape (batch_size, n_models)
        True model indices
    sim_data      : tf.Tensor of shape (batch_size, n_obs, data_dim) or (batch_size, summary_dim) 
        Synthetic data sets generated by the params or summary statistics thereof
    kl_weight         : float in [0, 1]
        The weight of the KL regularization term
    Returns
    -------
    loss : tf.Tensor
        A single scalar Monte-Carlo approximation of the regularized Bayes risk, shape (,)
    """

    # Compute evidences and softmax probabilities
    alpha, softmax_probs = network(sim_data)

    # Process evidential output
    # Obtain probs
    model_probs = alpha / tf.reduce_sum(alpha, axis=1, keepdims=True)
    # Numerical stability
    model_probs = tf.clip_by_value(model_probs, 1e-15, 1 - 1e-15)
    # Compute evidential loss + regularization (if given)
    evidential_loss = -tf.reduce_mean(tf.reduce_sum(model_indices * tf.math.log(model_probs), axis=1))
    if kl_weight > 0:
        kl = kl_dirichlet(model_indices, alpha)
        evidential_loss = evidential_loss + kl_weight * kl

    # Process softmax output
    # Numerical stability
    softmax_probs = tf.clip_by_value(softmax_probs, 1e-15, 1 - 1e-15)
    # Compute softmax loss
    cat_cross_entropy = CategoricalCrossentropy()
    softmax_loss = cat_cross_entropy(model_indices, softmax_probs)

    # Combine losses 
    loss = evidential_loss + softmax_loss

    return loss