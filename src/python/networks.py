import numpy as np
np.set_printoptions(suppress=True)
import tensorflow as tf
from tensorflow.keras.layers import Dropout



class InvariantModule(tf.keras.Model):
    """Implements an invariant nn module as proposed by Bloem-Reddy and Teh (2019)."""

    def __init__(self, meta, pooler=tf.reduce_mean):
        """
        Creates an invariant function with mean pooling.
        ----------
        
        Arguments:
        model_settings : dict -- hyperparameter settings for the invariant module
        """
        
        super(InvariantModule, self).__init__()
        
        # Pre pooling network
        self.pre_pooling_dense = tf.keras.Sequential([tf.keras.layers.Dense(**meta['dense_inv_pre_pooling_args'])
                                                     for _ in range(meta['n_dense_inv'])])
        
        self.pooler = pooler
        
        # Post pooling network
        self.post_pooling_dense = tf.keras.Sequential([tf.keras.layers.Dense(**meta['dense_inv_post_pooling_args'])
                                                      for _ in range(meta['n_dense_inv'])])
        
            
    def call(self, x):
        """
        Transforms the input into an invariant representation.
        ----------
        
        Arguments:
        x : tf.Tensor of variable shape - 4-dimensional with (batch_size, n_clusters, n_obs, n_variables)
                                          or 3-dimensional with (batch_size, n_clusters, inv_embedding)
        --------
        
        Returns:
        out: tf.Tensor of variable shape - either 3-dimensional (when input is 4D) with (batch_size, n_clusters, inv_embedding)
                                           or 2-dimensional (when input is 3D) with (batch_size, inv_embedding)
        """
        
        # Embed input before pooling
        x_emb = self.pre_pooling_dense(x)
        
        # Perform mean pooling, shape of pooled is (batch_size, K, dense_out)
        pooled = self.pooler(x_emb, axis=-2) # always reduce dimensionality of the lowest exchangable data level

        # Increase representational power
        out = self.post_pooling_dense(pooled)
        return out
    
    
class EquivariantModule(tf.keras.Model):
    
    def __init__(self, meta):
        """
        Creates an equivariant function.
        ----------
        
        Arguments:
        model_settings : dict -- hyperparameter settings for the equivariant module
        """
        
        super(EquivariantModule, self).__init__()
        
        self.inv = InvariantModule(meta['inv_inner'])
        self.equiv = tf.keras.Sequential([
            tf.keras.layers.Dense(**meta['dense_equiv_args'])
            for _ in range(meta['n_dense_equiv'])]
        )
        
    def call(self, x):
        """
        Performs an equivariant transformation using both equiv and inv transforms.
        """

        n_rep = x.shape[-2]
        inv = self.inv(x)
        inv = tf.stack([inv] * n_rep, axis=-2)
        inp = tf.concat([x, inv], axis=-1)
        out = self.equiv(inp)
        return out
    
    
class InvariantNetwork(tf.keras.Model):
    
    def __init__(self, meta, pooler=tf.reduce_mean):
        """
        Creates an invariant network consisting of equivariant and invariant functions.
        ----------
        
        Arguments:
        model_settings : dict -- hyperparameter settings for the invariant network
        """
        
        super(InvariantNetwork, self).__init__()
        
        self.inv = InvariantModule(meta['inv_outer'])
        self.equiv = tf.keras.Sequential([
                EquivariantModule(meta)
            for _ in range(meta['n_equiv'])
        ])
        
    def call(self, x):
        """
        Performs an invariant transformation using both equiv and inv transforms.
        """

        out = self.equiv(x)
        out = self.inv(out)
        return out
    
    
class HierarchicalInvariantNetwork(tf.keras.Model):
    """
    Implements a network that is able to summarize hierarchical data.
    """
    
    def __init__(self, meta):
        """
        Creates a hierarchical Network consisting of two stacked invariant modules.
        ----------
        
        Arguments:
        model_settings : dict -- hyperparameter settings for the hierarchical network
        """
        
        super(HierarchicalInvariantNetwork, self).__init__()
        
        self.inv_4d = InvariantNetwork(meta['level_1'])
        self.inv_3d = InvariantNetwork(meta['level_2'])
        
        
    def call(self, x):
        """
        Transforms the 4-dimensional input into learned summary statistics.
        ----------
        
        Arguments:
        x : tf.Tensor of shape (batch_size, n_clusters, n_obs, n_variables) -- simulated data
        --------
        
        Returns:
        out : tf.Tensor of shape (batch_size, (dense_inv_post_pooling_args on level_2) + 2) -- the learned summary statistics
        """

        # Pass through invariant networks
        out = self.inv_4d(x)
        out = self.inv_3d(out)

        # Extract number of clusters and observations into a repeated vector and concatenate with network output
        n_clust = int(x.shape[-3])
        n_clust_rep = tf.math.sqrt(n_clust * tf.ones((x.shape[0], 1)))
        out = tf.concat((out, n_clust_rep), axis=-1)

        n_obs = int(x.shape[-2])
        n_obs_rep = tf.math.sqrt(n_obs * tf.ones((x.shape[0], 1)))
        out = tf.concat((out, n_obs_rep), axis=-1)

        return out
    
    
class EvidentialNetwork(tf.keras.Model):
    """
    Implements a network that infers the parameters of a dirichlet distribution in order to quantify model evidences.
    """
    
    def __init__(self, meta):
        """
        Creates an evidential network.
        ----------
        
        Arguments:
        model_settings : dict -- hyperparameter settings for the evidential network
        """
        
        super(EvidentialNetwork, self).__init__()
        
        #self.dense = tf.keras.Sequential([
        #    tf.keras.layers.Dense(**meta['dense_args'])
        #    for _ in range(meta['n_dense'])
        #])

        self.dense = tf.keras.models.Sequential()

        for _ in range(meta['n_dense']):
            self.dense.add(tf.keras.layers.Dense(**meta['dense_args']))
            self.dense.add(MCDropout(dropout_prob=0.1))

        #self.evidence_layer = tf.keras.layers.Dense(meta['n_models'], activation=meta['activation_out'])

        if meta.get('multi_task_softmax') is not None:
            self.softmax_layer = tf.keras.layers.Dense(meta['n_models'], activation='softmax')
        else:
            self.softmax_layer = None

        self.J = meta['n_models']
        

    def call(self, x):
        """
        Forward pass through the evidential network.
        """
        
        out = self.dense(x)
        #alpha = self.evidence_layer(out) + 1
        if self.softmax_layer is not None: 
            probs = self.softmax_layer(out)
            #return alpha, probs
            return probs

        #return  alpha
        

    def predict(self, obs_data, output_softmax=False, to_numpy=True):
        """
        Returns the mean, variance and uncertainty implied by the estimated Dirichlet density.

        Parameters
        ----------
        obs_data: tf.Tensor
            Observed data or respective embeddings created by a summary network
        output_softmax: bool, default: False
            Flag that controls whether softmax instead of evidential probabilities are returned     
        to_numpy: bool, default: True
            Flag that controls whether the output is a np.array or tf.Tensor

        Returns
        -------
        out: dict
            Dictionary with keys {m_probs, m_var, uncertainty}
        """

        if self.softmax_layer is not None: 
            #alpha, probs = self(obs_data) # adapt to output shape of call(); only use probs if output_softmax=True
            probs = self(obs_data)
            if output_softmax == True: 
                if to_numpy:
                    probs = probs.numpy()
                return {'m_probs': probs}

        else:
            alpha = self(obs_data)

        alpha0 = tf.reduce_sum(alpha, axis=1, keepdims=True)
        mean = alpha / alpha0
        var = alpha * (alpha0 - alpha) / (alpha0 * alpha0 * (alpha0 + 1))
        uncertainty = self.J / alpha0

        if to_numpy:
            mean = mean.numpy()
            var = var.numpy()
            uncertainty = uncertainty.numpy()

        return {'m_probs': mean, 'm_var': var, 'uncertainty': uncertainty}
    

    def sample(self, obs_data, n_samples, to_numpy=True):
        """Samples posterior model probabilities from the second-order Dirichlet distro.

        Parameters
        ----------
        obs_data  : tf.Tensor
            The summary of the observed (or simulated) data, shape (n_datasets, summary_dim)
        n_samples : int
            Number of samples to obtain from the approximate posterior
        to_numpy  : bool, default: True
            Flag indicating whether to return the samples as a np.array or a tf.Tensor

        Returns
        -------
        pm_samples : tf.Tensor or np.array
            The posterior samples from the Dirichlet distribution, shape (n_samples, n_batch, n_models)
        """

        # Compute evidential values
        alpha = self(obs_data)
        n_datasets = alpha.shape[0]

        # Sample for each dataset
        pm_samples = np.stack([np.random.dirichlet(alpha[n, :], size=n_samples) for n in range(n_datasets)], axis=1)

        # Convert to tensor, if specified
        if not to_numpy:
             pm_samples = tf.convert_to_tensor(pm_samples, dtype=tf.float32)
        return pm_samples


class MCDropout(tf.keras.Model):
    """Implements Monte Carlo Dropout as a Bayesian approximation according to [1].

    Perhaps not the best approximation, but arguably the cheapest one out there!

    [1] Gal, Y., & Ghahramani, Z. (2016, June). Dropout as a bayesian approximation: 
    Representing model uncertainty in deep learning. 
    In international conference on machine learning (pp. 1050-1059). PMLR.
    """

    def __init__(self, dropout_prob=0.1, **kwargs):
        """Creates a custom instance of an MC Dropout layer. Will be used both
        during training and inference.
        
        Parameters
        ----------
        dropout_prob  : float, optional, default: 0.1
            The dropout rate to be passed to ``tf.nn.experimental.stateless_dropout()``.
            
        """
        super().__init__(**kwargs)
        self.drop = tf.keras.layers.Dropout(dropout_prob)

    def call(self, inputs):
        """Randomly sets elements of ``inputs`` to zero.

        Parameters
        ----------
        inputs : tf.Tensor
            Input of shape (batch_size, ...)
        
        Returns
        -------
        out    : tf.Tensor
            Output of shape (batch_size, ...), same as ``inputs``.

        """

        out = self.drop(inputs, training=True)
        return out