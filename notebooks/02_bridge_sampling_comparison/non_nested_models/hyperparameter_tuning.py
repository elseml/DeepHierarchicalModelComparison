import os, sys
sys.path.append(os.path.abspath(os.path.join('../Coding'))) # access sibling directories; DIFFERENT THAN IN IPYNB!
sys.path.append("C:\\Users\\lasse\\Documents\\GitHub\\BayesFlow")

from src.python.settings import summary_meta_validation, evidence_meta_validation
from src.python.networks import HierarchicalInvariantNetwork, EvidentialNetwork
from src.python.models import HierarchicalSdtMptSimulator, MainSimulator
from src.python.losses import multi_task_loss
from src.python.helpers import n_clust_obs_f_f, get_multiple_predictions, get_preds_and_bfs

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.experimental import CosineDecayRestarts
from tensorflow.keras.optimizers import Adam
from functools import partial

from bayesflow.trainers import ModelComparisonTrainer
from bayesflow.amortizers import MultiModelAmortizer
from bayesflow.losses import log_loss

# For testing: n_runs_per_setting = 1, epochs=2, iterations_per_epoch=2
# For running: n_runs_per_setting = 5, epochs=15, iterations_per_epoch=1000
#### Runs per setting combination 
n_runs_per_setting = 1

### Hyperparameter tuning params (cosine decay w/ restarts)
initial_lr_list = [0.00065, 0.0006, 0.00055, 0.0005]
m_mul_list = [0.95, 0.9, 0.85, 0.8, 0.75]
alpha_list = [0.2, 0.1, 0.0]

results = []

n_combinations = len(initial_lr_list)*len(m_mul_list)*len(alpha_list)*n_runs_per_setting

### prepare static components
# Learning rate
first_decay_steps = 1000
t_mul = 2

# Sample size
n_clusters = 75
n_obs = 50

# Storage path
file_path = os.path.join(os.getcwd(),'notebooks\\02_bridge_sampling_comparison\\non_nested_models\\hyperparameter_tuning_results')

### Run tuning
for initial_lr in initial_lr_list:
    for m_mul in m_mul_list:
        for alpha in alpha_list:
            for run in range(n_runs_per_setting):

                # Initialize from scratch for each run
                summary_net = HierarchicalInvariantNetwork(summary_meta_validation)
                evidence_net = EvidentialNetwork(evidence_meta_validation)

                amortizer = MultiModelAmortizer(evidence_net, summary_net)

                simulator = MainSimulator(HierarchicalSdtMptSimulator())

                lr_schedule_restart = CosineDecayRestarts(
                    initial_lr, first_decay_steps, t_mul=t_mul, m_mul=m_mul, alpha=alpha
                    )

                trainer = ModelComparisonTrainer(
                    network=amortizer, 
                    generative_model=simulator, 
                    loss=partial(multi_task_loss, kl_weight=0.25),
                    optimizer=partial(Adam, lr_schedule_restart),
                    skip_checks=True,
                    )

                # Train
                losses = trainer.train_online(
                    epochs=1, iterations_per_epoch=100, batch_size=32, 
                    n_obs=partial(n_clust_obs_f_f, n_clusters, n_obs), n_vars=2
                    )

                # Get running loss of final epoch
                final_loss = np.mean(list(losses.values())[-1])

                # Store results
                results.append({
                    'initial_lr': initial_lr,
                    'm_mul': m_mul,
                    'alpha': alpha,
                    'final_loss': final_loss
                })

                # Print progress
                if len(results) == n_combinations*0.25:
                    print('25% done')
                if len(results) == n_combinations*0.50:
                    print('50% done')
                if len(results) == n_combinations*0.75:
                    print('75% done')

# Save tuning results to csv
pd.DataFrame(results).to_csv(file_path)