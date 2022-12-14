import os, sys

# When on cloud (RStudio server)
sys.path.append(os.path.abspath(os.path.join('../../..'))) # access sibling directories; as in .ipynb

# When on desktop (requires different paths than cloud)
#sys.path.append(os.path.abspath(os.path.join('../Coding'))) # access sibling directories; different than .ipynb
#sys.path.append("C:\\Users\\lasse\\Documents\\GitHub\\BayesFlow")

from src.python.settings import summary_meta_validation, probability_meta_validation
from src.python.networks import HierarchicalInvariantNetwork, ModelProbabilityNetwork
from src.python.models import HierarchicalNormalSimulator, MainSimulator
from src.python.losses import softmax_loss
from src.python.helpers import n_clust_obs_v_v, get_preds_and_bfs
from src.python.visualization import compute_eces_variable, plot_eces_variable, plot_eces_marginalized

import numpy as np
import pandas as pd
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.optimizers import Adam
from functools import partial
from time import perf_counter
from datetime import datetime, timezone, timedelta

from bayesflow.trainers import ModelComparisonTrainer
from bayesflow.amortizers import MultiModelAmortizer



summary_net = HierarchicalInvariantNetwork(summary_meta_validation)
probability_net = ModelProbabilityNetwork(probability_meta_validation)

amortizer = MultiModelAmortizer(probability_net, summary_net)

simulator = MainSimulator(HierarchicalNormalSimulator())



# Training steps
epochs = 40 
iterations_per_epoch = 1000

# Range of groups and nested observations
n_clust_min = 1
n_clust_max = 100
n_obs_min = 1
n_obs_max = 100

# Cosine decaying learning rate
initial_lr = 0.0005
decay_steps = epochs*iterations_per_epoch
alpha = 0
lr_schedule = CosineDecay(
    initial_lr, decay_steps, alpha=alpha
    )

# Checkpoint path for loading pretrained network and saving the final network
checkpoint_path = 'C:\\Users\\lasse\\Documents\\hierarchical_model_comparison_project\\checkpoints\\01_calibration_validation\\checkpoints_var_sizes'

trainer = ModelComparisonTrainer(
    network=amortizer, 
    generative_model=simulator, 
    loss=partial(softmax_loss),
    optimizer=partial(Adam, lr_schedule),
    checkpoint_path=checkpoint_path,
    skip_checks=True,
    )



# Validation
n_val_per_setting = 5000
get_accuracies = True # Compute accuracies additionally to ECEs?

if not get_accuracies:
    eces = compute_eces_variable(
    probability_net, summary_net, simulator, 
    n_val_per_setting, n_clust_min, n_clust_max, 
    n_obs_min, n_obs_max
    )

if get_accuracies:
    eces, accuracies = compute_eces_variable(
        probability_net, summary_net, simulator, 
        n_val_per_setting, n_clust_min, n_clust_max, 
        n_obs_min, n_obs_max, add_accuracy=True
        )



# Export ECE data
local_timezone = datetime.now(timezone(timedelta(0))).astimezone().tzinfo
filename = pd.Timestamp.today(tz=local_timezone).strftime('%Y_%m_%d_eces_var_sizes')
val_folder = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'data', '01_calibration_validation', 'eces')
np.save(os.path.join(val_folder, filename), eces)