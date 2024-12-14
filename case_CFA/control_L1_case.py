#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""
import numpy as np
from utils_case.init_location_utils import fixed_sampling_2D
from utils_case.control_repeat_utils import control_repeat_seed_L1

#%% load data
seed = 42
option = 2
data_path = "case/data/" 

control_pre_name = f"control_pre_same_{seed}_{option}.npz"
variables = np.load(data_path + control_pre_name, allow_pickle=True)

Y_0 = variables['Y_0']
target = variables['target']

data_name = f"data_same_{seed}.npz"
data = np.load(data_path + data_name, allow_pickle=True)

I1 = data['I1']
I2 = data['I2']
b_B = data['b_B']
magnitude_B = data['magnitude_B']

data_control_name = f"data_control_same_{seed}.npz"
data_control = np.load(data_path + data_control_name, allow_pickle=True)

log_sigma_1 = data_control['log_sigma_1']
log_sigma_2 = data_control['log_sigma_2']
base = data_control['base']

#%% initial control locations
numControl = 3 # 3, 4, 5, 6
center_B = fixed_sampling_2D(numControl, I1, I2)

#%% repeat experiments
Np = 100
noise_or = np.random.uniform(-0.001, 0.001, (Np, I1, I2))

seeds = list(range(10))

# Conduct experiments and collect RMSD values
RMSD_values = [control_repeat_seed_L1(seed, data_path, I1, I2, Y_0, noise_or, numControl, center_B, log_sigma_1, log_sigma_2, base, b_B, magnitude_B, target, Np) for seed in seeds]

RMSD_values = np.array(RMSD_values)

RMSD_seed = np.mean(RMSD_values, axis=1)
mean_RMSD = np.mean(RMSD_seed)
var_RMSD = np.var(RMSD_values)

print(f'case L1, Mean RMSD: {mean_RMSD:.3f}')
print(f'case L1, Variance RMSD: {var_RMSD:.2e}')
