#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""
import numpy as np
from utils_2D.init_location_utils import fix_sampling_2D
from utils_2D.control_repeat_utils import control_repeat_seed_L1

#%% load data
data_path = "2D/data/" 
data_seed = 42
noise_ratio = 1
seed = 42
option = 2

control_pre_name = f"control_pre_same_{data_seed}_{noise_ratio}_{seed}_{option}.npz"
variables = np.load(data_path + control_pre_name, allow_pickle=True)

ratio = 100 # noise levels: 1, 10, 100
Np = variables['Np']
Y_0 = variables['Y_0']
target = variables['target']

data_name = f"data_same_{seed}_{noise_ratio}.npz"
data = np.load(data_path + data_name, allow_pickle=True)

I1 = data['I1']
I2 = data['I2']
A = data['A']
b_B = data['b_B']
magnitude_B = data['magnitude_B']
noise_std = data['noise_std']

data_control_name = f"data_control_same_{data_seed}_{noise_ratio}_{seed}.npz"
data_control = np.load(data_path + data_control_name, allow_pickle=True)

log_sigma_1 = data_control['log_sigma_1']
log_sigma_2 = data_control['log_sigma_2']
base = data_control['base']
log_sigma_1_B = data_control['log_sigma_1_B']
log_sigma_2_B = data_control['log_sigma_2_B']
base_B = data_control['base_B']

#%% initial control locations
numControl = 6 # 3, 4, 5, 6
center_B = fix_sampling_2D(numControl, I1, I2)

#%% repeat experiments
seeds = list(range(10, 20))

RMSD_values = [control_repeat_seed_L1(seed, Np, ratio, noise_std, data_path, I1, I2, Y_0, numControl, center_B, log_sigma_1, log_sigma_2, base, log_sigma_1_B, log_sigma_2_B, base_B, A, b_B, magnitude_B, target) for seed in seeds]

RMSD_values = np.array(RMSD_values)

RMSD_seed = np.mean(RMSD_values)
mean_RMSD = np.mean(RMSD_seed)
var_RMSD = np.var(RMSD_values)

print(f'L1, Mean RMSD: {mean_RMSD:.3f}')
print(f'L1, Variance RMSD: {var_RMSD:.2e}')
