#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""

import numpy as np
from utils_2D.init_location_utils import fix_sampling_2D
from utils_2D.control_L1_utils import onlineControl_dymL1_B_alpha

#%% load data
data_path = "2D/data/" 
data_seed = 42
noise_ratio = 1
seed = 42
option = 2

control_pre_name = f"control_pre_same_{data_seed}_{noise_ratio}_{seed}_{option}.npz"
variables = np.load(data_path + control_pre_name, allow_pickle=True)

ratio = 1
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
numControl = 3
center_B = fix_sampling_2D(numControl, I1, I2)

#%% t = 1
iteration_info = onlineControl_dymL1_B_alpha(Y_0, Np, ratio, noise_std, numControl, center_B, log_sigma_1, log_sigma_2, base, log_sigma_1_B, log_sigma_2_B, base_B, A, b_B, magnitude_B, target, 0)
