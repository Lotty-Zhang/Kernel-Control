#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""

import numpy as np
from utils_case.control_L1_utils import onlineControl_dymL1_B
from utils_case.save_utils import save_after_control

def control_repeat_seed_L1(seed, data_path, I1, I2, Y_0, noise_or, numControl, center_B, log_sigma_1, log_sigma_2, base, b_B, magnitude_B, target, Np):
    np.random.seed(seed)

    Y_pred, Y_phys_wc, Y_phys_nc, X_pred, center_B_save = onlineControl_dymL1_B(Y_0, noise_or, numControl, 
                                                          center_B, log_sigma_1, log_sigma_2, base, b_B, magnitude_B, target, Np)


    save_after_control(data_path, I1, I2, numControl, Y_pred, Y_phys_wc, Y_phys_nc, X_pred, center_B_save, target, seed, method='L1')

    N = noise_or.shape[0]
    RMSD = np.zeros(N-1)
    for t in range(1, N):
        RMSD[t-1] = np.mean( (Y_phys_wc[t] - target) ** 2 ) / np.mean( (Y_phys_nc[t] - target) ** 2 )
        
    return RMSD[45:]
