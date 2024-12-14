#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""

import numpy as np
from utils_2D.control_L1_utils import onlineControl_dymL1_B
from utils_2D.save_utils import save_after_control

def control_repeat_seed_L1(seed, Np, ratio, noise_std, data_path, I1, I2, Y_0, numControl, center_B, log_sigma_1, log_sigma_2, base, log_sigma_1_B, log_sigma_2_B, base_B, A, b_B, magnitude_B, target):
    
    np.random.seed(seed)

    Y_pred, Y_phys_wc, Y_phys_nc, X_pred, center_B_save = onlineControl_dymL1_B(Y_0, Np, ratio, noise_std, numControl, 
                                                      center_B, log_sigma_1, log_sigma_2, base, log_sigma_1_B, log_sigma_2_B, base_B, A, b_B, 
                                                      magnitude_B, target, seed)

    save_after_control(data_path, I1, I2, numControl, Y_pred, Y_phys_wc, Y_phys_nc, X_pred, center_B_save, target, seed, ratio, method='L1')

    numerator_sum = 0
    denominator_sum = 0
    
    for t in range(30, Np):
        numerator_sum += np.linalg.norm(Y_phys_wc[t] - target, ord='fro')**2
        denominator_sum += np.linalg.norm(Y_phys_nc[t] - target, ord='fro')**2
    
    RMSD = numerator_sum / denominator_sum
  
    return RMSD
