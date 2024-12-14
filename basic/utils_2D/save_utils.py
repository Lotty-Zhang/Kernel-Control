#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""

import numpy as np

def load_before_control(data_path, data_name, results_name, control_pre_name):
    
    loaded_data = np.load(data_path+data_name, allow_pickle=True)
    I1 = loaded_data['I1']
    I2 = loaded_data['I2']
    numControl = loaded_data['numControl']
    numPart = loaded_data['numPart']
    A = loaded_data['A']
    bX_B = loaded_data['bX_B']
    bY_B = loaded_data['bY_B']
    magnitude_B = loaded_data['magnitude_B']
    
    loaded_results = np.load(data_path+results_name, allow_pickle=True)
    log_sigma_1 = loaded_results['log_sigma_1']
    log_sigma_2 = loaded_results['log_sigma_2']
    base = loaded_results['base']
    log_sigma_1_B = loaded_results['log_sigma_1_B']
    log_sigma_2_B = loaded_results['log_sigma_2_B']
    base_B = loaded_results['base_B']
    
    loaded_control_pre = np.load(data_path+control_pre_name, allow_pickle=True)
    noise_or = loaded_control_pre['noise_or']
    Y_0 = loaded_control_pre['Y_0']
    target = loaded_control_pre['target']
    
    variables = {
        'I1': I1,
        'I2': I2,
        'numControl': numControl,
        'numPart': numPart,
        'A': A,
        'bX_B': bX_B,
        'bY_B': bY_B,
        'magnitude_B': magnitude_B,
        'noise_or': noise_or,
        'Y_0': Y_0,
        'target': target
    }
    
    results = {'log_sigma_1': log_sigma_1, 'log_sigma_2': log_sigma_2, 'base': base}
    results_B = {'log_sigma_1_B': log_sigma_1_B, 'log_sigma_2_B': log_sigma_2_B, 'base_B': base_B}
    
    return variables, results, results_B


def save_after_control(data_path, I1, I2, numControl, Y_pred, Y_phys_wc, Y_phys_nc, X_pred, center_B_save, target, seed, ratio, method):
    
    filename = f"data_eval_same_{seed}_{ratio}_{numControl}_{method}.npz"
    np.savez(data_path + '/' + filename,
             Y_pred=Y_pred, Y_phys_wc=Y_phys_wc, Y_phys_nc=Y_phys_nc, 
             X_pred=X_pred, center_B_save=center_B_save, target=target)
