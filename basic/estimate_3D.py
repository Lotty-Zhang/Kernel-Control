#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""
import os
import numpy as np
import torch
from utils_3D.estimate_pre_utils import split_prepare_data
from utils_3D.estimate_utils import train_val, predict

#%% load data
data_path = "3D/data" 
data_seed = 42
noise_ratio = 1
data = np.load(data_path+ f'/data_same_{data_seed}_{noise_ratio}.npz', allow_pickle=True)

numControl = data['numControl']

#%% prepare data
est_ratio = 0.9
train_ratio = 0.6
val_ratio = 0.3

(Y_train, X_train, Y_val, X_val, Y_test, X_test, 
 center_B_train, center_B_val, center_B_test, 
 control_dict) = split_prepare_data(data_seed, noise_ratio, data['Y'], data['X'], data['center_B'], 
                                    data['noise'], data['T'], est_ratio, train_ratio, val_ratio)

#%% Train and Validation
lr_init = 0.0003
epoch_init = 900
step = 600
lambda1 = 0
lambda2 = 0
lambda3 = 0
tol = 5e-4

ckp_path = '3D/checkpoints/'

seed_values = [42, 43, 44, 45, 46]
mse_test_values = []

for seed in seed_values:
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    ckp_file = f'estimate_checkpoint_{data_seed}_{noise_ratio}_{seed}.pth'
    checkpoint_path = os.path.join(ckp_path, ckp_file)

    results, results_B = train_val(Y_train, X_train, Y_val, X_val, center_B_train, center_B_val, lr_init, epoch_init, step, lambda1, lambda2, lambda3, tol, checkpoint_path, option="new")
    
    ##########
    # Test and Prediction
    sigma_1 = torch.exp(results['log_sigma_1'])
    sigma_2 = torch.exp(results['log_sigma_2'])
    sigma_3 = torch.exp(results['log_sigma_3'])
    base = results['base']
    sigma_1_B = torch.exp(results_B['log_sigma_1_B'])
    sigma_2_B = torch.exp(results_B['log_sigma_2_B'])
    sigma_3_B = torch.exp(results_B['log_sigma_3_B'])
    base_B = results_B['base_B']
    
    # Make predictions for the test set
    Y_test_pred = predict(Y_test, X_test, sigma_1, sigma_2, sigma_3, base, center_B_test, sigma_1_B, sigma_2_B, sigma_3_B, base_B)
    
    mse_test = []
    mse_test = torch.mean( torch.sum((Y_test[1:] - Y_test_pred) ** 2) / torch.sum((Y_test[1:]) ** 2) )
    mse_test_values.append(mse_test.item())
    print(f"Test Mean Squared Error: {mse_test.item():.4f}")
    
    # save data for control pre
    filename = f"control_pre_same_{data_seed}_{noise_ratio}_{seed}.npz"

    np.savez(data_path + '/' + filename, 
             Y_or=control_dict['Y_or'], X_or=control_dict['X_or'], 
             noise_or=control_dict['noise_or'], center_B_or=control_dict['center_B_or'])
    
    ##########
    # save data for estimation results
    log_sigma_1 = results['log_sigma_1'].detach().numpy()
    log_sigma_2 = results['log_sigma_2'].detach().numpy()
    log_sigma_3 = results['log_sigma_3'].detach().numpy()
    base = results['base'].detach().numpy()
    
    log_sigma_1_B = results_B['log_sigma_1_B'].detach().numpy()
    log_sigma_2_B = results_B['log_sigma_2_B'].detach().numpy()
    log_sigma_3_B = results_B['log_sigma_3_B'].detach().numpy()
    base_B = results_B['base_B'].detach().numpy()
    
    filename = f"data_control_same_{data_seed}_{noise_ratio}_{seed}.npz"

    np.savez(data_path + '/' + filename, 
             log_sigma_1=log_sigma_1, log_sigma_2=log_sigma_2, log_sigma_3=log_sigma_3, base=base,
             log_sigma_1_B=log_sigma_1_B, log_sigma_2_B=log_sigma_2_B, log_sigma_3_B=log_sigma_3_B, base_B=base_B)
    
mean_mse_test = np.mean(mse_test_values)
variance_mse_test = np.var(mse_test_values)

print(f"Mean of Test MSE across all seeds: {mean_mse_test:.4e}")
print(f"Variance of Test MSE across all seeds: {variance_mse_test:.16e}")
