#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""
import os
import numpy as np
import torch
from utils_case.estimate_pre_utils import split_prepare_data
from utils_case.estimate_utils import train_val, predict

#%% load data
data_path = "case/data" 
seed = 42
data = np.load(data_path+ f'/data_same_{seed}.npz', allow_pickle=True)

#%% prepare data
est_ratio = 0.6
train_ratio = 0.7
val_ratio = 0.2

(Y_train, Y_val, Y_test, control_dict) = split_prepare_data(seed, data['Y'], data['T'], est_ratio, train_ratio, val_ratio)

#%% Train and Validation
lr_init = 0.1
epoch_init = 200
step = 200
lambda1 = 0
lambda2 = 0
tol = 5e-4
ckp_path = 'case/checkpoints/'
ckp_file = "estimate_checkpoint.pth"
checkpoint_path = os.path.join(ckp_path, ckp_file)
results = train_val(Y_train, Y_val, lr_init, epoch_init, step, lambda1, lambda2, tol, checkpoint_path, option="new")

#%% Test and Prediction
sigma_1 = torch.exp(results['log_sigma_1'])
sigma_2 = torch.exp(results['log_sigma_2'])
base = results['base']

Y_test_pred = predict(Y_test, sigma_1, sigma_2, base)

mse_test = []
mse_test = torch.mean( torch.sum((Y_test[1:] - Y_test_pred) ** 2) / torch.sum((Y_test[1:]) ** 2) )
print(f"Test Mean Squared Error: {mse_test.item():.4f}")

#%% save data for control pre
filename = f"control_pre_same_{seed}.npz"
np.savez(data_path + '/' + filename, Y_or=control_dict['Y_or'])

#%% save data for estimation results
log_sigma_1 = results['log_sigma_1'].detach().numpy()
log_sigma_2 = results['log_sigma_2'].detach().numpy()
base = results['base'].detach().numpy()

filename = f"data_control_same_{seed}.npz"
np.savez(data_path + '/' + filename, 
         log_sigma_1=log_sigma_1, log_sigma_2=log_sigma_2, base=base)

