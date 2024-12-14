#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""
import torch
import scipy.io as sio

def split_prepare_data(seed, Y, T, est_ratio, train_ratio, val_ratio):
    
    torch.manual_seed(seed)

    est_size = int(T * est_ratio)

    Y_np = Y[:est_size]

    total_samples = Y_np.shape[0]
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    train_idx = train_size
    val_idx = train_size + val_size

    # Split the dataset
    Y_train_np, Y_val_np, Y_test_np = Y_np[:train_idx], Y_np[train_idx:val_idx], Y_np[val_idx:]

    Y_train = torch.tensor(Y_train_np, dtype=torch.float32)
    
    Y_val = torch.tensor(Y_val_np, dtype=torch.float32)

    Y_test = torch.tensor(Y_test_np, dtype=torch.float32)
    
    control_dict = {
    'Y_or': Y[est_size:]
    }
    
    return Y_train, Y_val, Y_test, control_dict
