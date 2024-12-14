#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""
import torch

def split_prepare_data(seed, noise_ratio, Y, X, center_B, noise, T, est_ratio, train_ratio, val_ratio):
    
    torch.manual_seed(seed)  # For reproducibility

    est_size = int(T * est_ratio)

    Y_np = Y[:est_size]
    X_np = X[:est_size]
    center_B_np = center_B[:est_size]

    # Calculate the number of samples in each set
    total_samples = Y_np.shape[0]
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    # Calculate the indices for splitting the dataset
    train_idx = train_size
    val_idx = train_size + val_size

    # Split the dataset
    Y_train_np, Y_val_np, Y_test_np = Y_np[:train_idx], Y_np[train_idx:val_idx], Y_np[val_idx:]
    X_train_np, X_val_np, X_test_np = X_np[:train_idx], X_np[train_idx:val_idx], X_np[val_idx:]
    center_B_train, center_B_val, center_B_test = center_B_np[:train_idx], center_B_np[train_idx:val_idx], center_B_np[val_idx:]

    # Convert the NumPy arrays to PyTorch tensors
    Y_train = torch.tensor(Y_train_np, dtype=torch.float32)
    X_train = torch.tensor(X_train_np, dtype=torch.float32)

    Y_val = torch.tensor(Y_val_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)

    Y_test = torch.tensor(Y_test_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
        
    # save data for future control
    control_dict = {
    'Y_or': Y[est_size:],
    'X_or': X[est_size:],
    'noise_or': noise[est_size:],
    'center_B_or': center_B[est_size:]
    }
    
    return Y_train, X_train, Y_val, X_val, Y_test, X_test, center_B_train, center_B_val, center_B_test, control_dict
