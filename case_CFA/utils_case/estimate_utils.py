#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils_case.kernel_utils import gaussian_product_kernel

def init_parameters_and_optimizer(log_sigma_1_init, log_sigma_2_init, base_init, lr):
    log_sigma_1 = torch.tensor(log_sigma_1_init, requires_grad=True)
    log_sigma_2 = torch.tensor(log_sigma_2_init, requires_grad=True)
    base = torch.tensor(base_init, requires_grad=True)
    
    optimizer = optim.Adam([log_sigma_1, log_sigma_2, base], lr=lr)

    return log_sigma_1, log_sigma_2, base, optimizer

def objective_function(Y, log_sigma_1, log_sigma_2, base, lambda1, lambda2):
    
    N, I1, I2 = Y.shape

    sigma_1 = torch.exp(log_sigma_1)
    sigma_2 = torch.exp(log_sigma_2)
        
    x = torch.arange(I1)
    y = torch.arange(I2)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    
    A = torch.zeros((I1,I2,I1,I2))
        
    for i1 in range(I1):
        for i2 in range(I2):
            A[:, :, i1, i2] = gaussian_product_kernel(xx, yy, i1, i2, sigma_1[i1,i2], sigma_2[i1,i2]).T * base
            
    diff = torch.zeros(N)
    
    for t in range(1, N):
        # AY
        Y_estimate = torch.sum(A * Y[t-1], axis=(2, 3))

        epsilon = 1e-8  # A small constant value
        diff[t] = torch.mean(torch.sum((Y[t] - Y_estimate) ** 2) / (torch.sum(Y[t] ** 2) + epsilon))

        
    loss1 = torch.sum(diff) / (N-1)
    loss2 = torch.norm(sigma_1 - 0.1, p=1)
    loss3 = torch.norm(sigma_2 - 0.1, p=1)
    loss = loss1 + lambda1 * loss2 + lambda2 * loss3
          
    return loss, loss1

def train_val(Y_train, Y_val, lr_init, epoch_init, step, lambda1, lambda2, tol, checkpoint_path, option="new"):
    
    _, I1, I2 = Y_train.shape
    
    log_sigma_1_init = np.maximum(0.0, np.round(np.random.uniform(np.log(I1/8.0), np.log(I1/6.0), size=(I1, I2)), 1))
    log_sigma_2_init = np.maximum(0.0, np.round(np.random.uniform(np.log(I2/8.0), np.log(I2/6.0), size=(I1, I2)), 1))
    base_init = np.full(1, 0.05)
     
    lr = lr_init
    epochs = epoch_init
    
    # Initialize parameters and optimizer
    (log_sigma_1, log_sigma_2, base, optimizer) = init_parameters_and_optimizer(log_sigma_1_init, log_sigma_2_init, base_init, lr)
                                                
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    
    if option == "retrain":
        checkpoint = torch.load(checkpoint_path)
        log_sigma_1 = checkpoint['log_sigma_1'].clone().detach().requires_grad_(True)
        log_sigma_2 = checkpoint['log_sigma_2'].clone().detach().requires_grad_(True)
        base = checkpoint['base'].clone().detach().requires_grad_(True)
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    for epoch in range(epochs):
   
        loss_train, loss_obj_train = objective_function(Y_train, log_sigma_1, log_sigma_2, base, lambda1, lambda2)
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        scheduler.step()
        
        loss_val, loss_obj_val = objective_function(Y_val, log_sigma_1, log_sigma_2, base, lambda1, lambda2)
        
        torch.save({
            'log_sigma_1': log_sigma_1.clone(),
            'log_sigma_2': log_sigma_2.clone(),
            'base': base.clone(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, checkpoint_path)        
        
        print(f'Epoch [{epoch+1}/{epochs}], Train: {loss_obj_train.item():.4f}, Val: {loss_obj_val.item():.4f}')
        
        if loss_obj_train.item() < tol and loss_obj_val.item() < tol:
            break

    results = {'log_sigma_1': log_sigma_1, 'log_sigma_2': log_sigma_2, 'base': base}
        
    return results

def predict(Y, sigma_1, sigma_2, base):
    N, I1, I2 = Y.shape
    
    x = torch.arange(I1)
    y = torch.arange(I2)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    
    A = torch.zeros((I1,I2,I1,I2))
        
    for i1 in range(I1):
        for i2 in range(I2):
            A[:, :, i1, i2] = gaussian_product_kernel(xx, yy, i1, i2, sigma_1[i1,i2], sigma_2[i1,i2]).T * base
            
    pred = torch.zeros(N, I1, I2)
    
    for t in range(1, N):
        # AY
        Y_estimate = torch.sum(A * Y[t-1], axis=(2, 3))
        
        pred[t] = Y_estimate
    
    return pred[1:]
