#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils_2D.kernel_utils import gaussian_product_kernel

def init_parameters_and_optimizer(log_sigma_1_init, log_sigma_2_init, base_init, log_sigma_1_init_B, log_sigma_2_init_B, base_init_B, lr):
    log_sigma_1 = torch.tensor(log_sigma_1_init, requires_grad=True)
    log_sigma_2 = torch.tensor(log_sigma_2_init, requires_grad=True)
    base = torch.tensor(base_init, requires_grad=True)

    log_sigma_1_B = torch.tensor(log_sigma_1_init_B, requires_grad=True)
    log_sigma_2_B = torch.tensor(log_sigma_2_init_B, requires_grad=True)
    base_B = torch.tensor(base_init_B, requires_grad=False)

    optimizer = optim.Adam([log_sigma_1, log_sigma_2, base, log_sigma_1_B, log_sigma_2_B, base_B], lr=lr)

    return log_sigma_1, log_sigma_2, base, log_sigma_1_B, log_sigma_2_B, base_B, optimizer

def objective_function(Y, X, log_sigma_1, log_sigma_2, base, center_B, log_sigma_1_B, log_sigma_2_B, base_B, lambda1, lambda2):
    
    N, I1, I2 = Y.shape
    N, numControl = X.shape
    
    sigma_1 = torch.exp(log_sigma_1)
    sigma_2 = torch.exp(log_sigma_2)
    sigma_1_B = torch.exp(log_sigma_1_B)
    sigma_2_B = torch.exp(log_sigma_2_B) 
        
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
        # BX
        control = torch.zeros((I1,I2))
        for act in range(numControl):
            cX_B, cY_B = center_B[t, act]
            B_tmp = gaussian_product_kernel(xx, yy, cX_B, cY_B, sigma_1_B, sigma_2_B).T * base_B
            control += X[t,act] * B_tmp 
        
        diff[t] = torch.mean( torch.sum((Y[t] - Y_estimate - control) ** 2) / torch.sum((Y[t]) ** 2) )  
        
    loss1 = torch.sum(diff) / (N-1)
    loss2 = torch.norm(sigma_1 - 1, p=1)
    loss3 = torch.norm(sigma_2 - 1, p=1)
    loss = loss1 + lambda1 * loss2 + lambda2 * loss3
              
    return loss, loss1

def train_val(seed, Y_train, X_train, Y_val, X_val, center_B_train, center_B_val, 
              lr_init, epoch_init, step, lambda1, lambda2, tol, checkpoint_path, option="new"):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    _, I1, I2 = Y_train.shape
    
    log_sigma_1_init = np.maximum(0.0, np.round(np.random.uniform(np.log(I1/5.0), np.log(I1/3.5), size=(I1, I2)), 1))
    log_sigma_2_init = np.maximum(0.0, np.round(np.random.uniform(np.log(I2/5.0), np.log(I2/3.5), size=(I1, I2)), 1))
    base_init = np.full(1, 0.65)
    
    log_sigma_1_init_B = np.maximum(0.0, np.round(np.random.uniform(np.log(I1/8.0), np.log(I1/6.0), size=(1)), 1))
    log_sigma_2_init_B = np.maximum(0.0, np.round(np.random.uniform(np.log(I2/8.0), np.log(I2/6.0), size=(1)), 1))
    base_init_B = np.full(1, 1.0)    
    lr = lr_init
    epochs = epoch_init
    
    # Initialize parameters and optimizer
    (log_sigma_1, log_sigma_2, base, 
     log_sigma_1_B, log_sigma_2_B, base_B, 
     optimizer) = init_parameters_and_optimizer(log_sigma_1_init, log_sigma_2_init, base_init, 
                                                log_sigma_1_init_B, log_sigma_2_init_B, base_init_B, 
                                                lr)
                                                
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    
    if option == "retrain":
        checkpoint = torch.load(checkpoint_path)
        log_sigma_1 = checkpoint['log_sigma_1'].clone().detach().requires_grad_(True)
        log_sigma_2 = checkpoint['log_sigma_2'].clone().detach().requires_grad_(True)
        base = checkpoint['base'].clone().detach().requires_grad_(True)
        log_sigma_1_B = checkpoint['log_sigma_1_B'].clone().detach().requires_grad_(True)
        log_sigma_2_B = checkpoint['log_sigma_2_B'].clone().detach().requires_grad_(True)
        base_B = checkpoint['base_B'].clone().detach().requires_grad_(False)
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    for epoch in range(epochs):
   
        # Compute the objective function value for the training set
        loss_train, loss_obj_train = objective_function(Y_train, X_train, log_sigma_1, log_sigma_2, base, 
                                        center_B_train, log_sigma_1_B, log_sigma_2_B, base_B,
                                        lambda1, lambda2)
        
        # Zero gradients, perform a backward pass, and update the parameters
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        scheduler.step()
        
        # Compute the objective function value for the validation set
        loss_val, loss_obj_val = objective_function(Y_val, X_val, log_sigma_1, log_sigma_2, base, 
                                      center_B_val, log_sigma_1_B, log_sigma_2_B, base_B,
                                      lambda1, lambda2)
        
        torch.save({
            'log_sigma_1': log_sigma_1.clone(),
            'log_sigma_2': log_sigma_2.clone(),
            'base': base.clone(),
            'log_sigma_1_B': log_sigma_1_B.clone(),
            'log_sigma_2_B': log_sigma_2_B.clone(),
            'base_B': base_B.clone(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, checkpoint_path)        
        
        # Print the current training and validation losses
        print(f'Epoch [{epoch+1}/{epochs}], Train: {loss_obj_train.item():.4f}, Val: {loss_obj_val.item():.4f}')
        
        if loss_obj_train.item() < tol and loss_obj_val.item() < tol:
            break

    results = {'log_sigma_1': log_sigma_1, 'log_sigma_2': log_sigma_2, 'base': base}
    results_B = {'log_sigma_1_B': log_sigma_1_B, 'log_sigma_2_B': log_sigma_2_B, 'base_B': base_B}
        
    return results, results_B

def predict(Y, X, sigma_1, sigma_2, base, center_B_test, sigma_1_B, sigma_2_B, base_B):
    N, I1, I2 = Y.shape
    _, numControl = X.shape
    
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
        # BX
        control = torch.zeros((I1,I2))
        for act in range(numControl):
            cX_B, cY_B = center_B_test[t, act]
            B_tmp = gaussian_product_kernel(xx, yy, cX_B, cY_B, sigma_1_B, sigma_2_B).T * base_B
            control += X[t,act] * B_tmp 
        
        pred[t] = Y_estimate + control  
    
    return pred[1:]
