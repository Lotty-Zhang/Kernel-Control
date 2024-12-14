#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils_3D.kernel_utils import gaussian_product_kernel

def init_parameters_and_optimizer(log_sigma_1_init, log_sigma_2_init, log_sigma_3_init, base_init, log_sigma_1_init_B, log_sigma_2_init_B, log_sigma_3_init_B, base_init_B, lr):
    log_sigma_1 = torch.tensor(log_sigma_1_init, requires_grad=True)
    log_sigma_2 = torch.tensor(log_sigma_2_init, requires_grad=True)
    log_sigma_3 = torch.tensor(log_sigma_3_init, requires_grad=True)
    base = torch.tensor(base_init, requires_grad=True)

    log_sigma_1_B = torch.tensor(log_sigma_1_init_B, requires_grad=True)
    log_sigma_2_B = torch.tensor(log_sigma_2_init_B, requires_grad=True)
    log_sigma_3_B = torch.tensor(log_sigma_3_init_B, requires_grad=True)
    base_B = torch.tensor(base_init_B, requires_grad=False)

    optimizer = optim.Adam([log_sigma_1, log_sigma_2, log_sigma_3, base, log_sigma_1_B, log_sigma_2_B, log_sigma_3_B, base_B], lr=lr, betas=(0.8, 0.98))

    return log_sigma_1, log_sigma_2, log_sigma_3, base, log_sigma_1_B, log_sigma_2_B, log_sigma_3_B, base_B, optimizer

def objective_function(Y, X, log_sigma_1, log_sigma_2, log_sigma_3, base, center_B, log_sigma_1_B, log_sigma_2_B, log_sigma_3_B, base_B, lambda1, lambda2, lambda3):
    
    N, I1, I2, I3 = Y.shape
    N, numControl = X.shape
    
    sigma_1 = torch.exp(log_sigma_1)
    sigma_2 = torch.exp(log_sigma_2)
    sigma_3 = torch.exp(log_sigma_3)
    sigma_1_B = torch.exp(log_sigma_1_B)
    sigma_2_B = torch.exp(log_sigma_2_B) 
    sigma_3_B = torch.exp(log_sigma_3_B) 
        
    x = torch.arange(I1)
    y = torch.arange(I2)
    z = torch.arange(I3)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    
    A = torch.zeros((I1,I2,I3,I1,I2,I3))
        
    for i1 in range(I1):
        for i2 in range(I2):
            for i3 in range(I3):
                A[:, :, i1, i2, i3] = gaussian_product_kernel(xx, yy, zz, i1, i2, i3, sigma_1[i1,i2,i3], sigma_2[i1,i2,i3], sigma_3[i1,i2,i3]).permute(2, 1, 0) * base
            
    diff = torch.zeros(N)
    
    for t in range(1, N):
        # AY
        Y_estimate = torch.sum(A * Y[t-1], axis=(3,4,5))
        # BX
        control = torch.zeros((I1,I2,I3))
        for act in range(numControl):
            cX_B, cY_B, cZ_B = center_B[t, act]
            B_tmp = gaussian_product_kernel(xx, yy, zz, cX_B, cY_B, cZ_B, sigma_1_B, sigma_2_B, sigma_3_B).permute(2, 1, 0) * base_B
            control += X[t,act] * B_tmp 
            
        diff[t] = torch.norm(Y[t] - Y_estimate - control, p='fro') / torch.norm(Y[t], p='fro') 
        
    loss1 = torch.sum(diff) / (N-1)
    loss2 = torch.norm(sigma_1-1, p=1)
    loss = loss1
          
    return loss, loss1, loss2

def train_val(Y_train, X_train, Y_val, X_val, center_B_train, center_B_val, 
              lr_init, epoch_init, step, lambda1, lambda2, lambda3, tol, checkpoint_path, option="retrain"):
    
    _, I1, I2, I3 = Y_train.shape
    log_sigma_1_init = np.maximum(0.0, np.round(np.random.uniform(np.log(I1/2.5), np.log(I1/2.0), size=(I1, I2, I3)), 1))
    log_sigma_2_init = np.maximum(0.0, np.round(np.random.uniform(np.log(I2/2.5), np.log(I2/2.0), size=(I1, I2, I3)), 1))
    log_sigma_3_init = np.maximum(0.0, np.round(np.random.uniform(np.log(I3/2.5), np.log(I3/2.0), size=(I1, I2, I3)), 1))
    base_init = np.full(1, 0.005)
    log_sigma_1_init_B = np.maximum(0.0, np.round(np.random.uniform(np.log(1.3), np.log(1.6), size=(1)), 1))
    log_sigma_2_init_B = np.maximum(0.0, np.round(np.random.uniform(np.log(1.3), np.log(1.6), size=(1)), 1))
    log_sigma_3_init_B = np.maximum(0.0, np.round(np.random.uniform(np.log(1.3), np.log(1.6), size=(1)), 1))
    base_init_B = np.full(1, 1.0)    
    lr = lr_init
    epochs = epoch_init
    
    # Initialize parameters and optimizer
    (log_sigma_1, log_sigma_2, log_sigma_3, base, 
     log_sigma_1_B, log_sigma_2_B, log_sigma_3_B, base_B, 
     optimizer) = init_parameters_and_optimizer(log_sigma_1_init, log_sigma_2_init, log_sigma_3_init, base_init, 
                                                log_sigma_1_init_B, log_sigma_2_init_B, log_sigma_3_init_B, base_init_B, 
                                                lr)
                                                
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    
    if option == "retrain":
        checkpoint = torch.load(checkpoint_path)
        log_sigma_1 = checkpoint['log_sigma_1'].clone().detach().requires_grad_(True)
        log_sigma_2 = checkpoint['log_sigma_2'].clone().detach().requires_grad_(True)
        log_sigma_3 = checkpoint['log_sigma_3'].clone().detach().requires_grad_(True)
        base = checkpoint['base'].clone().detach().requires_grad_(True)
        log_sigma_1_B = checkpoint['log_sigma_1_B'].clone().detach().requires_grad_(True)
        log_sigma_2_B = checkpoint['log_sigma_2_B'].clone().detach().requires_grad_(True)
        log_sigma_3_B = checkpoint['log_sigma_3_B'].clone().detach().requires_grad_(True)
        base_B = checkpoint['base_B'].clone().detach().requires_grad_(False)
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    for epoch in range(epochs):
   
        # Compute the objective function value for the training set
        loss_train, loss_obj_train, loss_L1_train = objective_function(Y_train, X_train, log_sigma_1, log_sigma_2, log_sigma_3, base, 
                                        center_B_train, log_sigma_1_B, log_sigma_2_B, log_sigma_3_B, base_B,
                                        lambda1, lambda2, lambda3)
        
        # Zero gradients, perform a backward pass, and update the parameters
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        scheduler.step()
        
        # Compute the objective function value for the validation set
        loss_val, loss_obj_val, loss_L1_val = objective_function(Y_val, X_val, log_sigma_1, log_sigma_2, log_sigma_3, base, 
                                      center_B_val, log_sigma_1_B, log_sigma_2_B, log_sigma_3_B, base_B,
                                      lambda1, lambda2, lambda3)
        
        torch.save({
            'log_sigma_1': log_sigma_1.clone(),
            'log_sigma_2': log_sigma_2.clone(),
            'log_sigma_3': log_sigma_3.clone(),
            'base': base.clone(),
            'log_sigma_1_B': log_sigma_1_B.clone(),
            'log_sigma_2_B': log_sigma_2_B.clone(),
            'log_sigma_3_B': log_sigma_3_B.clone(),
            'base_B': base_B.clone(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, checkpoint_path)        
        
        # Print the current training and validation losses
        print(f'Epoch [{epoch+1}/{epochs}], Train: {loss_obj_train.item():.4f}, Val: {loss_obj_val.item():.4f}')
        
        if loss_obj_train.item() < tol and loss_obj_val.item() < tol:
            break

    results = {'log_sigma_1': log_sigma_1, 'log_sigma_2': log_sigma_2, 'log_sigma_3': log_sigma_3, 'base': base}
    results_B = {'log_sigma_1_B': log_sigma_1_B, 'log_sigma_2_B': log_sigma_2_B, 'log_sigma_3_B': log_sigma_3_B, 'base_B': base_B}
        
    return results, results_B

def predict(Y, X, sigma_1, sigma_2, sigma_3, base, center_B_test, sigma_1_B, sigma_2_B, sigma_3_B, base_B):
    N, I1, I2, I3 = Y.shape
    _, numControl = X.shape
    
    x = torch.arange(I1)
    y = torch.arange(I2)
    z = torch.arange(I3)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    
    A = torch.zeros((I1,I2,I3,I1,I2,I3))
        
    for i1 in range(I1):
        for i2 in range(I2):
            for i3 in range(I3):
                A[:, :, i1, i2, i3] = gaussian_product_kernel(xx, yy, zz, i1, i2, i3, sigma_1[i1,i2,i3], sigma_2[i1,i2,i3], sigma_3[i1,i2,i3]).permute(2, 1, 0) * base
            
    pred = torch.zeros(N, I1, I2, I3)
    
    for t in range(1, N):
        # AY
        Y_estimate = torch.sum(A * Y[t-1], axis=(3,4,5))
        # BX
        control = torch.zeros((I1,I2,I3))
        for act in range(numControl):
            cX_B, cY_B, cZ_B = center_B_test[t, act]
            B_tmp = gaussian_product_kernel(xx, yy, zz, cX_B, cY_B, cZ_B, sigma_1_B, sigma_2_B, sigma_3_B).permute(2, 1, 0) * base_B
            control += X[t,act] * B_tmp 
        
        pred[t] = Y_estimate + control  
    
    return pred[1:]
