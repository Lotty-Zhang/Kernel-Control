#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""
import numpy as np
from utils_3D.kernel_utils import gaussian_product_kernel
from sklearn.linear_model import Lasso

def X_to_center(X, numControl):

    nonzero_indices = np.transpose(np.nonzero(X))
    
    # If there are more non-zero elements than needed, sort by the values at those indices
    if len(nonzero_indices) > numControl:
        values = X[tuple(nonzero_indices.T)]
        sorted_indices = nonzero_indices[np.argsort(-values)][:numControl]
    else:
        sorted_indices = nonzero_indices
    
    center_B = sorted_indices.astype(int)
    
    return center_B

def center_to_X(center_B, I1, I2, I3, seed):
    
    np.random.seed(seed)
    
    X = np.zeros((I1, I2, I3))
    
    for point in center_B:
        x, y, z = point
        X[int(x), int(y), int(z)] = 1
    
    return X

def cal_alpha_max(B_L1, tarAY, I1, I2, I3):
    
    B_L1_reshaped = B_L1.reshape(I1 * I2 * I3, I1 * I2 * I3)
    tarAY_reshaped = tarAY.reshape(I1 * I2 * I3, 1)
    alpha_max = np.linalg.norm(2 * B_L1_reshaped.T @ tarAY_reshaped, ord=np.inf)
    
    return alpha_max
    
def best_center_alpha(alpha_min, alpha_max, numControl, max_iterations, tarAY, B_L1, I1, I2, I3, X, t, seed):
    
    B_L1_reshape = B_L1.reshape(I1 * I2 * I3, I1 * I2 * I3)
    tarAY_reshape = tarAY.reshape(I1 * I2 * I3, 1)    
    
    for i in range(max_iterations):
        alpha = (alpha_min + alpha_max) / 2
        lasso = Lasso(alpha=alpha/(2*(I1*I2*I3)), fit_intercept=False)
        lasso.fit(B_L1_reshape, tarAY_reshape)
        coef = lasso.coef_
        X[t] = coef.reshape(I1, I2, I3)
        
        non_zero_count = np.count_nonzero(X[t])
        
        if non_zero_count == numControl:
            print(f"Time {t} search done at iter {i}.")
            return X, alpha
        elif non_zero_count < numControl:
            alpha_max = alpha
        else:
            alpha_min = alpha
            
    print(f"Time {t}, non_zero_count {non_zero_count}")

    return X, alpha

def B_set(I1, I2, I3, xx, yy, zz, sigma_1, sigma_2, sigma_3, base):
    
    B_set = np.zeros((I1, I2, I3, I1, I2, I3))
    for i1 in range(I1):
        for i2 in range(I2):
            for i3 in range(I3):
                
                B_set[:, :, :, i1, i2, i3] = gaussian_product_kernel(xx, yy, zz, i1, i2, i3, sigma_1, sigma_2, sigma_3).T * base
            
    return B_set

def onlineControl_dymL1_B(Y_0, Np, ratio, noise_std, numControl, center_B, log_sigma_1, log_sigma_2, log_sigma_3, base, log_sigma_1_B, log_sigma_2_B, log_sigma_3_B, base_B, A, b_B, magnitude_B, target, seed):
    
    np.random.seed(seed)
    
    I1, I2, I3 = Y_0.shape
    
    sigma_1_B = np.exp(log_sigma_1_B)
    sigma_2_B = np.exp(log_sigma_2_B)
    sigma_3_B = np.exp(log_sigma_3_B)

    x = np.arange(I1)
    y = np.arange(I2)
    z = np.arange(I3)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    Y_pred = np.zeros((Np, I1, I2, I3))
    Y_phys_wc = np.zeros((Np, I1, I2, I3))
    Y_phys_nc = np.zeros((Np, I1, I2, I3))
    
    Y_pred[0] = Y_0
    Y_phys_wc[0] = Y_0
    Y_phys_nc[0] = Y_0
    
    A_est = np.zeros((I1, I2, I3, I1, I2, I3))
    for i1 in range(I1):
        for i2 in range(I2):
            for i3 in range(I3):
                
                A_est[:, :, :, i1, i2, i3] = gaussian_product_kernel(xx, yy, zz, i1, i2, i3, np.exp(log_sigma_1[i1,i2,i3]), np.exp(log_sigma_2[i1,i2,i3]), np.exp(log_sigma_3[i1,i2,i3])).T * base
     
    X_pred = np.zeros((Np, numControl))
    center_B_save = np.zeros((Np, numControl, 3))
    X = np.zeros((Np, I1, I2, I3))
    
    center_B_save[0] = center_B    
    X[0] = center_to_X(center_B, I1, I2, I3, seed)
    
    B_L1 = B_set(I1, I2, I3, xx, yy, zz, sigma_1_B, sigma_2_B, sigma_3_B, base_B)
    B_true = B_set(I1, I2, I3, xx, yy, zz, b_B[0,0], b_B[0,1], b_B[0,2], magnitude_B)
    
    max_iter = 1000
    
    for t in range(1, Np):
        
        Y_tmp_pred = np.sum(A_est * Y_pred[t-1], axis=(3,4,5))
        Y_tmp_wc = np.sum(A * Y_phys_wc[t-1], axis=(3,4,5))
        Y_tmp_nc = np.sum(A * Y_phys_nc[t-1], axis=(3,4,5))
        tmp_X = np.sum(A_est * Y_phys_wc[t-1], axis=(3,4,5))
        
        # determine control center
        tarAY = target - tmp_X
        alpha_max = cal_alpha_max(B_L1, tarAY, I1, I2, I3)
        X, alpha = best_center_alpha(0, alpha_max, numControl, max_iter, tarAY, B_L1, I1, I2, I3, X, t, seed)
        print(f"Time {t}, alpha_max {alpha_max:.4f}, alpha {alpha:.4f}\n")
        center_B = X_to_center(X[t], numControl)
        print(center_B)
        
        # determine control actions
        B_est = np.zeros((I1, I2, I3, numControl))
        for act in range(numControl):
            B_est[:,:,:,act] = gaussian_product_kernel(xx, yy, zz, center_B[act,0], center_B[act,1], center_B[act,2], sigma_1_B, sigma_2_B, sigma_3_B).T * base_B        
        
        # calculate the control action
        tarAY = target - tmp_X
        tarAY_reshaped = tarAY.reshape(I1 * I2 * I3, 1)
        B_est_reshaped = B_est.reshape(I1 * I2 * I3, center_B.shape[0])
    
        X_est, _, _, _ = np.linalg.lstsq(B_est_reshaped, tarAY_reshaped, rcond=None)
        
        X_pred[t] = X_est.flatten()
    
        XB_pred = np.zeros((I1, I2, I3))
        XB_phys = np.zeros((I1, I2, I3))
        
        center_B_save[t] = center_B
        
        for act in range(center_B.shape[0]):
            i1, i2, i3 = center_B[act]
            XB_pred += X_est[act] * B_L1[:, :, :, int(i1), int(i2), int(i3)]        
            XB_phys += X_est[act] * B_true[:, :, :, int(i1), int(i2), int(i3)]               
            
        Y_pred[t] = Y_tmp_pred + XB_pred
        Y_phys_wc[t] = Y_tmp_wc + XB_phys + np.random.normal(0, noise_std, (I1, I2, I3)) * ratio
        Y_phys_nc[t] = Y_tmp_nc + np.random.normal(0, noise_std, (I1, I2, I3)) * ratio
        
        print(f'{t} done')

    return Y_pred, Y_phys_wc, Y_phys_nc, X_pred, center_B_save
