#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""
import numpy as np
from utils_case.kernel_utils import gaussian_product_kernel
from sklearn.linear_model import Lasso

def X_to_center(X, numControl):
    nonzero_indices = np.transpose(np.nonzero(X))

    if len(nonzero_indices) > numControl:
        np.random.shuffle(nonzero_indices)
        nonzero_indices = nonzero_indices[:numControl]
    
    center_B = nonzero_indices.astype(int)
    
    return center_B

def center_to_X(center_B, I1, I2):
    X = np.zeros((I1, I2))
    
    for point in center_B:
        x, y = point
        X[int(x), int(y)] = 1
    
    return X

def cal_alpha_max(B_L1, tarAY, I1, I2):
    
    B_L1_reshaped = B_L1.reshape(I1 * I2, I1 * I2)
    tarAY_reshaped = tarAY.reshape(I1 * I2, 1)
    alpha_max = np.linalg.norm(2 * B_L1_reshaped.T @ tarAY_reshaped, ord=np.inf)
    
    return alpha_max
    
def best_center_alpha(alpha_min, alpha_max, numControl, max_iterations, tarAY, B_L1, I1, I2, X, t):
    
    B_L1_reshape = B_L1.reshape(I1 * I2, I1 * I2)
    tarAY_reshape = tarAY.reshape(I1 * I2, 1)    
    
    for i in range(max_iterations):
        alpha = (alpha_min + alpha_max) / 2
        lasso = Lasso(alpha=alpha/(2*(I1*I2)), fit_intercept=False, tol=1e-8, max_iter=10000)
        lasso.fit(B_L1_reshape, tarAY_reshape)
        coef = lasso.coef_
        X[t] = coef.reshape(I1, I2)
        
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

def B_set(I1, I2, xx, yy, sigma_1, sigma_2, base):
    
    B_set = np.zeros((I1, I2, I1, I2))
    for i1 in range(I1):
        for i2 in range(I2):
            B_set[:, :, i1, i2] = gaussian_product_kernel(xx, yy, i1, i2, sigma_1, sigma_2).T * base
            
    return B_set

def get_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def onlineControl_dymL1_B(Y_0, noise_or, numControl, center_B, log_sigma_1, log_sigma_2, base, b_B, magnitude_B, target, Np):
    
    I1, I2 = Y_0.shape

    x = np.arange(I1)
    y = np.arange(I2)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    
    Y_pred = np.zeros((Np, I1, I2))
    Y_phys_wc = np.zeros((Np, I1, I2))
    Y_phys_nc = np.zeros((Np, I1, I2))
    
    Y_pred[0] = Y_0
    Y_phys_wc[0] = Y_0
    Y_phys_nc[0] = Y_0
    
    A_est = np.zeros((I1, I2, I1, I2))
    for i1 in range(I1):
        for i2 in range(I2):
            A_est[:, :, i1, i2] = gaussian_product_kernel(xx, yy, i1, i2, np.exp(log_sigma_1[i1,i2]), np.exp(log_sigma_2[i1,i2])).T * base
     
    X_pred = np.zeros((Np, numControl))
    center_B_save = np.zeros((Np, numControl, 2))
    X = np.zeros((Np, I1, I2))
    
    center_B_save[0] = center_B    
    X[0] = center_to_X(center_B, I1, I2)
    
    B_L1 = B_set(I1, I2, xx, yy, b_B[0,0], b_B[0,1], magnitude_B)
    B_true = B_set(I1, I2, xx, yy, b_B[0,0], b_B[0,1], magnitude_B)
    
    max_iter = 1000
    
    for t in range(1, Np):
        
        Y_tmp_pred = np.sum(A_est * Y_pred[t-1], axis=(2, 3))
        Y_tmp_wc = np.sum(A_est * Y_phys_wc[t-1], axis=(2, 3))
        Y_tmp_nc = np.sum(A_est * Y_phys_nc[t-1], axis=(2, 3))
        tmp_X = np.sum(A_est * Y_phys_wc[t-1], axis=(2, 3))
        
        # determine control center
        tarAY = target - tmp_X
        alpha_max = cal_alpha_max(B_L1, tarAY, I1, I2)
        X, alpha = best_center_alpha(0, alpha_max, numControl, max_iter, tarAY, B_L1, I1, I2, X, t)
        print(f"Time {t}, alpha_max {alpha_max:.4f}, alpha {alpha:.4f}\n")
        center_B = X_to_center(X[t], numControl)
        print(center_B)
        
        # determine control actions
        B_est = np.zeros((I1, I2, numControl))
        for act in range(numControl):
            B_est[:,:,act] = gaussian_product_kernel(xx, yy, center_B[act,0], center_B[act,1], b_B[0,0], b_B[0,1]).T * magnitude_B       
        
        # calculate the control action
        tarAY = target - tmp_X
        tarAY_reshaped = tarAY.reshape(I1 * I2, 1)
        B_est_reshaped = B_est.reshape(I1 * I2, center_B.shape[0])
    
        X_est, _, _, _ = np.linalg.lstsq(B_est_reshaped, tarAY_reshaped, rcond=None)
        
        X_pred[t] = X_est.flatten()
    
        XB_pred = np.zeros((I1, I2))
        XB_phys = np.zeros((I1, I2))
        
        center_B_save[t] = center_B
        
        for act in range(center_B.shape[0]):
            i1, i2 = center_B[act]
            XB_pred += X_est[act] * B_L1[:, :, int(i1), int(i2)]        
            XB_phys += X_est[act] * B_true[:, :, int(i1), int(i2)]               
            
        Y_pred[t] = Y_tmp_pred + XB_pred
        Y_phys_wc[t] = Y_tmp_wc + XB_phys + noise_or[t]
        Y_phys_nc[t] = Y_tmp_nc + noise_or[t]
        
        print(f'{t} done')

    return Y_pred, Y_phys_wc, Y_phys_nc, X_pred, center_B_save
