#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""
import numpy as np
import torch

def gaussian_product_kernel(x, y, z, center_x, center_y, center_z, sigma_1, sigma_2, sigma_3):
    diff_x = x - center_x
    diff_y = y - center_y
    diff_z = z - center_z
    exponent = -(diff_x ** 2) / (2 * (sigma_1 ** 2)) - (diff_y ** 2) / (2 * (sigma_2 ** 2)) - (diff_z ** 2) / (2 * (sigma_3 ** 2))
    
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(z, np.ndarray):
        return np.exp(exponent)
    elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and isinstance(z, torch.Tensor):
        return torch.exp(exponent)
    else:
        raise TypeError("Input types are not supported. Please provide either NumPy arrays or PyTorch tensors.")    
        