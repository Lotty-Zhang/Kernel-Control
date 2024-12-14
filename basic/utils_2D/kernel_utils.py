#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""
import numpy as np
import torch

def gaussian_product_kernel(x, y, center_x, center_y, sigma_1, sigma_2):
    diff_x = x - center_x
    diff_y = y - center_y
    exponent = -(diff_x ** 2) / (2 * (sigma_1 ** 2)) - (diff_y ** 2) / (2 * (sigma_2 ** 2))
    
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.exp(exponent)
    elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return torch.exp(exponent)
    else:
        raise TypeError("Input types are not supported. Please provide either NumPy arrays or PyTorch tensors.")