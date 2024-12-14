#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import cv2
import imageio.v2 as imageio
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

def scale(item, scale_min=0, scale_max=255):
    # Compute the range of the input item
    item_range = max(np.max(item) - np.min(item), np.finfo(float).eps)
    
    # Scale the item to the desired range and clip values outside the range
    scaled_item = np.clip((item - np.min(item)) / item_range * (scale_max - scale_min) + scale_min,
                         scale_min, scale_max)
    
    return scaled_item
    
def save_image(image, path):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(path, facecolor='none')
    plt.close()    
