#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lotty
"""

import numpy as np

def fix_sampling_3D(numControl, I1, I2, I3):
    
    np.random.seed(15)
    
    points_I1 = np.random.choice(np.linspace(0, I1-1, numControl, dtype=int), size=numControl, replace=False)
    points_I2 = np.random.choice(np.linspace(0, I2-1, numControl, dtype=int), size=numControl, replace=False)
    points_I3 = np.random.choice(np.linspace(0, I3-1, numControl, dtype=int), size=numControl, replace=False)
    
    np.random.shuffle(points_I1)
    np.random.shuffle(points_I2)
    np.random.shuffle(points_I3)
    
    center_B = np.vstack([points_I1, points_I2, points_I3]).T
    
    return center_B
