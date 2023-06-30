#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:15:23 2023

@author: tianyuan
"""

import numpy as np

import torch
import os
import random




# set seed to make sure reproducibility

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()



# create circle phantom

def create_circular_mask(h, w, scale, shift_x, shift_y, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/shift_x), int(h/shift_y))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w] # define the phantom's size
    
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2) # Circle formulation
    mask = dist_from_center <= radius/scale
    return mask.astype(int)




# set the ranges
scale_range = np.random.uniform(1.8, 2.2, 1000)
shift_x_range = np.random.uniform(1.5, 3, 1000)
shift_y_range = np.random.uniform(1.5, 3, 1000)

# generate 1000 phantoms
P_all = []
for i in range(1000):
    ph = create_circular_mask(128, 128, scale_range[i], shift_x_range[i], shift_y_range[i])
    P_all.append(ph)


