#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:30:45 2023

@author: tianyuan
"""

import numpy as np

import torch
import os
import random
from scipy import ndimage

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


# create elli[pse phantom

def create_ellipse_mask(h, w, m_l, m_s, scale, h_s, w_s, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2+w_s), int(h/2+h_s))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w] # define the phantom's size
    
    dist_from_center = np.sqrt(((X - center[0])/m_l)**2 + ((Y-center[1])/m_s)**2) # Ellipse formulation
    mask = dist_from_center <= scale
    return mask.astype(int)



# set the ranges
scale_range = np.random.uniform(0.8, 1.2, 3000)

w_range = np.random.randint(-15, 15, 3000)
h_range = np.random.randint(-15, 15, 3000)

rotation_range = np.linspace(0,180,36,False)
rotation_lable = np.random.randint(0, 36, 3000)


# generate 3000 phantoms
P_all = []
L = []
for i in range(3000):
    ph = create_ellipse_mask(128, 128, 18, 35, scale_range[i], h_range[i], w_range[i])
    label_n = rotation_range[rotation_lable[i]]
    ph = ndimage.rotate(ph, label_n, reshape=False)
    # It could change pixel value after rotation
    ph[ph<0.5] = 0
    ph[ph>=0.5] = 1
    P_all.append(ph)
    L.append(label_n)

