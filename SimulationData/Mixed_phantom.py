#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:44:48 2023

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



def create_five_mask_pentagons(h, w, mx, my):
    Y, X = np.ogrid[:h, :w] # phantom's size
    # Pentagon's formulation
    mask1 = (X-30) - 10/np.tan(36*np.pi/180)-mx <Y/np.tan(36*np.pi/180)-my
    mask2 = -(X+30)+ 83/np.tan(36*np.pi/180)+mx <Y/np.tan(36*np.pi/180)-my
    mask3 = -(X+30) + 477/np.tan(72*np.pi/180)+mx >Y/np.tan(72*np.pi/180)+my
    mask4 = (X-30) + 83/np.tan(72*np.pi/180)-mx  >Y/np.tan(72*np.pi/180)+my
    mask5 = Y+my < 100 
    mask = np.logical_and(mask1, mask2) 
    mask = np.logical_and(mask, mask3)
    mask = np.logical_and(mask, mask4)
    mask = np.logical_and(mask, mask5)
    mask = mask.astype(int)

    return mask


# set the ranges
mx_range_f = np.random.randint(-10, 10, 3000)
my_range_f = np.random.randint(0, 10, 3000)
rotation_range_f = np.linspace(0, 180, 36, False)
rotation_label_f = np.random.randint(0, 36, 3000)


# generate 3000 phantoms    
P_f = []
L_f = []

for i in range(3000):
    ph = create_five_mask_pentagons(128, 128, mx_range_f[i], my_range_f[i])
    label_n_f = rotation_range_f[rotation_label_f[i]]
    ph = ndimage.rotate(ph, label_n_f, reshape=False)
    ph[ph<0.5] = 0
    ph[ph>=0.5] = 1
    P_f.append(ph)
    L_f.append(label_n_f)
    
    
    
    
def create_three_mask_triangle(h, w, mx, my):
    Y, X = np.ogrid[:h, :w] # phantom's size
    # Triangle's formulation 
    mask1 = X - mx < 100 + my
    mask2 = -X+ 120/np.tan(45*np.pi/180)+mx <Y/np.tan(45*np.pi/180) + my
    mask3 = Y < 100+my
    mask = np.logical_and(mask1, mask2) 
    mask = np.logical_and(mask, mask3)
    mask = mask.astype(int)

    return mask


# set the ranges
mx_range_t = np.random.randint(-10, 10, 3000)
my_range_t = np.random.randint(-10, 0, 3000)
rotation_range_t = np.linspace(0, 180, 36, False)
rotation_label_t = np.random.randint(0, 36, 3000)

# generate 3000 phantoms   
P_t = []
L_t = []

for i in range(3000):
    ph = create_three_mask_triangle(128, 128, mx_range_t[i], my_range_t[i])
    label_n_t = rotation_range_t[rotation_label_t[i]]
    ph = ndimage.rotate(ph, label_n_t, reshape=False)
    ph[ph<0.5] = 0
    ph[ph>=0.5] = 1
    P_t.append(ph)
    L_t.append(label_n_t)
    

    
def create_six_mask_hexagon(h, w, mx, my):
    Y, X = np.ogrid[:h, :w] # phantom's size
    # Hexagon's formulation
    mask1 = (X-30) - 10/np.tan(30*np.pi/180)-mx <Y/np.tan(30*np.pi/180)-my
    mask2 = -(X+30)+ 64/np.tan(30*np.pi/180)+mx <Y/np.tan(30*np.pi/180)-my
    mask3 = (X-30) - 10/np.tan(30*np.pi/180)-mx >Y/np.tan(30*np.pi/180)+my-180
    mask4 = -(X+30)+ 64/np.tan(30*np.pi/180)+mx >Y/np.tan(30*np.pi/180)+my-180
    mask5 = X > 20 + mx + my/2
    mask6 = X < 108 + mx - my/2
    mask = np.logical_and(mask1, mask2) 
    mask = np.logical_and(mask, mask3)
    mask = np.logical_and(mask, mask4)
    mask = np.logical_and(mask, mask5)
    mask = np.logical_and(mask, mask6)
    mask = mask.astype(int)

    return mask




# set the range
mx_range_s = np.random.randint(0, 10, 3000)
my_range_s = np.random.randint(0, 10, 3000)
rotation_range_s = np.linspace(0, 180, 36, False)
rotation_label_s = np.random.randint(0, 36, 3000)

# generate 3000 phantoms
P_s = []
L_s = []

for i in range(3000):
    ph = create_six_mask_hexagon(128, 128, mx_range_s[i], my_range_s[i])
    label_n_s = rotation_range_s[rotation_label_s[i]]
    ph = ndimage.rotate(ph, label_n_s, reshape=False)
    ph[ph<0.5] = 0
    ph[ph>=0.5] = 1
    P_s.append(ph)
    L_s.append(label_n_s)  


# Put all phantoms together and shuffle them
P_all = P_t + P_f + P_s
random.shuffle(P_all)
