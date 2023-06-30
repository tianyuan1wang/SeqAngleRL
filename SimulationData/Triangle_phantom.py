#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:39:28 2023

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

def generate_phantom(a_range, mean_angle, b_range, size):
    # define the phantom's size
    xx, yy = np.meshgrid(np.arange(0, size), np.arange(0, size))
   
    size1 = int(size+5*a_range)
    P = np.ones([size,size])
    
    # Triangle formulation
    x = round(1*size1/2)
    y = round(1*size1/2)
    slope = 0
    z1 = (yy-3/2*y<-slope*(xx-slope*(3/2*x)/(1+slope**2)**0.5))
    z2 = (slope*(yy-slope*(3/2*y)/(1+slope**2)**0.5)<(xx-1/2*x))
    z3 = (yy-1/2*y>-slope*(xx+slope*(-x)/(1+slope**2)**0.5))
    z4 = (slope*(yy-slope*(y)/(1+slope**2)**0.5)>(xx-3/2*x))
    z5 = (yy-y+1 > -(1-slope)*(xx-x))

    P *= z1
    P *= z2
    P *= z3
    P *= z4
    P *= z5    
        
    
    return P

# set the ranges
P_all = []
L = []
v_a = np.random.uniform(-10, 2.5, 3000)
rotation_range = np.linspace(0, 180, 36, False)
rotation_label = np.random.randint(0, 36, 3000)
  
# generate 3000 phantoms    
for i in range(3000):
   a_range = v_a[i]
       
   P = generate_phantom(a_range, 0, 0, 128)
           
            
   P = ndimage.rotate(P, rotation_range[rotation_label[i]], reshape=False)
   P[P<0.5] = 0
   P[P>=0.5] = 1
   P_all.append(P)
   L.append(rotation_range[rotation_label[i]])