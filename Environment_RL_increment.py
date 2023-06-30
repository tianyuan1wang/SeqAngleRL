#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:52:25 2023

@author: tianyuan
"""

import numpy as np

import astra
import math

from SimulationData.Mixed_phantom import *




def reconstruction_noise(P, proj_angles, proj_size, vol_geom, n_iter_sirt, percentage=0.0):
    proj_geom = astra.create_proj_geom('parallel', 1.0, proj_size, proj_angles)
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
    # construct the OpTomo object
    
    W = astra.OpTomo(proj_id)
    sinogram = W * P
    sinogram = sinogram.reshape([len(proj_angles), proj_size])
    
    n = np.random.normal(0, sinogram.std(), (len(proj_angles), proj_size)) * percentage
    
    # gauss1 = gauss.reshape(len(proj_angles), proj_size)
    
    sinogram_n = sinogram + n
    
    rec_sirt = W.reconstruct('SIRT_CUDA', sinogram_n, iterations=n_iter_sirt, extraOptions={'MinConstraint':0.0,'MaxConstraint':1.0})

    
    return rec_sirt



def psnr(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    #print(rmse)
    return 20*math.log10(1.0/rmse)

def angle_range(N_a):
    return np.linspace(0,np.pi,N_a,False)


a_start = 0

image_size = 128
proj_size = int(1.5*image_size)
vol_geom = astra.create_vol_geom(image_size, image_size)
n_iter_sirt = 150


N_a = 180
angles = angle_range(N_a)
len_p = len(P_all)

class env():
    
    def __init__(self):
        self.n = np.random.randint(0,len_p)
        self.criteria = 0
 
        
    def step(self, action):
        
       
        # Execute action on environment 
        self.a_start += 1
            
        # Wait for environment to transition to next state
        self.angle_action = angles[action]
        
            
        self.angles_seq.append(self.angle_action)
        
        self.state = reconstruction_noise(P_all[self.n], self.angles_seq, proj_size, vol_geom, n_iter_sirt)
       
        # Get reward for new state
        self.reward, self.previous_action  = self._get_reward(self.angles_seq)
      
       
        self.total_reward += self.reward
        
            
        self.criteria = psnr(P_all[self.n], self.state)
        
        
        
        if self.a_start > 6:
            # self.n = np.random.randint(0,4)
            self.a_start = 0
            self.angles_seq = []
            self.done = True
         
            
      

        return np.array(self.state), self.reward, self.done, self.angle_action, self.angles_seq, self.n
           
    
    def reset(self):
        self.n = np.random.randint(0,len_p)
       
        self.a_start = a_start
        
        self.curr_iteration = 0
              
        self.total_reward = 0.0
        self.angles_seq = []
        
        #Default initialization for action
        self.previous_action=0 
        self.previous_reward = 0
        self.action=self.previous_action
       
        self.reward = 0
        
        self.done=False 
        #self.n = 0
        
        
        self.state = np.zeros((128,128))
        self.criteria = 0
        
        return self.state
    
    def _get_reward(self,angles_seq):
        
        
        self.current_reward = psnr(P_all[self.n], self.state)
        
        reward = self.current_reward - self.previous_reward
        self.previous_reward = self.current_reward
        
     
        self.previous_action=self.angles_seq[-1]
        
      
        
        return reward,self.previous_action