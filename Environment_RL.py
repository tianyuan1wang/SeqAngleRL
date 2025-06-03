#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:52:25 2023

@author: tianyuan
"""

import numpy as np

import astra
import math
import argparse
import importlib

from PhantomGenerator import PhantomGenerator

# Generate your phantoms
gen = PhantomGenerator()
P_all = gen.generate_mixed(n_samples=3000)


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



class env():
    
    def __init__(self, num_angles, reward_mode, image_size, action_size):
        # Select phantom index
        self.n = np.random.randint(0,len(P_all))
        self.criteria = 0
        # Total number of angles for this experiment
        self.num_angles = num_angles
        # Reward mode: increment or endtoend
        self.reward_mode = reward_mode
        # Image size 
        self.image_size = image_size
        # The size of action space
        self.action_size = action_size
        # Parameters for astra
        self.proj_size = int(1.5*self.image_size)
        self.vol_geom = astra.create_vol_geom(self.image_size, self.image_size)
        self.angles = angle_range(self.action_size)
        self.n_iter_sirt = 150
        self.init_start = 0

 
    def step(self, action):
        
       
        # # The number of angles after selecting the next angle
        self.a_start += 1
            
        # Find which angle is selected and store it together with previous selected angles
        self.angle_action = self.angles[action]
        
            
        self.angles_seq.append(self.angle_action)
        
        # Use all selected angles to do reconstruction using SIRT as a belief state
        self.state = reconstruction_noise(P_all[self.n], self.angles_seq, self.proj_size, self.vol_geom, self.n_iter_sirt)
       
        # Get reward for new state
        if self.reward_mode == "increment":
            self.reward  = self._get_reward_increm()
        elif self.reward_mode == "endtoend":
            self.reward = self._get_reward_end()
      
        # Calculate the total rewards
        self.total_reward += self.reward
        
            
        # self.criteria = psnr(P_all[self.n], self.state)
        
        
        # The stop criteria depends on the number of angles; if the criteria is reached, go another round 
        if self.a_start > self.num_angles:
            # self.n = np.random.randint(0,4)
            self.a_start = self.init_start
            self.angles_seq = []
            self.done = True
         
            
      

        return np.array(self.state), self.reward, self.done, self.angle_action, self.angles_seq, self.n
           
    
    def reset(self):
        self.n = np.random.randint(0,len(P_all))
       
        self.a_start = self.init_start
        
        self.curr_iteration = 0
              
        self.total_reward = 0.0
        self.angles_seq = []
        
        # initialization for action
        self.previous_reward = 0
       
        self.reward = 0
        
        self.done=False 
        #self.n = 0
        
        # set a zero matrix as the first reconstruction or belief state
        self.state = np.zeros((128,128))
        # self.criteria = 0
        
        return self.state
    
    def _get_reward_increm(self,):
        # calculate the psnr value for the current reconstruction
        self.current_reward = psnr(P_all[self.n], self.state)
        
        # incremental reward setting
        reward = self.current_reward - self.previous_reward
        self.previous_reward = self.current_reward
        
     
        self.previous_action=self.angles_seq[-1]

        return reward

    def _get_reward_end(self,):
        # calculate the psnr value for the current reconstruction
        self.current_reward = psnr(P_all[self.n], self.state)   
        # end-to-end reward setting
        if self.a_start > self.num_angles:
            reward = self.current_reward
        else:
            reward = 0
        
        return reward