#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:07:23 2023

@author: tianyuan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim 

import os
import random

import numpy as np

import argparse
import importlib

# import subprocess

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Experiments parameters in main')


parser.add_argument('--NUM_EPISODES', type=int, default=300000,
                        help='the number of training episodes')

parser.add_argument('--GAMMA', type=float, default=0.99,
                        help='discount for RL')

parser.add_argument('--LR', type=float, default=1e-4,
                        help='learning rate')

parser.add_argument('--WD', type=float, default=1e-5,
                        help='weight decay')


parser.add_argument('--Environment', type=str, default='Environment_RL_endtoend.py', choices=['Environment_RL_endtoend.py', 'Environment_RL_increment.py'],
                        help='the choice of the environment')

parser.add_argument('--Note', type=str, default='mix', 
                        help='name for saving')



args = parser.parse_args()






module_name = args.Environment.replace('.py', '')

# module.my_function(args.Mixed_phantom)
module = importlib.import_module(module_name)



env = module.env()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


note = args.Note



# episode_rewards = []
# A = []
# P_A = []
# P_S = []

# NUM = []
# E = []
# P_N = []
# STATE = []
# NUM_A = []

# total_rewards = []
# A = []
# P_A = []
# NUM = []



class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim_1, output_dim, n_layers=2):
        super(ActorCritic, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        

        to_pad = int((3 - 1) / 2)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, 
                          out_channels=12, 
                          kernel_size=3,
                          padding=to_pad,
                          stride=2),
                nn.GroupNorm(num_channels=12, 
                              num_groups=4),
                # nn.BatchNorm2d(12),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=12, 
                          out_channels=24, 
                          kernel_size=3,
                          stride=1, 
                          padding=to_pad),
                nn.GroupNorm(num_channels=24, num_groups=4),
                # nn.BatchNorm2d(24),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=24, 
                          out_channels=48, 
                          kernel_size=3,
                          stride=1, 
                          padding=to_pad),
                nn.GroupNorm(num_channels=48, num_groups=4),
                # nn.BatchNorm2d(24),
                nn.LeakyReLU(0.2),
               
                nn.MaxPool2d(kernel_size=4))             
        
        
        
        self.actor = nn.Sequential(
                                  nn.Linear(768+hidden_dim_1, output_dim),
                                  nn.Softmax(dim=-1))
        
       
        
        self.critic = nn.Sequential(
                                    nn.Linear(768+hidden_dim_1, 768+hidden_dim_1),
                                    nn.ReLU(),
                                    nn.Linear(768+hidden_dim_1, 1))
        
    def forward(self,state, state_a):
        state = torch.from_numpy(state).float().squeeze().to(device)
        state = state.unsqueeze(0)
        state = state.unsqueeze(0)
        state_a = torch.from_numpy(state_a).float().to(device)
        p2 = self.conv1(state)
        
        
        p2 = p2.view(p2.size(0), -1)
        # print(p2.shape)
        p3 = torch.cat((p2, state_a),1)
        value = self.critic(p3)
        probs = self.actor(p3)
        dist = Categorical(probs)
        return dist, value

  
def main():
    # set parameters for network
    INPUT_DIM = module.image_size
    HIDDEN_DIM = 4*INPUT_DIM + 1

    OUTPUT_DIM = module.N_a
    HIDDEN_DIM_1 = module.N_a

    model = ActorCritic(INPUT_DIM, HIDDEN_DIM, HIDDEN_DIM_1, OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.WD)



    for e in range(args.NUM_EPISODES):
        # reset the environment and the action vector
        state = env.reset()
        state_a = np.array([[0]*module.N_a])
 
   
    
        # track the total rewards
        score = 0
        # a_episode = []
        # log_probs = []
        # values = []
        # rewards = []
        # masks = []
        # returns = []
        # angle_prob = []
        # Num_P = []
        # e_change = []
        # S = []
        # p_n = []
        # p_s = []
   
  
        while True:
            # outputs from Actor-Critic model based on the current reconstruction
            dist, value = model(state, state_a)
            # Sample action from the output probability distribution
            action = dist.sample()
            # log-trick to cumpute the gradient on policy
            log_prob = dist.log_prob(action)
            # compute the entropy
            entropy = dist.entropy().mean()
            angle_dist = dist.probs.detach().cpu().numpy()
            # angle_prob.append(angle_dist)
            
            
            # outputs from the environment after selecting an angles
            next_state, reward, done, _, c_r, n = env.step(action.item())
        
            # update the action vector
            state_a[0][action] = 1
       
            # outputs from Actor-Critic model based on the next reconstruction
            next_dist, next_value = model(next_state, state_a)
        
            # compute the temperal difference as an approximation of the advantage function
            advantage = reward + (1-done)* args.GAMMA*next_value - value
            
            # compute the different losses with weights
            actor_loss = -(log_prob * advantage.detach())
            critic_loss = advantage.pow(2).mean()
        
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
       
        
            # a_episode.append(action.item())
            # Num_P.append(n)
      
            # e_change.append(entropy.item())
        
            score += reward
        
            state = next_state
        
            # if args.NUM_EPISODES - e <= 200:
            #     S.append(state)
            #     p_n.append(env.n)

        
            if done:
                break
        
        # A.append(a_episode)    
        # P_A.append(angle_prob)
        # discounted_reward = 0
        # NUM.append(Num_P)
        # E.append(e_change)
        # STATE.append(S)
        # P_N.append(p_n)
        # P_S.append(p_s)
        # NUM_A.append(num_angles)
    
        if e % 100 == 0:
            print("episode", e)
            print("score", score, "entropy", entropy.item())
    
        if e % 1000 == 0:
            torch.save(model.state_dict(), 'actor_critic_{}_{}'.format(e, note))
    
        
        # episode_rewards.append(score)
  
    
    # np.save("episode_rewards_{}_{}.npy".format(args.NUM_EPISODES, note),episode_rewards)
    # np.save("P_A_{}_{}.npy".format(args.NUM_EPISODES, note),P_A)
    # np.save("Num_P_{}_{}.npy".format(args.NUM_EPISODES, note), NUM)    
    # np.save("Actions_{}_{}.npy".format(args.NUM_EPISODES, note),A)  
    # np.save("States_{}_{}.npy".format(args.NUM_EPISODES, note),STATE)  
    # np.save("PN_{}_{}.npy".format(args.NUM_EPISODES, note),P_N)  
    # np.save("Num_A_{}_{}.npy".format(args.NUM_EPISODES, note), NUM_A)    
    
    
if __name__ == '__main__':
    main()
    