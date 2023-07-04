# SeqAngleRL
This repository contains code for the Sequential Experimental Design for X-ray CT using Deep Reinforcement Learning (SeqAngleRL) project.

## Project Overview
The SeqAngleRL project aims to develop a framework for sequential experimental design in X-ray CT using deep reinforcement learning techniques. The goal is to optimize the selection of X-ray angles in a sequential manner during the imaging process, ultimately improving the image quality while minimizing acquisition time. We examine in intuitive numerical experiments whether the learned policies are able to sequentially adapt the scan angles to the object (a-posteriori adaptation). For this, we use various simple numerical phantoms for which the informative angles are well-known. Throughout our experiments, we focus on parallel-beam geometry and simple 2D tomography using synthetic data. 

## Structure
The repository has the following structure:

* SimulationData: This directory contains all the 2D synthetic data used for the experiments.

* Environment_endtoend.py: This file implements the environment with the end-to-end reward setting. The deep reinforcement learning agent interacts with this environment. It includes functionalities for forward projection, reconstruction using the SIRT algorithm, sequential angle selection, and reward calculation.

* Environment_increment.py: This file implements the environment with the increment reward setting. Similar to the previous file, it includes functionalities for forward projection, reconstruction, sequential angle selection, and reward calculation.

* Actor-Critic.py: This file contains the implementation of the encoder network and the main function. The main function is responsible for computing the gradient on the policy and updating the networks, following the actor-critic reinforcement learning algorithm.

## Getting Started
Clone the repository: 'git clone https://github.com/tianyuan1wang/SeqAngleRL.git'
Use environment.yml to set up the whole environment.

## Running the Code
Run the following command with hyperparameter settings in the terminal. You can change the values after these hyperparameters.
'python Actor_Critic.py --NUM_EPISODES 300000 --GAMMA 0.99 --LR 0.0001 --WD 0.00001 --Environment Environment_RL_endtoend.py --Note mix'
