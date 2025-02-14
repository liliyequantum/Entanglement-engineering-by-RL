# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:33:56 2022

@author: LiliYe
"""

import numpy as np
import matplotlib.pyplot as plt
import torch as th
#https://pytorch.org/get-started/locally/ install pytroch on cuda

import time
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from qutip import *
import os

## notic for cluster server
# uncomment plt.switch_backend('agg')

import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import sys

import gym
from gym import spaces, logger

from qutip import *
from numpy import genfromtxt #read excel for parallel env
from scipy.ndimage import gaussian_filter1d
import argparse

def update_dir(dirname):
    if not os.path.exists(dirname):
      os.mkdir(dirname)
      print("Folder %s created!" % dirname)
    else:
      print("Folder %s already exists" % dirname)
    
if __name__ == '__main__':
    
    # ######### parameter transfer from slurm script ##################
    # parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    # parser.add_argument('repeat_index',
    #                     help='repeat index',
    #                     type=int)
    # parser.add_argument('learning_rate',
    #                     help='learning_rate',
    #                     type=float)
    # args = parser.parse_args()
    
    # repeat_index = args.repeat_index
    # learning_rate = args.learning_rate
    repeat_index = 1
    learning_rate = 1e-4
    
    ####### cpu or gpu  #######
    device = 'cpu'#'cuda
    
    ####### system parameter  #######
    measure_rate = 0.5
    kappa = 0.01 # decay rate of photon mode
    
    ####### training parameter #######
    n_steps = 500 # times
    n_update = 5 # update neural network
    n_envs = 5
    n_episode = 3000
    #repeat_index = 2
    
    mean_interval = 5 # mean current window steps
    ntraj = 5
    filter_interval = 10
    gaussian_var = 6 # variance of Gaussian filter
      
    #learning_rate = 1e-4
    n_epochs = 10
    batch_size = int(n_steps*n_update*n_envs/10)
    
    ####### output parameter  #######
    output_interval = 50 # plot reward...
    
    ####### dir  #######
    rundir = './n_envs_'+str(n_envs)
    update_dir(rundir)
    rundir = rundir + '/mix_p_random'
    update_dir(rundir)
    rundir = rundir + '/kappa_'+str(kappa)
    update_dir(rundir)
    rundir = rundir + '/measureRate_'+str(measure_rate)
    update_dir(rundir)
    rundir = rundir + '/learnRate_'+str(learning_rate)
    update_dir(rundir)
    rundir = rundir + '/'+str(repeat_index)
    update_dir(rundir)
    
    dirname = ['dataImp','picture']
    for i in range(len(dirname)):
        update_dir(rundir+'/'+dirname[i])
      
    ####### env  #######
    env_kwargs = {'measure_rate':measure_rate,'n_steps':n_steps,'n_envs':n_envs,\
                  'n_episode':n_episode,'mean_interval':mean_interval,'ntraj':ntraj,\
                   'filter_interval':filter_interval, 'gaussian_var':gaussian_var,\
                   'output_interval':output_interval,'rundir':rundir,\
                   'kappa':kappa}
        
    env_id = "quantumEnvMeasureFilter_randomInitial-v0"
    env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv,monitor_dir=rundir+'/monitor',env_kwargs=env_kwargs)
    
    ####### PPO  #######
    policy_kwargs = dict(activation_fn=th.nn.ReLU,\
                          net_arch=dict(vf=[256, 128, 64], pi=[256, 128, 64]))
        
    model = PPO("MlpPolicy", env,policy_kwargs = policy_kwargs,\
                learning_rate=learning_rate, n_steps = n_steps*n_update, \
                    batch_size = batch_size, n_epochs = n_epochs,\
                    verbose=1,device=device)
    
    start = time.time()
    model.learn(total_timesteps = n_steps * n_episode * n_envs)
    end = time.time()
    print('training time elapsed: ',end-start)
                
    model.save(rundir+"/ppo_quantumOptomechanics")
    model.save(rundir+"/dataImp/ppo_quantumOptomechanics")
    del model    
    
    sys.exit()