# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:33:56 2022

@author: LiliYe
"""

import numpy as np
import matplotlib.pyplot as plt
import torch as th

import time
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from qutip import *
import os

def update_dir(dirname):
    if not os.path.exists(dirname):
      os.mkdir(dirname)
      print("Folder %s created!" % dirname)
    else:
      print("Folder %s already exists" % dirname)
    
if __name__ == '__main__':
  
    n_envs = 5
    n_episode = 3000
    n_steps = 500 # times
    n_update = 5 # update neural network
    output_interval = 20 # plot reward...
    
    measure_rate = 0.1
    ntraj = 1
    mix_p = 0
    
    learning_rate = 1e-3/2
    n_epochs = 10
    batch_size = int(n_steps*n_update*n_envs/10)
    
    dirname = ['dataImp','monitor','picture','train_par']
    for i in range(len(dirname)):
        update_dir(dirname[i])
      
    np.save('./train_par/n_steps.npy',n_steps)
    np.save('./train_par/n_episode.npy',n_episode)
    np.save('./train_par/output_interval.npy',output_interval)
    np.save('./train_par/measure_rate.npy',measure_rate)
    np.save('./train_par/ntraj.npy',ntraj)
    np.save('./train_par/n_envs.npy',n_envs)
    np.save('./train_par/mix_p.npy',mix_p)
    
    env_id = "quantumEnvExp-v0"
    env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv,monitor_dir='monitor')
    
    policy_kwargs = dict(activation_fn=th.nn.ReLU,\
                          net_arch=dict(vf=[256, 128, 64], pi=[256, 128, 64]))
        
    model = PPO("MlpPolicy", env,policy_kwargs = policy_kwargs,\
                learning_rate=learning_rate, n_steps = n_steps*n_update, \
                    batch_size = batch_size, n_epochs = n_epochs,\
                    verbose=1,device="cpu")
    
    start = time.time()
    model.learn(total_timesteps = n_steps * n_episode * n_envs)
    end = time.time()
    print('training time elapsed: ',end-start)
                
    model.save("./ppo_quantumOptomechanics")
    model.save("./dataImp/ppo_quantumOptomechanics")
    del model    
    
    sys.exit()
   