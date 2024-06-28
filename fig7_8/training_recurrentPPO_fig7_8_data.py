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

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from qutip import *
import os
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
    # parser.add_argument('measure_rate',
    #                     help='measure_rate',
    #                     type=float)
    # parser.add_argument('target_idx',
    #                     help='target_idx',
    #                     type=float)
    # args = parser.parse_args()
    # repeat_index = args.repeat_index
    # measure_rate = args.measure_rate
    # target_idx = int(args.target_idx)
    
    repeat_index = 1
    measure_rate = 0.1
    target_idx = 4300
    
    ########## cpu or gpu ###################
    device = 'cpu'
        
    ######### system parameter ##############
    N = 10  # N Fock basis
    g_0 = 0.2
    kappa = 0.1 # Cavity damping rate
    #measure_rate = 0.1
    #target_idx = 820
    
    ######### training parameter
    n_envs = 1
    n_episode = 2500
    n_steps = 500 # times
    n_update = 5 # update neural network
    output_interval = 50 # plot reward...
    #repeat_index = 1
    
    learning_rate = 1e-3/2
    n_epochs = 10
    batch_size = int(n_steps*n_update*n_envs/10)#n_steps*n_update*n_envs#
    
    ####### dir  #######
    rundir = './measure_rate_'+str(measure_rate)+'_repeatIdx_'+str(repeat_index)
    update_dir(rundir)
    
    dirname = ['dataImp','picture','train_par']
    for i in range(len(dirname)):
        update_dir(rundir+'/'+dirname[i])
      
    np.save(rundir+'/train_par/N.npy',N)
    np.save(rundir+'/train_par/g0.npy',g_0)
    np.save(rundir+'/train_par/kappa.npy',kappa)
    np.save(rundir+'/train_par/measure_rate.npy',measure_rate)
    np.save(rundir+'/train_par/n_steps.npy',n_steps)
    np.save(rundir+'/train_par/n_episode.npy',n_episode)
    np.save(rundir+'/train_par/output_interval.npy',output_interval)
    np.save(rundir+'/train_par/n_envs.npy',n_envs)
    np.save(rundir+'/train_par/learning_rate.npy',learning_rate)
    
    env_kwargs = {'rundir':rundir,'N':N,'g_0':g_0,'kappa':kappa,'measure_rate':measure_rate,\
                  'n_steps':n_steps,'n_episode':n_episode,\
                  'output_interval':output_interval,'n_envs':n_envs,\
                  'target_idx':target_idx}
    env_id = "quantumNonlinearEnvSMEExpPn-v0"
    env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv,\
                       monitor_dir=rundir+'/monitor',env_kwargs=env_kwargs)
    
    policy_kwargs = dict(activation_fn=th.nn.ReLU,\
                          net_arch=dict(vf=[256, 128, 64], pi=[256, 128, 64]))
        
    model = RecurrentPPO("MlpLstmPolicy", env,policy_kwargs = policy_kwargs,\
                learning_rate=learning_rate, n_steps = n_steps*n_update, \
                    batch_size = batch_size, n_epochs = n_epochs,\
                    verbose=1,device=device)
    
    start = time.time()
    model.learn(total_timesteps = n_steps * n_episode * n_envs)
    end = time.time()
    print('training time elapsed: ',end-start)
                
    model.save(rundir+'/ppo_quantumOptomechanics')
    model.save(rundir+'/dataImp/ppo_quantumOptomechanics')
    del model    
    
    sys.exit()
   