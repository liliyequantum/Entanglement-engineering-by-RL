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
  
    ######### parameter transfer from slurm script ##################
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
    
    ########## cpu or gpu ###################
    device = 'cpu'
        
    ######### system parameter ##############
    N = 10  # N Fock basis
    g_0 = 0.2
    kappa = 0.1 # Cavity damping rate
    target_idx = 760
    
    ######### training parameter
    n_envs = 5
    n_episode = 2000
    n_steps = 500 # times
    n_update = 5 # update neural network
    output_interval = 20 # plot reward...
    repeat_index = 1
    
    learning_rate = 1e-3/2
    n_epochs = 10
    batch_size = int(n_steps*n_update*n_envs/10)
    
    ####### dir  #######
    rundir = './N_'+str(N)
    update_dir(rundir)
    rundir = rundir + '/g0_'+str(g_0)
    update_dir(rundir)
    rundir = rundir + '/kappa_'+str(kappa)
    update_dir(rundir)
    rundir = rundir +'/learnRate_'+str(learning_rate)
    update_dir(rundir)
    rundir = rundir + '/'+str(repeat_index)
    update_dir(rundir)
        
    dirname = ['dataImp','picture','train_par']
    for i in range(len(dirname)):
        update_dir(rundir+'/'+dirname[i])
      
    np.save(rundir+'/train_par/N.npy',N)
    np.save(rundir+'/train_par/g0.npy',g_0)
    np.save(rundir+'/train_par/kappa.npy',kappa)
    np.save(rundir+'/train_par/n_steps.npy',n_steps)
    np.save(rundir+'/train_par/n_episode.npy',n_episode)
    np.save(rundir+'/train_par/output_interval.npy',output_interval)
    np.save(rundir+'/train_par/n_envs.npy',n_envs)
    np.save(rundir+'/train_par/learning_rate.npy',learning_rate)
    
    env_kwargs = {'rundir':rundir,'N':N,'g_0':g_0,'kappa':kappa,\
                  'n_steps':n_steps,'n_episode':n_episode,\
                  'output_interval':output_interval,'n_envs':n_envs,'target_idx':target_idx}
    env_id = "quantum_nonlinear_obser_Pn-v0"
    env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv,\
                       monitor_dir=rundir+'/monitor',env_kwargs=env_kwargs)
    
    policy_kwargs = dict(activation_fn=th.nn.ReLU,\
                          net_arch=[dict(vf=[256, 128, 64], pi=[256, 128, 64])])
        
    model = RecurrentPPO("MlpLstmPolicy", env,policy_kwargs = policy_kwargs,\
                learning_rate=learning_rate, n_steps = n_steps*n_update, \
                    batch_size = batch_size, n_epochs = n_epochs,\
                    verbose=1,device=device)
    
    start = time.time()
    model.learn(total_timesteps = n_steps * n_episode * n_envs)
    end = time.time()
    print('training time elapsed: ',end-start)
                
    model.save(rundir+'/dataImp/ppo_quantumOptomechanics')
    del model    
    
    sys.exit()
   