# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:14:24 2023

@author: liliye
"""

import gym
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# import gym_SME
import time
import sys
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

def _ave(n_steps, ave_step, data_array):
    measure4 = np.array([])
    ave_data= np.zeros(n_steps,)
    for i in range(n_steps):
        if len(measure4)>=ave_step:
            measure4 = np.delete(measure4,0)
        measure4 = np.append(measure4, data_array[i])
        ave_data[i] = np.sum(measure4)/np.size(measure4)
    return ave_data 

def update_dir(dirname):
    if not os.path.exists(dirname):
      os.mkdir(dirname)
      print("Folder %s created!" % dirname)
    else:
      print("Folder %s already exists" % dirname)

def plot(dirname,num_time,num_repeat):
    
    use_exp_current = np.load(dirname+'/use_exp_current.npy')
    E_N_0 = np.load(dirname+'/E_N_0.npy')
    E_N_1 = np.load(dirname+'/E_N_1.npy')
    Delta = np.load(dirname+'/Delta.npy')
    Alpha_L = np.load(dirname+'/Alpha_L.npy')
    
    num_time = num_time - 2
    use_exp_current = use_exp_current[:-2,:]
    E_N_0 = E_N_0[:-2,:]
    E_N_1 = E_N_1[:-2,:]
    Delta = Delta[:-2,:]
    Alpha_L = Alpha_L[:-2,:]
    
    E_N_0_repeat = np.sum(E_N_0,axis=0)/num_time
    E_N_1_repeat = np.sum(E_N_1,axis=0)/num_time
    
    times = np.linspace(0, num_time * 0.01, num_time)
    
    plt.figure()
    plt.plot(times,E_N_0[:,num_repeat-1],linewidth=3,label='E_N_0')
    plt.plot(times,E_N_1[:,num_repeat-1],linewidth=3,label='E_N_1')
    plt.legend(fontsize=12)
    plt.xlabel('times',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    plt.figure()
    ave_E_N_0_repeat = _ave(num_repeat, 10, E_N_0_repeat)
    ave_E_N_1_repeat = _ave(num_repeat, 10, E_N_1_repeat)
    plt.plot(range(num_repeat),E_N_0_repeat,linewidth=3,label='E_N_0')
    plt.plot(range(num_repeat),E_N_1_repeat,linewidth=3,label='E_N_1')
    plt.plot(range(num_repeat),ave_E_N_0_repeat,linewidth=3,label='ave E_N_0')
    plt.plot(range(num_repeat),ave_E_N_1_repeat,linewidth=3,label='ave E_N_1')
    plt.legend(fontsize=12)
    plt.xlabel('num repeat',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    
    ave_EN = np.sum(E_N_0_repeat)/num_repeat
    print('random control')
    print('ave E_N: '+str(ave_EN))
    print('stand deviation: '+str(np.std(E_N_0_repeat)))
    print('percent ave E_N: '+str(ave_EN/np.log(2)))
    print('percent stand deviation: '+str(np.std(E_N_0_repeat)/np.log(2)))
    random_ave_EN = ave_EN/np.log(2)
    random_std_EN = np.std(E_N_0_repeat)/np.log(2)
    
    plt.figure()
    plt.plot(0.01*np.arange(0,num_time,1),use_exp_current[:,num_repeat-1],linewidth=3) 
    plt.xlabel('times',fontsize=16)
    plt.ylabel('exp current',fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.show()
    
    return random_ave_EN, random_std_EN
    
    
      
if __name__ == '__main__':
    
    num_repeat = 10
 
    N = 10  # N Fock basis
    g_0 = 0.2
    kappa = 0.1 # Cavity damping rate
    n_envs = 1
    n_episode = 2500
    n_steps = 500 # times
    output_interval = 50 # plot reward...
    
    num_time = n_steps - 2
    Delta = np.random.uniform(-5,5,(num_time,num_repeat)) # driven detuning
    Alpha_L = np.random.uniform(-5,5,(num_time,num_repeat)) # driven amplitude
    
    max_measure_rate = [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
    max_target_idx = [2100, 4300, 1000, 1400, 1100, 2600]
    
    random_ave_E_N = np.zeros(len(max_measure_rate),)
    random_std_E_N = np.zeros(len(max_measure_rate),)
    
    for i in range(0,len(max_measure_rate),1):
        if i==0:
            rundir = './measure_rate_0.05'
        elif i==1:
            rundir = './measure_rate_0.1'
        elif i==2:
            rundir = './measure_rate_0.3'
        elif i==3:
            rundir = './measure_rate_0.5'
        elif i==4:
            rundir = './measure_rate_0.7'
        elif i==5:
            rundir = './measure_rate_1.0'
        
   
        # measure_rate = max_measure_rate[i]
        # target_idx = max_target_idx[i]
        
        # env_kwargs = {'rundir':rundir,'N':N,'g_0':g_0,'kappa':kappa,'measure_rate':measure_rate,\
        #               'n_steps':n_steps,'n_episode':n_episode,\
        #               'output_interval':output_interval,'n_envs':n_envs,\
        #               'target_idx':target_idx}
        # env_id = "quantumNonlinearEnvSMEExpPn-v0"
        # env = make_vec_env(env_id, n_envs=n_envs,env_kwargs=env_kwargs)
    
        # use_exp_current = np.zeros((num_time,num_repeat))
        # E_N_0 = np.zeros((num_time,num_repeat))
        # E_N_1 = np.zeros((num_time,num_repeat))
       
        # model = PPO.load(rundir+'/ppo_quantumOptomechanics')
        # for j in range(num_repeat): 
        #     Obs = env.reset() 
        #     for i in tqdm(range(num_time)):
        #         Action, _states = model.predict(Obs)
        #         Action[0][0] = Delta[i,j]
        #         Action[0][1] = Alpha_L[i,j]
        #         Obs, Rewards, done, info = env.step(Action)
                
        #         use_exp_current[i,j] = Obs[0][0] # state1
        #         E_N_0[i,j] = info[0][1]
        #         E_N_1[i,j] = info[0][0]
              
        dirname = rundir+'/random_control'
        # update_dir(dirname)
        # np.save(dirname+'/use_exp_current.npy', use_exp_current)
        # np.save(dirname+'/E_N_0.npy', E_N_0)
        # np.save(dirname+'/E_N_1.npy', E_N_1)
        # np.save(dirname+'/Delta.npy', Delta)
        # np.save(dirname+'/Alpha_L.npy', Alpha_L)
        
        [random_ave_E_N[i], random_std_E_N[i]] = plot(dirname,num_time,num_repeat)
        
    dirname = './random_perform'
    update_dir(dirname)
    np.save(dirname+'/random_ave_E_N.npy',random_ave_E_N)
    np.save(dirname+'/random_std_E_N.npy',random_std_E_N)