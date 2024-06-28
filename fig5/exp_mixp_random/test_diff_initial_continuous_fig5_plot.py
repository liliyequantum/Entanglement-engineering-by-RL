# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:51:20 2022

@author: LiliYe
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
from qutip import *
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

def PPOtest(env_id,model,num_time,num_repeat,dirname,measure_rate,n_steps,n_envs,\
            n_episode,ntraj,output_interval,rundir,kappa,mix_p):

    env_kwargs = {'measure_rate':measure_rate,'n_steps':n_steps,'n_envs':n_envs,\
                  'n_episode':n_episode,'ntraj':ntraj,\
                   'output_interval':output_interval,'rundir':rundir,\
                   'kappa':kappa,'mix_p':mix_p}
    env = make_vec_env(env_id, n_envs=n_envs,env_kwargs=env_kwargs)

    numphoton = np.zeros((num_time,num_repeat))
    E_N_0 = np.zeros((num_time,num_repeat))
    E_N_1 = np.zeros((num_time,num_repeat))
    G = np.zeros((num_time,num_repeat)) # driven amplitude
   
    for j in range(num_repeat): 
        Obs = env.reset() 
        for i in range(num_time):
            Action, _states = model.predict(Obs)
            Obs, Rewards, done, info = env.step(Action)
            
            numphoton[i,j] = Obs[0][0] # state1
            G[i,j] = Action[0][0] # action
            
            E_N_0[i,j] = info[0][0]
            E_N_1[i,j] = info[0][1]
     
    np.save(dirname+'/numphoton.npy', numphoton)
    np.save(dirname+'/E_N_0.npy', E_N_0)
    np.save(dirname+'/E_N_1.npy', E_N_1)
    np.save(dirname+'/G.npy', G)
    
if __name__ == '__main__':
    
    ###### training phase ################
    numEpisode = 3000
    output_interval = 50
    EN0Ep = np.load('./dataImp/EN0Ep'+str(numEpisode-output_interval)+'.npy')
    EN1Ep = np.load('./dataImp/EN1Ep'+str(numEpisode-output_interval)+'.npy')
    EN = EN0Ep[-output_interval-10:-output_interval]
    print('training phase')
    print('ave E_N: '+str(np.mean(EN)))
    print('stand deviation: '+str(np.std(EN)))  
    print('percent ave E_N: '+str(np.mean(EN)/0.7))
    print('percent stand deviation: '+str(np.std(EN)/0.7))
    
    ######## testing phase ################
  
    
    num_repeat = 10
    env_id = "quantumEnvExp_update-v0"
    
    model = PPO.load("./ppo_quantumOptomechanics")
    model.save("./dataImp/ppo_quantumOptomechanics")  
    
    dirname = './test_phase_diff_initial_continuous'
    update_dir(dirname)
    
    ## important #####
    measure_rate = 0.5
    n_steps = 500
    num_time = n_steps - 5
    n_envs = 1
    ntraj = 1
    kappa = 0.01
    
    num_Mix_p = 100
    Mix_p = np.linspace(0, 1, num_Mix_p)
    
    ''' testing process
    # percent_ave_EN = np.zeros(num_Mix_p,)
    # std_EN = np.zeros(num_Mix_p,)
    # for i in tqdm(range(num_Mix_p)):
        
    #     PPOtest(env_id,model,num_time,num_repeat,dirname,measure_rate,num_time,n_envs,\
    #                 numEpisode,ntraj,output_interval,dirname,kappa,Mix_p[i])
        
    #     E_N_0 = np.load(dirname+'/E_N_0.npy')
    #     E_N_0_repeat = np.sum(E_N_0,axis=0)/num_time
    #     ave_EN = np.sum(E_N_0_repeat)/num_repeat
        
    #     percent_ave_EN[i] = ave_EN/0.7
    #     std_EN[i] = np.std(E_N_0_repeat)/0.7
    
    # np.save(dirname+'/percent_ave_EN.npy',percent_ave_EN)
    # np.save(dirname+'/std_EN.npy',std_EN)
    '''
    
    percent_ave_EN = np.load(dirname+'/percent_ave_EN.npy')
    std_EN = np.load(dirname+'/std_EN.npy')
    fig = plt.figure()
    plt.plot(Mix_p, percent_ave_EN,'-',color='navy')
    plt.fill_between(Mix_p, percent_ave_EN-std_EN, percent_ave_EN+std_EN,color='cornflowerblue')
    plt.xlabel(r'$p$',fontsize=16)
    plt.ylabel(r'$E_{N}/E_0(\%)$',fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylim([0,1])
    plt.show()
    fig.savefig(dirname+'/test_diff_initial_continuous.pdf',bbox_inches='tight')
    
    