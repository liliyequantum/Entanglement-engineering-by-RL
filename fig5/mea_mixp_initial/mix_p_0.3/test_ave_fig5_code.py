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

def PPOtest(env_id,model,num_time,num_repeat,dirname,\
            measure_rate,n_steps,n_envs,\
            n_episode,mean_interval,ntraj,filter_interval,gaussian_var,output_interval,rundir,kappa,mix_p):

    env_kwargs = {'measure_rate':measure_rate,'n_steps':n_steps,'n_envs':n_envs,\
                  'n_episode':n_episode,'mean_interval':mean_interval,'ntraj':ntraj,\
                   'filter_interval':filter_interval,'gaussian_var':gaussian_var,\
                       'output_interval':output_interval,'rundir':rundir,\
                   'kappa':kappa,'mix_p':mix_p}
    env = make_vec_env(env_id, n_envs=n_envs,env_kwargs=env_kwargs)

    use_mean_current = np.zeros((num_time,num_repeat))
    E_N_0 = np.zeros((num_time,num_repeat))
    E_N_1 = np.zeros((num_time,num_repeat))
    G = np.zeros((num_time,num_repeat)) # driven amplitude
   
    for j in tqdm(range(num_repeat)): 
        Obs = env.reset() 
        for i in range(num_time):
            Action, _states = model.predict(Obs)
            Obs, Rewards, done, info = env.step(Action)
            
            use_mean_current[i,j] = Obs[0][0] # state1
            G[i,j] = Action[0][0] # action
            
            E_N_0[i,j] = info[0][0]
            E_N_1[i,j] = info[0][1]
     
    np.save(dirname+'/use_mean_current.npy', use_mean_current)
    np.save(dirname+'/E_N_0.npy', E_N_0)
    np.save(dirname+'/E_N_1.npy', E_N_1)
    np.save(dirname+'/G.npy', G)
    
def plot(dirname,num_time,num_repeat):
    
    use_mean_current = np.load(dirname+'/use_mean_current.npy')
    E_N_0 = np.load(dirname+'/E_N_0.npy')
    E_N_1 = np.load(dirname+'/E_N_1.npy')
    G = np.load(dirname+'/G.npy')
    
    num_time = num_time - 2
    use_mean_current = use_mean_current[:-2,:]
    E_N_0 = E_N_0[:-2,:]
    E_N_1 = E_N_1[:-2,:]
    G = G[:-2,:]
    
    E_N_0_repeat = np.sum(E_N_0,axis=0)/num_time
    E_N_1_repeat = np.sum(E_N_1,axis=0)/num_time
    
    times = np.linspace(0, num_time * 0.01, num_time)
    
    plt.figure()
    ave_E_N_0_time = _ave(num_time, 10, E_N_0[:,num_repeat-1])
    ave_E_N_1_time = _ave(num_time, 10, E_N_1[:,num_repeat-1])
    plt.plot(times,E_N_0[:,num_repeat-1],linewidth=3,label='E_N_0')
    plt.plot(times,E_N_1[:,num_repeat-1],linewidth=3,label='E_N_1')
    plt.plot(times,ave_E_N_0_time,linewidth=3,label='ave E_N_0')
    plt.plot(times,ave_E_N_1_time,linewidth=3,label='ave E_N_1')
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
    print('testing phase')
    print('ave E_N: '+str(ave_EN))
    print('stand deviation: '+str(np.std(E_N_0_repeat)))
    print('percent ave E_N: '+str(ave_EN/0.7))
    print('percent stand deviation: '+str(np.std(E_N_0_repeat)/0.7))
    
    ave_use_mean_current = _ave(num_time, 100, use_mean_current[:,num_repeat-1])   
    plt.figure()
    plt.plot(0.01*np.arange(0,num_time,1),use_mean_current[:,num_repeat-1],linewidth=3) 
    plt.plot(0.01*np.arange(0,num_time,1),ave_use_mean_current,linewidth=3) 
    plt.xlabel('times',fontsize=16)
    plt.ylabel('mean current',fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.show()
    
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
   
    ave_EN0_ep = _ave(numEpisode-output_interval, 100, EN0Ep[:(numEpisode-output_interval)])
    ave_EN1_ep = _ave(numEpisode-output_interval, 100, EN1Ep[:(numEpisode-output_interval)])
    
    plt.figure()
    plt.plot(np.arange(0,numEpisode-output_interval,1),EN0Ep[:(numEpisode-output_interval)],linewidth=3)    
    plt.plot(np.arange(0,numEpisode-output_interval,1),EN1Ep[:(numEpisode-output_interval)],linewidth=3)
    plt.plot(np.arange(0,numEpisode-output_interval,1),ave_EN0_ep,linewidth=3) 
    plt.plot(np.arange(0,numEpisode-output_interval,1),ave_EN1_ep,linewidth=3) 
    plt.xlabel('episode',fontsize=16)
    # plt.ylabel('E_N',fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.show()
    
    numEpisode_minus = numEpisode - output_interval
    reward_env = np.load('./dataImp/reward_env_'+str(numEpisode_minus)+'.npy')
    ave_reward_env = _ave(numEpisode_minus, 100, reward_env)
    plt.figure()
    plt.plot(np.arange(0,numEpisode_minus,1),reward_env,linewidth=3) 
    plt.plot(np.arange(0,numEpisode_minus,1),ave_reward_env,linewidth=3) 
    plt.xlabel('episode',fontsize=16)
    plt.ylabel('mean reward over envs',fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.show()
    ######## testing phase ################
    num_repeat = 10
    env_id = "quantumEnvMeasureFilter_update-v0"
    
    model = PPO.load("./ppo_quantumOptomechanics")
    model.save("./dataImp/ppo_quantumOptomechanics")  
   
    dirname = './test_phase'
    update_dir(dirname)
    
    ## important #####
    measure_rate = 0.5
    n_steps = 500
    num_time = n_steps - 2
    n_envs = 1
    ntraj = 5
    kappa = 0.01
    mix_p = 0.3
    mean_interval = 5
    filter_interval = 10
    gaussian_var = 6
    
    ## not important ######
    output_interval = 50
    
    # PPOtest(env_id,model,num_time,num_repeat,dirname,\
    #             measure_rate,num_time,n_envs,\
    #             numEpisode,mean_interval,ntraj,filter_interval,gaussian_var,\
    #                 output_interval,dirname,kappa,mix_p)
    plot(dirname,num_time,num_repeat)
    