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
import matplotlib as mpl  
mpl.rc('font',family='Times New Roman')
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

def PPOtest(env_id,model,num_time,num_repeat,dirname):

    env = make_vec_env(env_id, n_envs=1)

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
    
    E_N_0_repeat = np.sum(E_N_0,axis=0)/num_time
    E_N_1_repeat = np.sum(E_N_1,axis=0)/num_time
    
    times = np.linspace(0, num_time * 0.01, num_time)
    
    width = 6
    height = 3
    fig = plt.figure(figsize=(width,height))
    ave_E_N_0_time = _ave(num_time, 10, E_N_0[:,num_repeat-1])
    ave_E_N_1_time = _ave(num_time, 10, E_N_1[:,num_repeat-1])
    plt.plot(times,E_N_0[:,num_repeat-1],linewidth=2,label='E_N_0',color='lightcoral',alpha=0.5) 
    # plt.plot(times,E_N_1[:,num_repeat-1],linewidth=2,label='E_N_1')
    plt.plot(times,ave_E_N_0_time,linewidth=2,label='ave E_N_0',color='maroon',alpha=0.8) 
    # plt.plot(times,ave_E_N_1_time,linewidth=2,label='ave E_N_1')
    # plt.legend(fontsize=12)
    # plt.xlabel('times',fontsize=18)
    plt.xticks(fontsize=25)
    plt.yticks([0,0.3,0.7],fontsize=25)
    plt.show()
    fig.savefig('E_N_testing.pdf',bbox_inches='tight')
    
    width = 6
    height = 3
    fig = plt.figure(figsize=(width,height))
    ave_E_N_0_repeat = _ave(num_repeat, 10, E_N_0_repeat)
    ave_E_N_1_repeat = _ave(num_repeat, 10, E_N_1_repeat)
    plt.plot(range(num_repeat),E_N_0_repeat,linewidth=2,label='E_N_0')
    # plt.plot(range(num_repeat),E_N_1_repeat,linewidth=2,label='E_N_1')
    plt.plot(range(num_repeat),ave_E_N_0_repeat,linewidth=2,label='ave E_N_0')
    # plt.plot(range(num_repeat),ave_E_N_1_repeat,linewidth=2,label='ave E_N_1')
    # plt.legend(fontsize=12)
    # plt.xlabel('num repeat',fontsize=18)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()

    
    ave_EN = np.sum(E_N_0_repeat)/num_repeat
    print('testing phase')
    print('ave E_N: '+str(ave_EN))
    print('stand deviation: '+str(np.std(E_N_0_repeat)))
    print('percent ave E_N: '+str(ave_EN/np.log(2)))
    print('percent stand deviation: '+str(np.std(E_N_0_repeat)/np.log(2)))
    
    ave_use_mean_current = _ave(num_time, 100, use_mean_current[:,num_repeat-1])   
    width = 6
    height = 3
    fig = plt.figure(figsize=(width,height))
    plt.plot(0.01*np.arange(0,num_time,1),use_mean_current[:,num_repeat-1],color='cornflowerblue',linewidth=2,alpha=0.5)  
    plt.plot(0.01*np.arange(0,num_time,1),ave_use_mean_current,color='navy',linewidth=2,alpha=0.8) 
    # plt.xlabel('times',fontsize=16)
    # plt.ylabel('mean current',fontsize=16)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    plt.show()
    fig.savefig('use_mean_current_testing.pdf',bbox_inches='tight')
    
if __name__ == '__main__':
    ###### training phase ################
    EN0Ep = np.load('./dataImp/EN0Ep.npy')
    EN1Ep = np.load('./dataImp/EN1Ep.npy')
    EN = EN0Ep[-1-10:-1]
    print('training phase')
    print('ave E_N: '+str(np.mean(EN)))
    print('stand deviation: '+str(np.std(EN)))  
    print('percent ave E_N: '+str(np.mean(EN)/np.log(2)))
    print('percent stand deviation: '+str(np.std(EN)/np.log(2)))
    
    numEpisode = 3000
    output_interval = 20
    ave_EN0_ep = _ave(numEpisode, 100, EN0Ep)
    ave_EN1_ep = _ave(numEpisode, 100, EN1Ep)
    
    width = 6
    height = 3
    fig = plt.figure(figsize=(width,height))
    plt.plot(np.arange(0,numEpisode,1),EN0Ep[:numEpisode],color='lightcoral',linewidth=2,alpha=0.5)    
    # plt.plot(np.arange(0,numEpisode,1),EN1Ep[:numEpisode],linewidth=2,alpha=0.5)
    plt.plot(np.arange(0,numEpisode,1),ave_EN0_ep,color='maroon',linewidth=2,alpha=0.8) 
    # plt.plot(np.arange(0,numEpisode,1),ave_EN1_ep,linewidth=2) 
    # plt.xlabel('episode',fontsize=16)
    # plt.ylabel('E_N',fontsize=16)
    plt.yticks([0,0.3,0.5],fontsize=25)
    plt.xticks([0,1000,2000,3000],fontsize=25)
    plt.show()
    fig.savefig('E_N_training.pdf',bbox_inches='tight')
    
    numEpisode_minus = numEpisode - output_interval
    reward_env = np.load('./dataImp/reward_env_'+str(numEpisode_minus)+'.npy')
    ave_reward_env = _ave(numEpisode_minus, 100, reward_env)
    width = 6
    height = 3
    fig = plt.figure(figsize=(width,height))
    plt.plot(np.arange(1,numEpisode_minus,1),reward_env[1:],color='cornflowerblue',linewidth=2,alpha=0.5) 
    plt.plot(np.arange(1,numEpisode_minus,1),ave_reward_env[1:],color='navy',linewidth=2,alpha=0.8) 
    # plt.xlabel('episode',fontsize=16)
    # plt.ylabel('mean reward over envs',fontsize=16)
    plt.yticks([-0.8,-0.6],fontsize=25)
    plt.xticks([0,1000,2000,3000],fontsize=25)
    plt.show()
    fig.savefig('reward_training.pdf',bbox_inches='tight')
    ######## testing phase ################
    num_repeat = 10
    env_id = "quantumEnvMeasureFilter-v0"
    
    model = PPO.load("./ppo_quantumOptomechanics")
    model.save("./dataImp/ppo_quantumOptomechanics")  
   
    n_steps = 1500#np.load('./train_par/n_steps.npy')
    num_time = n_steps - 2
    np.save('./train_par/n_steps.npy',n_steps)
    
    dirname = './test_phase'
    update_dir(dirname)
    
    # PPOtest(env_id,model,num_time,num_repeat,dirname)
    plot(dirname,num_time,num_repeat)
    