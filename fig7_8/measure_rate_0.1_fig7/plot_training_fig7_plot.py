# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:02:26 2023

@author: liliye
"""

import numpy as np
import matplotlib.pyplot as plt

def _ave(n_steps, ave_step, data_array):
    measure4 = np.array([])
    ave_data= np.zeros(n_steps,)
    for i in range(n_steps):
        if len(measure4)>=ave_step:
            measure4 = np.delete(measure4,0)
        measure4 = np.append(measure4, data_array[i])
        ave_data[i] = np.sum(measure4)/np.size(measure4)
    return ave_data 

if __name__ == '__main__':
    NIdx = 2250
    Nend = 2450
    
    t = np.linspace(0,5,499)
    alpha_L = np.load('./dataImp/alpha_L_'+str(NIdx)+'.npy')
    Delta = np.load('./dataImp/Delta_'+str(NIdx)+'.npy')
    
    fontsize_define = 16
    
    plt.figure(figsize=(8,2))
    plt.plot(t,Delta,linewidth=3,label=r'$\Delta$') 
    plt.plot(t,alpha_L,linewidth=3,label=r'$\alpha_L$') 
    # plt.legend(fontsize=fontsize_define)
    plt.xlabel(r'$t(\omega_{m}^{-1})$',fontsize=fontsize_define)
    plt.ylabel('Laser control',fontsize=fontsize_define)
    plt.yticks(fontsize=fontsize_define)
    plt.xticks(fontsize=fontsize_define)
    plt.savefig('./laser_control.pdf',bbox_inches='tight')
    
    E_N_0 = np.load('./dataImp/EN0_step_'+str(NIdx)+'.npy')
    E_N_1 = np.load('./dataImp/EN1_step_'+str(NIdx)+'.npy')
    
    plt.figure(figsize=(8,2))
    plt.plot(t,E_N_0,linewidth=3) 
    plt.plot(t,E_N_1,linewidth=3) 
    plt.xlabel(r'$t(\omega_{m}^{-1})$',fontsize=fontsize_define)
    plt.ylabel(r'$E_N(t)$',fontsize=fontsize_define)
    plt.yticks(fontsize=fontsize_define)
    plt.xticks(fontsize=fontsize_define)
    plt.savefig('./EN.pdf',bbox_inches='tight')
    
    Pn = np.load('./dataImp/Pn_step_'+str(NIdx)+'.npy')
    Mn = np.load('./dataImp/Mn_step_'+str(NIdx)+'.npy')
    
    plt.figure()
    plt.plot(t,Pn,linewidth=3,label=r'$\langle n_p\rangle$') 
    plt.plot(t,Mn,linewidth=3,label=r'$\langle n_m\rangle$')
    plt.legend(fontsize=fontsize_define) 
    plt.xlabel(r'$t(\omega_{m}^{-1})$',fontsize=fontsize_define)
    plt.ylabel('Exp current',fontsize=fontsize_define)
    plt.yticks(fontsize=fontsize_define)
    plt.xticks(fontsize=fontsize_define)
    plt.savefig('./exp_current.pdf',bbox_inches='tight')
    
    reward = np.load('./dataImp/reward_env_'+str(Nend)+'.npy')
    ave_reward = _ave(np.size(reward), 100, reward)
    plt.figure(figsize=(8,2))
    plt.plot(np.arange(0,np.size(reward),1),reward,color='skyblue',linewidth=3) 
    plt.plot(np.arange(0,np.size(reward),1),ave_reward,color='tab:blue',linewidth=3) 
    plt.xlabel('Episode',fontsize=fontsize_define)
    plt.ylabel(r'$\widetilde{R}$',fontsize=fontsize_define)
    plt.yticks(fontsize=fontsize_define)
    plt.xticks(np.arange(0,3000,500),fontsize=fontsize_define)
    plt.savefig('./reward_over_envs.pdf',bbox_inches='tight')
    
    E_N_0_Ep = np.load('./dataImp/EN0Ep'+str(Nend)+'.npy')
    E_N_1_Ep = np.load('./dataImp/EN1Ep'+str(Nend)+'.npy')
    ave_E_N_0_Ep = _ave(np.size(E_N_0_Ep), 100, E_N_0_Ep)
    ave_E_N_1_Ep = _ave(np.size(E_N_1_Ep), 100, E_N_1_Ep)
    plt.figure(figsize=(8,2))
    plt.plot(np.arange(0,np.size(reward),1),E_N_0_Ep[0:np.size(reward)],color='bisque',linewidth=3) 
    # plt.plot(np.arange(0,np.size(reward),1),E_N_1_Ep[0:np.size(reward)],linewidth=3) 
    plt.plot(np.arange(0,np.size(reward),1),ave_E_N_0_Ep[0:np.size(reward)],color='darkorange',linewidth=3) 
    # plt.plot(np.arange(0,np.size(reward),1),ave_E_N_1_Ep[0:np.size(reward)],linewidth=3) 
    plt.xlabel('Episode',fontsize=fontsize_define)
    plt.ylabel(r'$\widetilde{E}_N$',fontsize=fontsize_define)
    plt.yticks(fontsize=fontsize_define)
    plt.xticks(np.arange(0,3000,500),fontsize=fontsize_define)
    plt.savefig('./ENEp.pdf',bbox_inches='tight')
    
