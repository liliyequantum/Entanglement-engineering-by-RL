# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:02:26 2023

@author: liliye
"""

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,5,499)
fontsize_define = 16
idx = 1160

alpha_L = np.load('./dataImp/alpha_L_'+str(idx)+'.npy')
Delta = np.load('./dataImp/Delta_'+str(idx)+'.npy')



plt.figure(figsize=(8,2))
plt.plot(t,Delta,linewidth=3,label=r'$\Delta$') 
plt.plot(t,alpha_L,linewidth=3,label=r'$\alpha_L$') 
plt.legend(fontsize=fontsize_define)
plt.xlabel(r'$t(\omega_{m}^{-1})$',fontsize=fontsize_define)
plt.ylabel('Laser control',fontsize=fontsize_define)
plt.yticks(fontsize=fontsize_define)
plt.xticks(fontsize=fontsize_define)
plt.savefig('./laser_control.pdf',bbox_inches='tight')

E_N_0 = np.load('./dataImp/EN0_step_'+str(idx)+'.npy')
E_N_1 = np.load('./dataImp/EN1_step_'+str(idx)+'.npy')

plt.figure(figsize=(8,2))
plt.plot(t,E_N_0,linewidth=3) 
plt.plot(t,E_N_1,linewidth=3) 
plt.xlabel(r'$t(\omega_{m}^{-1})$',fontsize=fontsize_define)
plt.ylabel(r'$E_N(t)$',fontsize=fontsize_define)
plt.yticks(fontsize=fontsize_define)
plt.xticks(fontsize=fontsize_define)
plt.savefig('./EN.pdf',bbox_inches='tight')

Pn = np.load('./dataImp/Pn_step_900.npy')
Mn = np.load('./dataImp/Mn_step_900.npy')

plt.figure()
plt.plot(t,Pn,linewidth=3,label=r'$P_n$') 
plt.plot(t,Mn,linewidth=3,label=r'$M_n$')
plt.legend(fontsize=fontsize_define) 
plt.xlabel(r'$t(\omega_{m}^{-1})$',fontsize=fontsize_define)
plt.ylabel('Exp current',fontsize=fontsize_define)
plt.yticks(fontsize=fontsize_define)
plt.xticks(fontsize=fontsize_define)
plt.savefig('./exp_current.pdf',bbox_inches='tight')

reward = np.load('./dataImp/reward_env_'+str(idx)+'.npy')

plt.figure(figsize=(8,2))
plt.plot(np.arange(0,np.size(reward),1),reward,linewidth=3) 
plt.xlabel('Episode',fontsize=fontsize_define)
plt.ylabel(r'$\widetilde{R}$',fontsize=fontsize_define)
plt.yticks(fontsize=fontsize_define)
plt.xticks([0,200,400,600,800,1000,1200],fontsize=fontsize_define)
plt.savefig('./reward_over_envs.pdf',bbox_inches='tight')

E_N_0_Ep = np.load('./dataImp/EN0Ep.npy')
E_N_1_Ep = np.load('./dataImp/EN1Ep.npy')

plt.figure(figsize=(8,2))
plt.plot(np.arange(0,np.size(reward),1),E_N_0_Ep[0:np.size(reward)],linewidth=3) 
plt.plot(np.arange(0,np.size(reward),1),E_N_0_Ep[0:np.size(reward)],linewidth=3) 
plt.xlabel('Episode',fontsize=fontsize_define)
plt.ylabel(r'$\widetilde{E}_N$',fontsize=fontsize_define)
plt.yticks(fontsize=fontsize_define)
plt.xticks([0,200,400,600,800,1000,1200],fontsize=fontsize_define)
plt.savefig('./ENEp.pdf',bbox_inches='tight')

