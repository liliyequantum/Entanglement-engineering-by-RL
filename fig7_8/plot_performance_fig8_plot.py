# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 09:20:55 2023

@author: liliye
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl  
mpl.rc('font',family='Times New Roman')

max_measure_rate = [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]

dirname = './PPO_perform'
training_ave_E_N = np.load(dirname+'/training_ave_E_N.npy')
training_std_E_N = np.load(dirname+'/training_std_E_N.npy')
testing_ave_E_N = np.load(dirname+'/testing_ave_E_N.npy')
testing_std_E_N = np.load(dirname+'/testing_std_E_N.npy')
dirname = './random_perform'
random_ave_E_N = np.load(dirname+'/random_ave_E_N.npy')
random_std_E_N = np.load(dirname+'/random_std_E_N.npy')


width = 6
height = 3

fig, ax = plt.subplots(figsize=(width,height))
ax.errorbar(max_measure_rate, training_ave_E_N,
            yerr=training_std_E_N,elinewidth=2,linewidth=2,label='recurrent PPO training',
            capsize=4)
ax.errorbar(max_measure_rate, testing_ave_E_N,
            yerr=testing_std_E_N ,elinewidth=2,linewidth=2,label = 'recurrent PPO testing',
            capsize=4)
ax.errorbar(max_measure_rate, random_ave_E_N,
            yerr=random_std_E_N,elinewidth=2,linewidth=2,label = 'random control',
            capsize=4)

ax.legend(loc='best',fontsize=12)
# ax.set_ylim([0.55,1])
ax.set_xlabel(r'$\eta$',fontsize=18)
ax.set_ylabel(r'$\langle E_{N}\rangle/E_0$',fontsize=18)
ax.tick_params(labelsize=18)
fig.savefig('nonlinear_PPO_randomControl_performance.pdf',bbox_inches='tight')
