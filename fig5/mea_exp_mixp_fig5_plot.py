# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:03:35 2023

@author: liliye
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl  
mpl.rc('font',family='Times New Roman')

Mix_p = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
exp_training = np.array([89.37, 92.91, 89.24, 87.20, 82.95, 82.72])/100*0.7/np.log(2)
exp_testing = np.array([89.25, 90.05, 88.87, 86.17, 83.82, 84.34])/100*0.7/np.log(2)
exp_training_var = np.array([2.07, 1.71, 3.54, 4.27, 3.74, 4.00])/100*0.7/np.log(2)
exp_testing_var = np.array([2.51, 4.21, 3.66, 3.04, 3.84, 2.67])/100*0.7/np.log(2)


mea_training = np.array([65.01, 60.70, 55.39, 52.40, 52.24, 48.56])/100*0.7/np.log(2)
mea_testing = np.array([64.41, 60.75, 56.87, 55.66, 53.61, 49.57])/100*0.7/np.log(2)
mea_training_var = np.array([1.25, 2.12, 2.80, 5.30, 4.12, 3.60])/100*0.7/np.log(2)
mea_testing_var = np.array([1.31, 2.55, 2.11, 4.20, 2.35, 3.00])/100*0.7/np.log(2)

width = 6
height = 3
# plt.figure(figsize=(width,height))
# plt.plot(Mix_p, exp_training, '-',marker= 'o', linewidth=3, label='exp training')
# plt.plot(Mix_p, exp_testing, marker= 'o', linewidth=3, label = 'exp testing')
# plt.plot(Mix_p, mea_training, marker= 'o', linewidth=3, label='mea training')
# plt.plot(Mix_p, mea_testing, marker= 'o', linewidth=3, label = 'mea testing')
# plt.xscale('log')
# plt.legend(fontsize=14)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()
y0=np.ones(np.size(Mix_p),)
fig, ax = plt.subplots(figsize=(width,height))

ax.fill_between(Mix_p, (0.8009*y0+0.0329)*0.7/np.log(2), (0.8009*y0-0.0329)*0.7/np.log(2),color='steelblue',alpha=0.2)
ax.plot(Mix_p, (0.8009*y0)*0.7/np.log(2),'--',color='steelblue',alpha=0.5)

ax.errorbar(Mix_p, exp_training,
            yerr=exp_training_var,elinewidth=2,linewidth=2,label='exp training',
            capsize=4)
ax.errorbar(Mix_p, exp_testing,
            yerr=exp_testing_var,elinewidth=2,linewidth=2,label = 'exp testing',
            capsize=4)

ax.fill_between(Mix_p, (0.5879*y0+0.0253)*0.7/np.log(2), (0.5879*y0-0.0253)*0.7/np.log(2),color='lightcoral',alpha=0.2)
ax.plot(Mix_p, (0.5879*y0)*0.7/np.log(2),'--',color='lightcoral',alpha=0.5)

ax.errorbar(Mix_p, mea_training,
            yerr=mea_training_var,elinewidth=2,linewidth=2,label = 'mea training',
            capsize=4)
ax.errorbar(Mix_p, mea_testing,
            yerr=mea_testing_var,elinewidth=2,linewidth=2, label = 'mea testing',
            capsize=4)

# ax.legend(loc=[0.25,0.35],fontsize=12)
ax.set_ylim([0.4,1])
ax.set_xlabel(r'$p$',fontsize=18)
ax.set_ylabel(r'$\langle E_{N}\rangle/E_0$',fontsize=18)
ax.tick_params(labelsize=18)
fig.savefig('exp_mea_training_testing_mix_p.pdf',bbox_inches='tight')
# plt.plot(exp_measure_rate, exp_training, '-',marker= 'o', linewidth=3, label='exp training')
# plt.plot(exp_measure_rate, exp_testing, marker= 'o', linewidth=3, label = 'exp testing')
# plt.plot(mea_measure_rate, mea_training, marker= 'o', linewidth=3, label='mea training')
# plt.plot(mea_measure_rate, mea_testing, marker= 'o', linewidth=3, label = 'mea testing')
# plt.xscale('log')
# plt.legend(fontsize=14)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()