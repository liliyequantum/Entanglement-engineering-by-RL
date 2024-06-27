# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 09:20:55 2023

@author: liliye
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl  
mpl.rc('font',family='Times New Roman')

exp_measure_rate = np.array([0.05,0.1, 0.3, 0.5, 0.7, 1])
exp_training = np.array([95.75, 95.74, 95.12, 89.37, 86.14, 82.99])/100*0.7/np.log(2)
exp_testing = np.array([96.03, 95.72, 94.74, 89.25, 85.85, 84.12])/100*0.7/np.log(2)
exp_training_var = np.array([0.67, 0.63, 0.55, 2.07, 3.71, 1.83])/100*0.7/np.log(2)
exp_testing_var = np.array([0.31, 0.66, 1.69, 2.51, 2.42, 1.97])/100*0.7/np.log(2)


mea_measure_rate = np.array([0.05, 0.1, 0.3, 0.5, 0.7, 1])
mea_training = np.array([65.19, 65.10, 65.09, 65.01, 63.73, 64.18])/100*0.7/np.log(2)
mea_testing = np.array([66.26, 65.78, 65.01, 64.41, 64.51, 64.37])/100*0.7/np.log(2)
mea_training_var = np.array([3.08, 5.09, 1.16, 1.25, 1.51, 1.46])/100*0.7/np.log(2)
mea_testing_var = np.array([4.86, 7.95, 2.78, 1.31, 1.14, 1.74])/100*0.7/np.log(2)

width = 6
height = 3
# plt.figure(figsize=(width,height))
# plt.plot(exp_measure_rate, exp_training, '-',marker= 'o', linewidth=3, label='exp training')
# plt.plot(exp_measure_rate, exp_testing, marker= 'o', linewidth=3, label = 'exp testing')
# plt.plot(mea_measure_rate, mea_training, marker= 'o', linewidth=3, label='mea training')
# plt.plot(mea_measure_rate, mea_testing, marker= 'o', linewidth=3, label = 'mea testing')
# plt.xscale('log')
# plt.legend(fontsize=14)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()

fig, ax = plt.subplots(figsize=(width,height))
ax.errorbar(exp_measure_rate, exp_training,
            yerr=exp_training_var,elinewidth=2,linewidth=2,label='exp training',
            capsize=4)
ax.errorbar(exp_measure_rate, exp_testing,
            yerr=exp_testing_var,elinewidth=2,linewidth=2,label = 'exp testing',
            capsize=4)
ax.errorbar(mea_measure_rate, mea_training,
            yerr=mea_training_var,elinewidth=2,linewidth=2,label = 'mea training',
            capsize=4)
ax.errorbar(mea_measure_rate, mea_testing,
            yerr=mea_testing_var,elinewidth=2,linewidth=2, label = 'mea testing',
            capsize=4)
# ax.legend(loc=[0.1,0.35],fontsize=14)
ax.set_ylim([0.55,1])
ax.set_xlabel(r'$\eta$',fontsize=18)
ax.set_ylabel(r'$\langle E_{N}\rangle/E_0$',fontsize=18)
ax.tick_params(labelsize=18)
fig.savefig('exp_mea_training_testing_eta.pdf',bbox_inches='tight')
# plt.plot(exp_measure_rate, exp_training, '-',marker= 'o', linewidth=3, label='exp training')
# plt.plot(exp_measure_rate, exp_testing, marker= 'o', linewidth=3, label = 'exp testing')
# plt.plot(mea_measure_rate, mea_training, marker= 'o', linewidth=3, label='mea training')
# plt.plot(mea_measure_rate, mea_testing, marker= 'o', linewidth=3, label = 'mea testing')
# plt.xscale('log')
# plt.legend(fontsize=14)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()