# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 09:20:55 2023

@author: liliye
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl  
mpl.rc('font',family='Times New Roman')

T = np.array([500,1000,1500,2000,2500, 3000, 3500, 4000])

exp_DRL = np.array([84.95, 82.91, 82.18, 81.21, 75.56, 80.98, 79.77, 82.35])/100
mea_DRL = np.array([65.01, 64.34, 62.27, 61.07, 64.36, 58.35, 60.87, 61.40])/100
exp_DRL_var = np.array([1.99, 2.34, 1.74, 2.51, 19.08, 3.81, 9.03, 2.42])/100
mea_DRL_var = np.array([1.76, 3.08, 4.01, 4.13, 3.19, 6.45, 5.40, 6.96])/100

exp_bayesian = np.array([93.21, 93.18, 93.40, 93.14, 93.13, 85.27, 91.92, 85.72])/100
mea_bayesian = np.array([49.24, 47.28, 43.37, 44.85, 42.40, 44.43, 42.93, 40.17])/100
exp_bayesian_var = np.array([0.89, 1.09, 1.05, 1.99, 1.27, 15.86, 1.94, 20.19])/100
mea_bayesian_var = np.array([0.44, 2.51, 2.94, 2.59, 4.24, 1.65, 2.63, 2.26])/100

exp_random = np.array([38.15, 36.08, 32.06, 37.68, 30.62, 31.33, 31.27, 34.78])/100
mea_random = np.array([33.46, 30.03, 32.66, 33.35, 29.83, 30.06, 31.84, 29.73])/100
exp_random_var = np.array([9.46, 4.54, 9.28, 8.64, 9.25, 5.78, 6.17, 5.94])/100
mea_random_var = np.array([4.27, 2.37, 4.54, 3.02, 3.18, 2.96, 1.97, 1.69])/100

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
ax.errorbar(T, exp_DRL,
            yerr=exp_DRL_var,elinewidth=2,linewidth=2,label='exp DRL',color='tab:blue',
            capsize=4)
ax.errorbar(T, exp_bayesian,
            yerr=exp_bayesian_var,elinewidth=2,linewidth=2,label = 'exp bayesian',
            capsize=4,alpha=1,color='cornflowerblue')
ax.errorbar(T, mea_DRL,
            yerr=mea_DRL_var,elinewidth=2,linewidth=2,label = 'mea DRL',
            capsize=4,color='tab:red')
ax.errorbar(T, mea_bayesian,
            yerr=mea_bayesian_var,elinewidth=2,linewidth=2, label = 'mea bayesian',
            capsize=4,alpha=1,color='lightcoral')


ax.errorbar(T, exp_random,
            yerr=exp_random_var,elinewidth=2,linewidth=2,label = 'random ntraj=1',
            capsize=4,alpha=0.5,color='gray')
ax.errorbar(T, mea_random,
            yerr=mea_random_var,elinewidth=2,linewidth=2, label = 'random ntraj=5',
            capsize=4,alpha=0.5,color='black')
# ax.legend(fontsize=10)
# ax.set_ylim([0.55,1])
ax.set_xlabel(r'$T$',fontsize=18)
ax.set_ylabel(r'$\langle E_{N}\rangle/E_0$',fontsize=18)
ax.tick_params(labelsize=18)
fig.savefig('T_test.pdf',bbox_inches='tight')
# plt.plot(exp_measure_rate, exp_training, '-',marker= 'o', linewidth=3, label='exp training')
# plt.plot(exp_measure_rate, exp_testing, marker= 'o', linewidth=3, label = 'exp testing')
# plt.plot(mea_measure_rate, mea_training, marker= 'o', linewidth=3, label='mea training')
# plt.plot(mea_measure_rate, mea_testing, marker= 'o', linewidth=3, label = 'mea testing')
# plt.xscale('log')
# plt.legend(fontsize=14)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()