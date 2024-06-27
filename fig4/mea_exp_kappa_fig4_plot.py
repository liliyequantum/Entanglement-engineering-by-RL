# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:39:35 2023

@author: liliye
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl  
mpl.rc('font',family='Times New Roman')

# exp_kappa = np.array([0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5])
# exp_training = np.array([96.71, 89.37, 85.51, 81.18, 72.56, 48.44, 31.54])/100*0.7/np.log(2)
# exp_testing = np.array([97.00, 89.25, 85.88, 79.22, 72.50, 53.33, 27.60])/100*0.7/np.log(2)
# exp_training_var = np.array([0.19, 2.07, 6.04, 3.62, 4.17, 18.80, 12.84])/100*0.7/np.log(2)
# exp_testing_var = np.array([0.23, 2.51, 2.33, 4.84, 6.82, 12.68, 14.87])/100*0.7/np.log(2)

# mea_kappa = np.array([0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5])
# mea_training = np.array([64.76, 65.01, 60.52, 53.20, 44.21, 31.73, 19.82])/100*0.7/np.log(2)
# mea_testing = np.array([65.30, 64.41, 59.88, 52.85, 45.97, 28.48, 21.53])/100*0.7/np.log(2)
# mea_training_var = np.array([1.30, 1.25, 1.84, 4.75, 6.20, 4.35, 3.94])/100*0.7/np.log(2)
# mea_testing_var = np.array([1.38, 1.31, 3.73, 5.49, 2.86, 3.21, 5.80])/100*0.7/np.log(2)

exp_kappa = np.array([0, 0.01, 0.03, 0.05])
exp_training = np.array([96.71, 89.37, 85.51, 81.18])/100*0.7/np.log(2)
exp_testing = np.array([97.00, 89.25, 85.88, 79.22])/100*0.7/np.log(2)
exp_training_var = np.array([0.19, 2.07, 6.04, 3.62])/100*0.7/np.log(2)
exp_testing_var = np.array([0.23, 2.51, 2.33, 4.84])/100*0.7/np.log(2)

mea_kappa = np.array([0, 0.01, 0.03, 0.05])
mea_training = np.array([64.76, 65.01, 60.52, 53.20])/100*0.7/np.log(2)
mea_testing = np.array([65.30, 64.41, 59.88, 52.85])/100*0.7/np.log(2)
mea_training_var = np.array([1.30, 1.25, 1.84, 4.75])/100*0.7/np.log(2)
mea_testing_var = np.array([1.38, 1.31, 3.73, 5.49])/100*0.7/np.log(2)


width = 6
height = 3
# plt.figure(figsize=(width,height))
# plt.plot(exp_kappa, exp_training, '-',marker= 'o', linewidth=3, label='exp training')
# plt.plot(exp_kappa, exp_testing, marker= 'o', linewidth=3, label = 'exp testing')
# plt.plot(mea_kappa, mea_training, marker= 'o', linewidth=3, label='mea training')
# plt.plot(mea_kappa, mea_testing, marker= 'o', linewidth=3, label = 'mea testing')
# plt.xscale('log')
# plt.legend(fontsize=14)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()

fig, ax = plt.subplots(figsize=(width,height))
ax.errorbar(exp_kappa, exp_training,
            yerr=exp_training_var,elinewidth=2,linewidth=2,label='exp training',
            capsize=4)
ax.errorbar(exp_kappa, exp_testing,
            yerr=exp_testing_var,elinewidth=2,linewidth=2,label = 'exp testing',
            capsize=4)
ax.errorbar(mea_kappa, mea_training,
            yerr=mea_training_var,elinewidth=2,linewidth=2,label = 'mea training',
            capsize=4)
ax.errorbar(mea_kappa, mea_testing,
            yerr=mea_testing_var,elinewidth=2,linewidth=2, label = 'mea testing',
            capsize=4)
ax.legend(loc=[0.01,0.01],fontsize=13)
# ax.set_xscale('log')
ax.set_ylim([0.25,1])
ax.set_xlabel(r'$\kappa(\omega_m)$',fontsize=18)
ax.set_ylabel(r'$\langle E_{N}\rangle/E_0$',fontsize=18)
ax.tick_params(labelsize=18)
fig.savefig('exp_mea_training_testing_kappa.pdf',bbox_inches='tight')
# plt.plot(exp_measure_rate, exp_training, '-',marker= 'o', linewidth=3, label='exp training')
# plt.plot(exp_measure_rate, exp_testing, marker= 'o', linewidth=3, label = 'exp testing')
# plt.plot(mea_measure_rate, mea_training, marker= 'o', linewidth=3, label='mea training')
# plt.plot(mea_measure_rate, mea_testing, marker= 'o', linewidth=3, label = 'mea testing')
# plt.xscale('log')
# plt.legend(fontsize=14)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()