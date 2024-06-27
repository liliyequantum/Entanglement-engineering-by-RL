# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 08:03:02 2023

@author: liliye
"""
import numpy as np
import matplotlib.pyplot as plt

exp_percent_ave_EN = np.load('./exp/10_test_diff/percent_ave_EN.npy')*0.7/np.log(2)
exp_std_EN = np.load('./exp/10_test_diff/std_EN.npy')*0.7/np.log(2)
rand_exp_percent_ave_EN = np.load('./exp/random_test_diff/percent_ave_EN.npy')*0.7/np.log(2)
rand_exp_std_EN = np.load('./exp/random_test_diff/std_EN.npy')*0.7/np.log(2)
mea_percent_ave_EN = np.load('./mea/10_test_diff/percent_ave_EN.npy')*0.7/np.log(2)
mea_std_EN = np.load('./mea/10_test_diff/std_EN.npy')*0.7/np.log(2)
rand_mea_percent_ave_EN = np.load('./mea/random_test_diff/percent_ave_EN.npy')*0.7/np.log(2)
rand_mea_std_EN = np.load('./mea/random_test_diff/std_EN.npy')*0.7/np.log(2)

num_Mix_p = 100
Mix_p = np.linspace(0, 1, num_Mix_p)

width = 6
height = 3
fig = plt.figure(figsize=(width,height))

plt.plot(Mix_p, exp_percent_ave_EN,'-',color='navy',alpha=0.5)
plt.fill_between(Mix_p, exp_percent_ave_EN-exp_std_EN, exp_percent_ave_EN+exp_std_EN,\
                 color='cornflowerblue',alpha=0.5)
plt.plot(Mix_p, rand_exp_percent_ave_EN,'-',color='navy')
plt.fill_between(Mix_p, rand_exp_percent_ave_EN-exp_std_EN, \
                 exp_percent_ave_EN+exp_std_EN,\
                 color='cornflowerblue')
plt.plot(Mix_p, mea_percent_ave_EN,'-',color='maroon',alpha=0.5)
plt.fill_between(Mix_p, mea_percent_ave_EN-mea_std_EN, mea_percent_ave_EN+mea_std_EN,\
                  color='lightcoral',alpha=0.5)
plt.plot(Mix_p, rand_mea_percent_ave_EN,'-',color='maroon')
plt.fill_between(Mix_p, rand_mea_percent_ave_EN-mea_std_EN, mea_percent_ave_EN+mea_std_EN,\
                  color='lightcoral')
plt.xlabel(r'$p$',fontsize=18)
plt.ylabel(r'$\langle E_{N}\rangle/E_0$',fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
# plt.ylim([0,1])
plt.show()
fig.savefig('./exp_mea_test_diff_initial_continuous.pdf',bbox_inches='tight')
