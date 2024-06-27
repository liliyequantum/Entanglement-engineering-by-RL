# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:52:14 2023

@author: liliye
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:24:54 2023

@author: liliye
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from qutip import *
import os 
from matplotlib import cm
import matplotlib as mpl
import random
# from multiprocessing.pool import Pool
from scipy.ndimage import gaussian_filter1d
def plot_fock_distribution_1(rho, index, offset=0, fig=None, ax=None,
                           figsize=(8, 6), title=None, unit_y_range=True):
    """
    Plot the Fock distribution for a density matrix (or ket) that describes
    an oscillator mode.

    Parameters
    ----------
    rho : :class:`qutip.qobj.Qobj`
        The density matrix (or ket) of the state to visualize.

    fig : a matplotlib Figure instance
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.

    title : string
        An optional title for the figure.

    figsize : (width, height)
        The size of the matplotlib figure (in inches) if it is to be created
        (that is, if no 'fig' and 'ax' arguments are passed).

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.
    """

    if not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if isket(rho):
        rho = ket2dm(rho)

    N = rho.shape[0]

    ax.bar(np.arange(offset, offset + N), np.real(rho.diag()),
           color="green", alpha=0.6, width=0.8)
    if unit_y_range:
        ax.set_ylim(0, 1)

    ax.set_xlim(-0.5,N)
    ax.set_xlabel('Fock number', fontsize=18)
    ax.set_ylabel('Occupation probability'+str(index), fontsize=18)
#     ax.set_xticks([0,5,10,15])
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
#     ax.set_xticklabels([0,5,10,15],fontsize=18)  
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0],fontsize=18)  
    if title:
        ax.set_title(title)
    return fig, ax

def log_neg(rho_test, trans_subsystem):
    if trans_subsystem == 0:
        rho_partial = partial_transpose(rho_test,mask=[True, False])
    else:
        rho_partial = partial_transpose(rho_test,mask=[False, True])
    
    rho_partial_hermitian = rho_partial.trans().conj()
    A = rho_partial_hermitian*rho_partial
    B = A.sqrtm()
    C = B.tr()
    E_N = np.log(C)
    
    return E_N

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

def randomControl(N,kappa,gamma,num_time,dt,N_t,measure_rate,num_repeat,ntraj,\
                  mean_interval,filter_interval,gaussian_var,dirname):
   
    
        
    a = tensor(destroy(N),identity(N)) # photon
    b = tensor(identity(N),destroy(N)) # phonon
    P_0 = tensor(ket2dm(basis(N,0)),identity(N))
    P_1 = tensor(ket2dm(basis(N,1)),identity(N))
    # pn = a.dag()*a
    H_0 = a.dag() * a + b.dag() * b
    
    E_N_0 = np.zeros((num_time,num_repeat))
    E_N_1 = np.zeros((num_time,num_repeat))
    measure_traj = np.zeros((num_time,num_repeat))
    mean_current = np.zeros((num_time,num_repeat))
    use_mean_current = np.zeros((num_time,num_repeat))
    G = np.zeros((num_time,num_repeat))
    
    rho0 = ket2dm(tensor(basis(N,1),basis(N,0)))
    rho0_traj = np.zeros((N**2,N**2,ntraj),dtype=complex)
    
    opt = Options()
    opt.store_states = True
    
    from tqdm import tqdm
    for k in tqdm(range(num_repeat)):
        for j in range(ntraj):
            rho0_traj[:,:,j] = rho0
        measurewindow = np.array([])
        filter_current_window = np.array([])
       
        for i in range(num_time):
            G[i,k] = np.random.uniform(-5,5)
            H_coupling = G[i,k] * (b.dag()*a + b*a.dag())
            
            
            for j in range(ntraj):
                
                result = smesolve(H_0 + H_coupling, Qobj(rho0_traj[:,:,j],dims=[[N,N],[N,N]]), 
                                  times = np.linspace(0,dt,N_t), 
                                  c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * b], 
                                  sc_ops = [np.sqrt(measure_rate)*P_0, np.sqrt(measure_rate)*P_1],
                                  e_ops = [], ntraj=1, solver="taylor1.5",tol = 1e-6,
                                  m_ops= [np.sqrt(measure_rate)*P_1],
                                  dW_factors=[1/np.sqrt(4*measure_rate)],method='homodyne', 
                                  store_measurement=True, map_func=parallel_map, options=opt)
                rho0_traj[:,:,j] = result.states[0][N_t-1]
                
                
                E_N_0[i,k] = E_N_0[i,k] + np.real(log_neg(result.states[0][0], 0))
                E_N_1[i,k] = E_N_1[i,k] + np.real(log_neg(result.states[0][0], 1))
                
                measure_traj[i,k] = measure_traj[i,k] + np.real(result.measurement)[0][0][0]/np.sqrt(measure_rate)
               
            
            if len(measurewindow)>=mean_interval:
                measurewindow = np.delete(measurewindow,0)
            measurewindow = np.append(measurewindow, measure_traj[i,k]/ntraj)
            mean_current[i,k] = np.sum(measurewindow)/np.size(measurewindow)
            
            if len(filter_current_window)>=filter_interval:
                filter_current_window = np.delete(filter_current_window,0)
            filter_current_window = np.append(filter_current_window, mean_current[i,k])
            mea_par = gaussian_filter1d(filter_current_window,gaussian_var)[-1]
            use_mean_current[i,k] = mea_par
        
    E_N_0 = E_N_0/ntraj
    E_N_1 = E_N_1/ntraj
    measure_traj = measure_traj/ntraj
    
    np.save('./'+dirname+'/E_N_0.npy',E_N_0)
    np.save('./'+dirname+'/E_N_1.npy',E_N_1)
    np.save('./'+dirname+'/measure_traj.npy',measure_traj)
    np.save('./'+dirname+'/mean_current.npy',mean_current)
    np.save('./'+dirname+'/use_mean_current.npy',use_mean_current)
    np.save('./'+dirname+'/G.npy',G)
    np.save('./'+dirname+'/num_time.npy',num_time)
    np.save('./'+dirname+'/num_repeat.npy',num_repeat)
    
def plot(dirname):
    
    E_N_0 = np.load('./'+dirname+'/E_N_0.npy')
    E_N_1 = np.load('./'+dirname+'/E_N_1.npy')
    mean_current = np.load('./'+dirname+'/mean_current.npy')
    use_mean_current = np.load('./'+dirname+'/use_mean_current.npy')
    G = np.load('./'+dirname+'/G.npy')
    num_time = np.load('./'+dirname+'/num_time.npy')
    num_repeat = np.load('./'+dirname+'/num_repeat.npy')
    
    # plot
    E_N_0_repeat = np.sum(E_N_0,axis=0)/num_time
    E_N_1_repeat = np.sum(E_N_1,axis=0)/num_time
    
    times = np.linspace(0,num_time * 0.01,num_time)
   
    ave_mean_current = _ave(num_time, 10, mean_current[:,num_repeat-1])
    plt.plot(times,mean_current[:,num_repeat-1],linewidth=3,label='mean current')
    # plt.plot(times,ave_mean_current,linewidth=3,label='ave mean current')
    plt.plot(times,use_mean_current[:,num_repeat-1],linewidth=3,label='use mean current')
    plt.legend(fontsize=15)
    plt.xlabel('times',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    plt.plot(times,G[:,num_repeat-1],linewidth=3,label='G')
    plt.legend(fontsize=15)
    plt.xlabel('times',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
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
    print('ave E_N: '+str(ave_EN))
    print('stand deviation: '+str(np.std(E_N_0_repeat)))
    print('percent ave E_N: '+str(ave_EN/np.log(2)))
    print('percent stand deviation: '+str(np.std(E_N_0_repeat)/np.log(2)))
    # print(np.sum(mean_current)/num_time)
    
    
    
if __name__ == '__main__':  
    
    N = 2 # Fock basis
    # w_{M} = 1
    kappa = 0.01
    gamma = 0.0001
    
    num_time = 3500
    dt = 0.01
    N_t = 2
    measure_rate = 1
    num_repeat = 10
    ntraj = 1
    
    mean_interval = 5
    filter_interval = 10
    gaussian_var = 3
    
    dirname = './randomControl'
    update_dir(dirname)
    dirname = dirname+'/time_'+str(int(dt*num_time))+'_measureRate_'+str(measure_rate)+\
              '_ntraj_'+str(ntraj)
    update_dir(dirname)
    
    # randomControl(N,kappa,gamma,num_time,dt,N_t,measure_rate,num_repeat,ntraj,\
    #                   mean_interval,filter_interval,gaussian_var,dirname)
    plot(dirname)
    