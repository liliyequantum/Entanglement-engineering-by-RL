## notic for cluster server
# uncomment plt.switch_backend('agg')

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sys

import gym
from gym import spaces, logger

from qutip import *
from numpy import genfromtxt #read excel for parallel env
from scipy.ndimage import gaussian_filter1d

class quantumEnvMeasureFilter_update(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self,measure_rate,n_steps,n_envs,n_episode,mean_interval,ntraj,\
          filter_interval,gaussian_var,output_interval,kappa,rundir,mix_p):
        
        self.action_space = spaces.Box(-5, 5, shape=(1,), dtype=np.float32) # G
        self.observation_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32) # numphoton
        
        self.measure_rate = measure_rate
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.n_episode = n_episode
        self.mean_interval = mean_interval
        self.ntraj = ntraj
        self.filter_interval = filter_interval
        self.gaussian_var = gaussian_var
        self.output_interval = output_interval
        self.rundir = rundir
        self.mix_p = mix_p
        
        # parameter
        self.N = 2  # N Fock basis
        self.dt = 0.01 # WCM time step size
        self.N_t = 2
        self.kappa = kappa # photon decay rate
        self.gamma = 0.0001 # phonon decay rate
        
        self.rho0 = (1-self.mix_p)*ket2dm(tensor(basis(self.N, 1), basis(self.N, 0))) + \
            self.mix_p*ket2dm(tensor(basis(self.N, 0), basis(self.N, 1)))# intial state
        # self.rho0 = ket2dm(tensor(basis(self.N, 1), basis(self.N, 0)))# intial state
        self.rho0_traj = np.zeros((self.N**2,self.N**2,self.ntraj),dtype=complex)
        
        self.a = tensor(destroy(self.N),identity(self.N)) # photon
        self.b = tensor(identity(self.N),destroy(self.N)) # phonon
        
        self.P_0 = tensor(ket2dm(basis(self.N,0)),identity(self.N))
        self.P_1 = tensor(ket2dm(basis(self.N,1)),identity(self.N))
        self.H_0 = self.a.dag() * self.a + self.b.dag() * self.b # Hamiltonian

        self.meanRewardEp = np.zeros(self.n_episode,)
        self.EN0Ep = np.zeros(self.n_episode,)
        self.EN1Ep = np.zeros(self.n_episode,)
        
        self.numEpisode = 0
        
    def reset(self):
    
        # initial state
        for j in range(self.ntraj):
            self.rho0_traj[:,:,j] = self.rho0
            
        observation = [1] #numPhoton
        
        self.numStep = 1
        # print('numStep ',self.numStep)

        self.done = False
        self.rewardStep = 0
        self.EN0Step = 0
        self.EN1Step = 0

        self.measureWindow = np.array([])
        self.filter_current_window = np.array([])
        
        self.controlG = np.zeros(self.n_steps-1,)
        self.use_mean_current = np.zeros(self.n_steps-1,)
        self.EN0_step = np.zeros(self.n_steps-1,)
        self.EN1_step = np.zeros(self.n_steps-1,)
        
        return observation
    
    
    def step(self, action):
        
        self.numStep += 1
        #print('numStep ',self.numStep)
        
        G = action[0] 
        H_coupling = G * (self.a.dag()*self.b + self.a*self.b.dag())
 
        opt = Options()
        opt.store_states = True
        
        E_N_0 = 0
        E_N_1 = 0
        measure_traj = 0
        for j in range(self.ntraj):
            
            result = smesolve(self.H_0 + H_coupling, 
                        Qobj(self.rho0_traj[:,:,j],dims=[[self.N,self.N],[self.N,self.N]]), 
                        times = np.linspace(0,self.dt,self.N_t),
                        c_ops = [np.sqrt(self.kappa)*self.a, np.sqrt(self.gamma)*self.b], 
                        sc_ops = [np.sqrt(self.measure_rate)*self.P_0,np.sqrt(self.measure_rate)*self.P_1],
                        e_ops = [], 
                        ntraj=1, solver="taylor1.5", tol = 1e-6,
                        m_ops= [np.sqrt(self.measure_rate)*self.P_1], 
                        dW_factors=[1/np.sqrt(4*self.measure_rate)],
                        method='homodyne',store_measurement=True,progress_bar=None,
                        map_func=parallel_map, options=opt)
            self.rho0_traj[:,:,j] = result.states[0][self.N_t-1]
            
            E_N_0 = E_N_0 + np.real(self.log_neg(result.states[0][0], 0))
            E_N_1 = E_N_1 + np.real(self.log_neg(result.states[0][0], 1))
            measure_traj = measure_traj + np.real(result.measurement)[0][0][0]/np.sqrt(self.measure_rate)
            
            
        E_N_0 = E_N_0/self.ntraj
        E_N_1 = E_N_1/self.ntraj
        measure_traj = measure_traj/self.ntraj

        if len(self.measureWindow)>=self.mean_interval:
            self.measureWindow = np.delete(self.measureWindow,0)
        self.measureWindow = np.append(self.measureWindow, measure_traj)
        mean_numPhoton = np.sum(self.measureWindow)/np.size(self.measureWindow)
        # print('mean numPhoton: '+str(mean_numPhoton))
        
        # Gaussian filter
        if len(self.filter_current_window)>=self.filter_interval:
            self.filter_current_window = np.delete(self.filter_current_window,0)
        self.filter_current_window = np.append(self.filter_current_window, mean_numPhoton)
        filter_mean_numPhoton = gaussian_filter1d(self.filter_current_window,self.gaussian_var)[-1]
        
        observation = [filter_mean_numPhoton]
        reward = -abs(filter_mean_numPhoton-0.5)

        self.controlG[self.numStep-2] = G
        self.use_mean_current[self.numStep-2] = filter_mean_numPhoton
        self.EN0_step[self.numStep-2] = E_N_0
        self.EN1_step[self.numStep-2] = E_N_1
        
        self.rewardStep += reward
        self.EN0Step += E_N_0
        self.EN1Step += E_N_1
        
        if self.numStep == self.n_steps:
           self.done = True
           if self.numEpisode < self.n_episode:
               self.meanRewardEp[self.numEpisode] = self.rewardStep/self.n_steps
               self.EN0Ep[self.numEpisode] = self.EN0Step/self.n_steps
               self.EN1Ep[self.numEpisode] = self.EN1Step/self.n_steps
               np.save(self.rundir+'/dataImp/meanRewardEp.npy',self.meanRewardEp)
               np.save(self.rundir+'/dataImp/EN0Ep.npy',self.EN0Ep)
               np.save(self.rundir+'/dataImp/EN1Ep.npy',self.EN1Ep)
               
               if self.numEpisode in range(self.output_interval,self.n_episode,self.output_interval):
                     np.save(self.rundir+'/dataImp/controlG'+str(self.numEpisode)+'.npy',self.controlG)
                     np.save(self.rundir+'/dataImp/use_mean_current'+str(self.numEpisode)+'.npy',self.use_mean_current)
                     np.save(self.rundir+'/dataImp/EN0_step'+str(self.numEpisode)+'.npy',self.EN0_step)
                     np.save(self.rundir+'/dataImp/EN1_step'+str(self.numEpisode)+'.npy',self.EN1_step)

                     reward_env = np.zeros((self.numEpisode,self.n_envs))
                     for i in range(self.n_envs):
                         my_data = genfromtxt(self.rundir+'/monitor/'+str(i)+'.monitor.csv', delimiter=',')
                         reward_env[:,i] = my_data[1:self.numEpisode+1,0]
                     reward_env = np.sum(reward_env,axis=1)/self.n_envs/self.n_steps
                     ave_reward_env = self._ave(self.numEpisode, 100, reward_env)
                     np.save(self.rundir+'/dataImp/reward_env_'+str(self.numEpisode)+'.npy',reward_env)
                     
                     num = 0
                     num += 1
                     plt.figure(num)
                     plt.plot(np.arange(0,self.numEpisode,1),reward_env,linewidth=3) 
                     plt.plot(np.arange(0,self.numEpisode,1),ave_reward_env,linewidth=3) 
                     plt.xlabel('episode',fontsize=16)
                     plt.ylabel('mean reward over envs',fontsize=16)
                     plt.yticks(fontsize=16)
                     plt.xticks(fontsize=16)
                     plt.savefig(self.rundir+'/picture/reward_envs_'+str(self.numEpisode)+'.png',bbox_inches='tight')
                     plt.close()
                     
                     ave_Reward_ep = self._ave(self.numEpisode, 100, self.meanRewardEp) 
                     num += 1
                     plt.figure(num)
                     plt.plot(np.arange(0,self.numEpisode,1),self.meanRewardEp[:self.numEpisode],linewidth=3) 
                     plt.plot(np.arange(0,self.numEpisode,1),ave_Reward_ep,linewidth=3) 
                     plt.xlabel('episode',fontsize=16)
                     plt.ylabel('mean reward',fontsize=16)
                     plt.yticks(fontsize=16)
                     plt.xticks(fontsize=16)
                     plt.savefig(self.rundir+'/picture/reward_'+str(self.numEpisode)+'.png',bbox_inches='tight')
                     plt.close()
                     
                     ave_EN0_ep = self._ave(self.numEpisode, 100, self.EN0Ep)
                     ave_EN1_ep = self._ave(self.numEpisode, 100, self.EN1Ep)
                     num += 1
                     plt.figure(num)
                     plt.plot(np.arange(0,self.numEpisode,1),self.EN0Ep[:self.numEpisode],linewidth=3) 
                     plt.plot(np.arange(0,self.numEpisode,1),ave_EN0_ep,linewidth=3) 
                     plt.plot(np.arange(0,self.numEpisode,1),self.EN1Ep[:self.numEpisode],linewidth=3) 
                     plt.plot(np.arange(0,self.numEpisode,1),ave_EN1_ep,linewidth=3) 
                     plt.xlabel('episode',fontsize=16)
                     plt.ylabel('E_N',fontsize=16)
                     plt.yticks(fontsize=16)
                     plt.xticks(fontsize=16)
                     plt.savefig(self.rundir+'/picture/E_N_'+str(self.numEpisode)+'.png',bbox_inches='tight')
                     plt.close()
                     
                     ave_EN0_step = self._ave(self.n_steps-1, 10, self.EN0_step)
                     ave_EN1_step = self._ave(self.n_steps-1, 10, self.EN1_step)
                     num += 1
                     plt.figure(num)
                     plt.plot(0.01*np.arange(0,self.n_steps-1,1),self.EN0_step,linewidth=3) 
                     plt.plot(0.01*np.arange(0,self.n_steps-1,1),self.EN1_step,linewidth=3) 
                     plt.plot(0.01*np.arange(0,self.n_steps-1,1),ave_EN0_step,linewidth=3) 
                     plt.plot(0.01*np.arange(0,self.n_steps-1,1),ave_EN1_step,linewidth=3) 
                     plt.xlabel('times',fontsize=16)
                     plt.ylabel('E_N step',fontsize=16)
                     plt.yticks(fontsize=16)
                     plt.xticks(fontsize=16)
                     plt.savefig(self.rundir+'/picture/EN_step_'+str(self.numEpisode)+'.png',bbox_inches='tight')
                     plt.close()
                     
                     ave_use_mean_current = self._ave(self.n_steps-1, 100, self.use_mean_current)   
                     num += 1
                     plt.figure(num)
                     plt.plot(0.01*np.arange(0,self.n_steps-1,1),self.use_mean_current,linewidth=3) 
                     plt.plot(0.01*np.arange(0,self.n_steps-1,1),ave_use_mean_current,linewidth=3) 
                     plt.xlabel('times',fontsize=16)
                     plt.ylabel('mean current',fontsize=16)
                     plt.yticks(fontsize=16)
                     plt.xticks(fontsize=16)
                     plt.savefig(self.rundir+'/picture/use_mean_current_'+str(self.numEpisode)+'.png',bbox_inches='tight')
                     plt.close()
                     
                     num += 1
                     plt.figure(num)
                     plt.plot(0.01*np.arange(0,self.n_steps-1,1),self.controlG,linewidth=3) 
                     plt.xlabel('times',fontsize=16)
                     plt.ylabel('control G',fontsize=16)
                     plt.yticks(fontsize=16)
                     plt.xticks(fontsize=16)
                     plt.savefig(self.rundir+'/picture/controlG_'+str(self.numEpisode)+'.png',bbox_inches='tight')
                     plt.close()

        

           self.numEpisode +=1
           #print('episode #',str(self.numEpisode)+' mean_numPhoton: '+str(mean_numPhoton))
        
            
        return observation, reward, self.done, {0:E_N_0,1:E_N_1}
    
    def _ave(self, n_steps, ave_step, data_array):
        measureWindow = np.array([])
        ave_data= np.zeros(n_steps,)
        for i in range(n_steps):
            if len(measureWindow)>=ave_step:
                measureWindow = np.delete(measureWindow,0)
            measureWindow = np.append(measureWindow, data_array[i])
            ave_data[i] = np.sum(measureWindow)/np.size(measureWindow)
        return ave_data  
    
    def log_neg(self, rho_test, trans_subsystem):
        ## trans_subsystem = 0 means A subsystem
        ## trans_subsystem = 1 means B subsystem 
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

    def render(self, mode='human'):
        return 1

    def close(self):
        sys.exit()
