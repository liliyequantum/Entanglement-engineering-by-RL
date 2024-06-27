## notic for cluster server
# uncomment plt.switch_backend('agg')

import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import sys

import gym
from gym import spaces, logger

from qutip import *
from numpy import genfromtxt #read excel for parallel env

class quantumNonlinearEnvSMEExpEN(gym.Env):
    """Custom Environment that follows gym interface"""
    # metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self,rundir,N,g_0,kappa,measure_rate,n_steps,n_episode,output_interval,n_envs,ra,rb):
        #super(quantum, self).__init__()
        
        self.action_space = spaces.Box(-5, 5, shape=(2,), dtype=np.float32) # Delta, alpha_L
        self.observation_space = spaces.Box(0, np.inf, shape=(1,), dtype=np.float32) # EN
        
        self.n_envs = n_envs
        self.output_interval = output_interval
        self.n_steps = n_steps
        self.n_episode = n_episode
        
        # parameter
        self.rundir = rundir
        self.N = N # N Fock basis
        self.g_0 = g_0
        self.kappa = kappa # Cavity damping rate
        self.gamma = 0.01*self.kappa # Mech damping rate
        self.measure_rate = measure_rate
        self.n_th = 0 # mech bath temperature
        self.ra = ra # reward par
        self.rb = rb # reward par

        # intial state, target state
        self.rho_target = ket2dm(1/np.sqrt(2) * \
        (tensor(basis(self.N, 1), basis(self.N, 0)) - 1j * tensor(basis(self.N, 0), basis(self.N, 1))))
        self.rhoInitial = ket2dm(tensor(basis(self.N,0),basis(self.N,0)))
 
        # Hamiltonian
        self.a = tensor(destroy(self.N),identity(self.N)) # photon
        self.b = tensor(identity(self.N),destroy(self.N)) # phonon
        self.H_coupling = self.g_0 * (self.b.dag() + self.b) * self.a.dag() * self.a
        self.pn = self.a.dag()*self.a
        self.mn = self.b.dag()*self.b

        self.P_0 = tensor(ket2dm(basis(self.N,0)),identity(self.N))
        self.P_1 = tensor(ket2dm(basis(self.N,1)),identity(self.N))
        self.P_2 = tensor(ket2dm(basis(self.N,2)),identity(self.N))
        self.P_3 = tensor(ket2dm(basis(self.N,3)),identity(self.N))
        self.P_4 = tensor(ket2dm(basis(self.N,4)),identity(self.N))
        self.P_5 = tensor(ket2dm(basis(self.N,5)),identity(self.N))
        self.P_6 = tensor(ket2dm(basis(self.N,6)),identity(self.N))
        self.P_7 = tensor(ket2dm(basis(self.N,7)),identity(self.N))
        self.P_8 = tensor(ket2dm(basis(self.N,8)),identity(self.N))
        self.P_9 = tensor(ket2dm(basis(self.N,9)),identity(self.N))
        
        # time evolution
        self.dt = 0.01
        self.N_t = 2
        
        self.meanRewardEp = np.zeros(self.n_episode,)
        self.EN0Ep = np.zeros(self.n_episode,)
        self.EN1Ep = np.zeros(self.n_episode,)
        
        self.numEpisode = 0
        
    def reset(self):
    
        # initial state
        self.rho_AB_0 = self.rhoInitial.copy()
        observation = [0] # EN
        
        self.numStep = 1
        # print('numStep ',self.numStep)

        self.done = False
        self.rewardStep = 0
        self.EN0Step = 0
        self.EN1Step = 0
        
        self.Delta = np.zeros(self.n_steps-1,)
        self.alpha_L = np.zeros(self.n_steps-1,)
        self.EN0_step = np.zeros(self.n_steps-1,)
        self.EN1_step = np.zeros(self.n_steps-1,)
        self.Fidelity_step = np.zeros(self.n_steps-1,)
        self.Pn_step = np.zeros(self.n_steps-1,)
        self.Mn_step = np.zeros(self.n_steps-1,)
        
        return observation
    
    def step(self, action):
        
        self.numStep += 1
        # print('numStep ',self.numStep)
        Delta = action[0]
        alpha_L = action[1]
        H_0 = - Delta * self.a.dag() * self.a + self.b.dag() * self.b
        H_F = alpha_L * (self.a.dag() + self.a)
     
        opt = Options()
        opt.store_states = True
        
        result = smesolve(H_0 + self.H_coupling + H_F, self.rho_AB_0, times =    
                          np.linspace(0,self.dt,self.N_t), \
                              c_ops = [np.sqrt(self.kappa)*self.a, np.sqrt(self.gamma)*self.b], 
                          sc_ops = [np.sqrt(self.measure_rate)*self.P_0, np.sqrt(self.measure_rate)*self.P_1, 
                          np.sqrt(self.measure_rate)*self.P_2, np.sqrt(self.measure_rate)*self.P_3, 
                          np.sqrt(self.measure_rate)*self.P_4, np.sqrt(self.measure_rate)*self.P_5, 
                          np.sqrt(self.measure_rate)*self.P_6, np.sqrt(self.measure_rate)*self.P_7, 
                          np.sqrt(self.measure_rate)*self.P_8, np.sqrt(self.measure_rate)*self.P_9],
                      e_ops = [self.pn,self.mn], ntraj=1, solver="taylor1.5", tol = 1e-6,
                      m_ops=[], dW_factors=[], map_func=parallel_map, options=opt)
        self.rho_AB_0 = result.states[0][self.N_t-1].copy() # end of time
        Pn = result.expect[0][0]
        Mn = result.expect[1][0]
        
        Fidelity = fidelity(self.rho_AB_0, self.rho_target)
        E_N_0 = np.real(self.log_neg(result.states[0][0], 0))
        E_N_1 = np.real(self.log_neg(result.states[0][0], 1))

        observation = [E_N_0]#EN
        reward = -abs(E_N_0-np.log(2))-abs(Pn+Mn-self.ra)/self.rb
        
        self.Delta[self.numStep-2] = Delta
        self.alpha_L[self.numStep-2] = alpha_L
        self.EN0_step[self.numStep-2] = E_N_0
        self.EN1_step[self.numStep-2] = E_N_1
        self.Fidelity_step[self.numStep-2] = Fidelity
        self.Pn_step[self.numStep-2] = Pn
        self.Mn_step[self.numStep-2] = Mn
        
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
                     np.save(self.rundir+'/dataImp/Delta_'+str(self.numEpisode)+'.npy',self.Delta)
                     np.save(self.rundir+'/dataImp/alpha_L_'+str(self.numEpisode)+'.npy',self.alpha_L)
                     np.save(self.rundir+'/dataImp/EN0_step_'+str(self.numEpisode)+'.npy',self.EN0_step)
                     np.save(self.rundir+'/dataImp/EN1_step_'+str(self.numEpisode)+'.npy',self.EN1_step)
                     np.save(self.rundir+'/dataImp/Fidelity_step_'+str(self.numEpisode)+'.npy',self.Fidelity_step)
                     np.save(self.rundir+'/dataImp/Pn_step_'+str(self.numEpisode)+'.npy',self.Pn_step)
                     np.save(self.rundir+'/dataImp/Mn_step_'+str(self.numEpisode)+'.npy',self.Mn_step)

                     
                     num = 0
                     num += 1
                     plt.figure(num)
                     plt.bar(np.arange(0, self.N), np.real(result.states[0][self.N_t-1].ptrace(0).diag()))
                     plt.xlabel('Fock basis',fontsize=16)
                     plt.ylabel('Prob 0',fontsize=16)
                     plt.yticks(fontsize=16)
                     plt.xticks(fontsize=16)
                     plt.savefig(self.rundir+'/picture/Fock0_numEpisode_'+str(self.numEpisode)+'.png',bbox_inches='tight')
                     plt.close()
                     
                     num += 1
                     plt.figure(num)
                     plt.bar(np.arange(0, self.N), np.real(result.states[0][self.N_t-1].ptrace(1).diag()))
                     plt.xlabel('Fock basis',fontsize=16)
                     plt.ylabel('Prob 1',fontsize=16)
                     plt.yticks(fontsize=16)
                     plt.xticks(fontsize=16)
                     plt.savefig(self.rundir+'/picture/Fock1_numEpisode_'+str(self.numEpisode)+'.png',bbox_inches='tight')
                     plt.close()

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
 
                     num += 1
                     plt.figure(num)
                     plt.plot(0.01*np.arange(0,self.n_steps-1,1),self.Delta,linewidth=3,label='Delta') 
                     plt.plot(0.01*np.arange(0,self.n_steps-1,1),self.alpha_L,linewidth=3,label='alpha_L') 
                     plt.legend(fontsize=16)
                     plt.xlabel('times',fontsize=16)
                     plt.ylabel('laser control',fontsize=16)
                     plt.yticks(fontsize=16)
                     plt.xticks(fontsize=16)
                     plt.savefig(self.rundir+'/picture/laser_control_'+str(self.numEpisode)+'.png',bbox_inches='tight')
                     plt.close()
  
                     num += 1
                     plt.figure(num)
                     plt.plot(0.01*np.arange(0,self.n_steps-1,1),self.Fidelity_step,linewidth=3) 
                     plt.xlabel('times',fontsize=16)
                     plt.ylabel('Fidelity',fontsize=16)
                     plt.yticks(fontsize=16)
                     plt.xticks(fontsize=16)
                     plt.savefig(self.rundir+'/picture/Fidelity_'+str(self.numEpisode)+'.png',bbox_inches='tight')
                     plt.close()
                     
                     num += 1
                     plt.figure(num)
                     plt.plot(0.01*np.arange(0,self.n_steps-1,1),self.Pn_step,linewidth=3,label='Pn') 
                     plt.plot(0.01*np.arange(0,self.n_steps-1,1),self.Mn_step,linewidth=3,label='Mn')
                     plt.legend(fontsize=16)
                     plt.xlabel('times',fontsize=16)
                     plt.ylabel('exp current',fontsize=16)
                     plt.yticks(fontsize=16)
                     plt.xticks(fontsize=16)
                     plt.savefig(self.rundir+'/picture/Pn_Mn_'+str(self.numEpisode)+'.png',bbox_inches='tight')
                     plt.close()
                     
                     
                     
                   

           self.numEpisode +=1
           #print('episode #',str(self.numEpisode))
        
            
        return observation, reward, self.done, {0:E_N_1,1:Pn,2:Mn,3:Fidelity}
    
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