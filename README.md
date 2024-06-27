# entanglement_engineering

Under Anaconda Prompt window:
```bibtex
>conda create -n qutip_RL python=3.9
>pip install -r requirements.txt
```
Put files in the directory of './cust_env/classical_control/' to the qutip_RL env directory 'C:/users/yourUserName/anaconda3/envs/qutip_RL/Lib/site-packages/gym/envs/classic_control/'

Copy the file from the location './cust_env/__init__.py' and paste to 'C:/users/yourUserName/anaconda3/envs/qutip_RL/Lib/site-packages/gym/envs/__init__.py', replace the original one.

Open spyder by the command
```bibtex
> spyder
```

Stable-baselines3 for the PPO agent [https://stable-baselines3.readthedocs.io/en/master/index.html]

Sb3-contrib for the recurrent PPO agent [https://sb3-contrib.readthedocs.io/en/master/index.html]



