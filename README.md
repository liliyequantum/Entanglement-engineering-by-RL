# entanglement_engineering

## Setup

To set up the environment, open the Anaconda Prompt and run the following commands:

```sh
conda create -n qutip_RL python=3.9
pip install -r requirements.txt

Put all files in the directory of './cust_env/classical_control/' to the qutip_RL env directory 'C:/users/yourUserName/anaconda3/envs/qutip_RL/Lib/site-packages/gym/envs/classic_control/'

Copy the file from the location './cust_env/__init__.py' and paste to 'C:/users/yourUserName/anaconda3/envs/qutip_RL/Lib/site-packages/gym/envs/__init__.py', replace the original one.

Open spyder by the command
```bibtex
> spyder
```
All code files with the suffix 'Fig_plot' are used for plotting figures, while files with the suffix 'Fig_code' are used for generating data.

Stable-baselines3 for the PPO agent [https://stable-baselines3.readthedocs.io/en/master/index.html]

Sb3-contrib for the recurrent PPO agent [https://sb3-contrib.readthedocs.io/en/master/index.html]



