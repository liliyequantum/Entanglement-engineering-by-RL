# entanglement_engineering

## Setup

To set up the environment, open the Anaconda Prompt and run the following commands:

```sh
conda create -n qutip_RL python=3.9
pip install -r requirements.txt
```

## File Placement
1. Put all files from the directory `./cust_env/classical_control/` into the `qutip_RL` environment directory at `C:/users/yourUserName/anaconda3/envs/qutip_RL/Lib/site-packages/gym/envs/classic_control/`.

2. Copy the file from `./cust_env/__init__.py` and paste it into `C:/users/yourUserName/anaconda3/envs/qutip_RL/Lib/site-packages/gym/envs/__init__.py`, replacing the original file.

Open spyder by the command
```bibtex
> spyder
```
All code files with the suffix 'Fig_plot' are used for plotting figures, while files with the suffix 'Fig_code' are used for generating data.

Stable-baselines3 for the PPO agent [https://stable-baselines3.readthedocs.io/en/master/index.html]

Sb3-contrib for the recurrent PPO agent [https://sb3-contrib.readthedocs.io/en/master/index.html]



