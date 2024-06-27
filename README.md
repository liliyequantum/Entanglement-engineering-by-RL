# entanglement_engineering

## System requirement

Windows 10, CPU

## Setup

To set up the environment, open the Anaconda Prompt and run the following commands:

```sh
conda create -n qutip_RL python=3.9
pip install -r requirements.txt
```

## File Placement
1. Put all files from the directory `.\cust_env\classical_control\` into the `qutip_RL` environment directory at `C:\users\yourUserName\anaconda3\envs\qutip_RL\Lib\site-packages\gym\envs\classic_control\`, replacing the original files.

2. Copy the file from `.\cust_env\__init__.py` and paste it into `C:\users\yourUserName\anaconda3\envs\qutip_RL\Lib\site-packages\gym\envs\__init__.py`, replacing the original file.

## Running the Code

Open spyder by running the following command in the Anaconda Prompt
```sh
spyder
```

## Code Files

1. Files with the suffix `Fig_plot` are used for plotting figures.

2. Files with the suffix `Fig_code` are used for generating data.

## References

1. Stable-baselines3 for the PPO agent: [https://stable-baselines3.readthedocs.io/en/master/index.html]

2. Sb3-contrib for the recurrent PPO agent: [https://sb3-contrib.readthedocs.io/en/master/index.html]

3. QuTip [https://qutip.readthedocs.io/en/master/index.html]

