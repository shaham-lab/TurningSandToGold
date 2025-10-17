# Turning Sand to Gold: Recycling Data to Bridge On-Policy and Off-Policy Learning via Causal Bound

## Poster at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)

## Authors 
- [Tal Fiskus](https://www.linkedin.com/in/talfiskus/)
- [Dr. Uri Shaham](https://www.linkedin.com/in/urishaham/)
- 
## Paper Link
- [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/119864)  
- [OpenReview](https://openreview.net/forum?id=5UtsjOGsDx)  
- [ArXiv](https://www.arxiv.org/abs/2507.11269)  
- [Project Page](https://shaham-lab.github.io/TurningSandToGold/)  

## Code Overview
Our code is built on the Stable Baselines3 library.

This repository contains the **code modifications** to the Stable Baselines3 library necessary to implement our proposed **SUFT method**, as well as the **run scripts** to reproduce the experiments presented in the paper.

We used **Python 3.11.10**.

### Modified Files from Stable Baselines3:
Our implementation modifies six Python files from the Stable Baselines3 library:
1. buffers.py
2. dqn.py
3. off_policy_algorithm.py
4. ppo.py
5. sac.py
6. type_aliases.py

Modifications are indicated with **# SUFT CHANGE** in our code.

### Additional RL Agents:
In addition, there are two Python files with additional RL agents that are not included in Stable Baselines3:
1. regular_dqn.py
2. ddqn.py

## Run instructions
### Setup
1. Clone and set up the Stable Baselines3 library
2. Implement the modified code of the six Python files from this repository. Please note the **# SUFT CHANGE** comment in the code to add the relevant code modifications.
3. (Optional) Implement the additional RL agents.

### Run Scripts
To train the RL agent using the SUFT method, we created two Python run files:
1. Atari: atari_run.py.
2. MuJoCo: mujoco_run.py.

### Examples:
#### Atari PPO Example
Run script: 
```
python atari_run.py "cuda:0" 5.0 "PPO" 400000 42 "ALE/Alien-v5"
```

#### Atari Vanilla DQN / Double DQN / Regular DQN Example
Run script: 
```
python atari_run.py "cuda:0" 1.0 "DDQN" 400000 42 "ALE/Alien-v5" 4000
```
Arguments details:

| Argument       | Script Example Value    | Description                                                     |
|----------------|-------------------------|-----------------------------------------------------------------|
| `run_file`     | `atari_run.py`          | Training run script                                             |
| `run_device`   | `"cuda:0"`              | The device on which the code should be run (specific CUDA GPU)  |
| `tf_lambda`    | `1.0`                   | SUFT OPE term λ hyper-parameter                                 |
| `agent_name`   | `"DDQN"`                | RL agent (`"PPO"`, `"DQN"`, `"DDQN"`, `"DQN1"`)                 |
| `timesteps`    | `400000`                | Total training steps (400K)                                     |
| `random_seed`  | `42`                    | Random seed                                                     |
| `gym_env_name` | `"ALE/Alien-v5"`        | Gymnasium environment (any Atari game)                          |
| `buffer_size`  | `4000`                  | Training buffer size                                            |

### MuJoCo PPO Example:
Run script: 
```
python mujoco_run.py "cpu" 1.8 "PPO" 1000000 "Walker2d-v4"
```

#### MuJoCo SAC Example   
Run script: 
```
python mujoco_run.py "cpu" 0.6 "SAC" 1000000 "Walker2d-v4" 4000
```

Arguments details:

| Argument       | Script Example Value    | Description                                                     |
|----------------|-------------------------|-----------------------------------------------------------------|
| `run_file`     | `mujoco_run.py`         | Training run script                                             |
| `run_device`   | `"cpu"`                 | The device on which the code should be run (specific CUDA GPU)  |
| `tf_lambda`    | `0.6`                   | SUFT OPE term λ hyper-parameter                                 |
| `agent_name`   | `"SAC"`                 | RL agent (`"PPO"`, `"SAC"`)                                     |
| `timesteps`    | `1000000`               | Total training steps (1M)                                       |
| `gym_env_name` | `"Walker2d-v4"`         | Gymnasium environment (any MuJoCo environment)                  |
| `buffer_size`  | `4000`                  | Training buffer size                                            |

## References
[1] Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., and Dormann, N. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research, 22(268):1–8, 2021.
