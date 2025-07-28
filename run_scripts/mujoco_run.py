# This is a run script to train a RL agent using the SUFT method on MuJoCo environments

import gymnasium as gym
from stable_baselines3 import PPO, SAC

if __name__ == "__main__":
  # get run arguments
    # get the device (cuda/cpu)
    run_device = sys.argv[1]
    # get TF lambda value
    tf_lambda = float(sys.argv[2])
    # get the RL agent to train
    agent_name = sys.argv[3]
    # get the timestep to train
    timesteps = int(sys.argv[4])
    # get the environment name
    gym_env_name = sys.argv[5]
    # create the environment
    env = gym.make(gym_env_name, render_mode="rgb_array")
    if agent_name == 'PPO':
        model = PPO('MlpPolicy', env, device=run_device, tf_lambda=tf_lambda)
    elif agent_name == 'SAC':
        buffer_size = int(sys.argv[6])
        model = SAC('MlpPolicy', env, device=run_device, buffer_size=buffer_size, tf_lambda=tf_lambda)
    else:
        print("ERROR - no valid agent")
    # Train the agent
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
