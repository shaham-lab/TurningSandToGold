# This is a run script to train a RL agent using the SUFT method on Atari games
import gymnasium as gym

from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO, DQN, DDQN, DQN1
from stable_baselines3.common.env_util import make_atari_env

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
    # get the random seed
    random_seed = int(sys.argv[5])
    # get the environment name
    gym_env_name = sys.argv[6]
    
    # create the atari environment
    env = make_atari_env(gym_env_name, seed=random_seed)
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)
    # select agent by name
    if agent_name == 'PPO':
        model = PPO('CnnPolicy', env, device=run_device, tf_lambda=tf_lambda)
    elif agent_name == 'DQN':
        buffer_size = int(sys.argv[7])
        model = DQN('CnnPolicy', env, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size)
    elif agent_name == 'DDQN':
        buffer_size = int(sys.argv[7])
        model = DDQN('CnnPolicy', env, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size)
    elif agent_name == 'DQN1':
        buffer_size = int(sys.argv[7])
        model = DQN1('CnnPolicy', env, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size)

    else:
        print("ERROR - no valid agent")
    
    # Train the agent
    model.learn(total_timesteps=timesteps) #, callback=eval_callback)

  
