import gym
from stable_baselines3 import PPO
import os
from snakeenv import SnakeEnv

TIMESTEPS = 10000

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = SnakeEnv()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

for i in range(1, 5):
    model.learn(total_timesteps=TIMESTEPS, 
                reset_num_timesteps=False,
                tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()