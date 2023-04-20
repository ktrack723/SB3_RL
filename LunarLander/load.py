import gym
from stable_baselines3 import A2C
import os

env = gym.make("LunarLander-v2")
env.reset()

models_dir = "models/A2C"
model_path = f"{models_dir}/60000.zip"

model = A2C.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
    
env.close()