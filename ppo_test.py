import numpy as np
import gymnasium as gym
from PPO import PPOAgent, PPOModel

env = gym.make('Acrobot-v1', render_mode='human')
action_space = env.action_space.n

model = PPOModel(env.action_space.n)
agent = PPOAgent(model)
agent.test(env)
