import numpy as np
import gymnasium as gym
from A2C import A2CAgent, A2CModel

env = gym.make('Acrobot-v1', render_mode='human')
action_space = env.action_space.n

model = A2CModel(env.action_space.n)
agent = A2CAgent(model)
agent.test(env)
