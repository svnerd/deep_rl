import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from drl.dpn.agent.naive_ddpg import DDPGAgent
from drl.framework.config import DDPGConfig
from drl.util.score_tracker import ScoreTracker
from drl.util.noise import OUNoise

env = gym.make('Pendulum-v0')
env.seed(2)
config = DDPGConfig()
config.env_driver = env

config.soft_update_tau = 1e-3
config.discount = 0.99
config.score_tracker = ScoreTracker(good_target=100, window_len=100)
config.noise = [OUNoise(1, seed=2)] * 1


agent = DDPGAgent(config)

while True:
    if agent.step():
        break

