import gym
from .constant import RANDOM_SEED
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box

import numpy as np

def _get_observation_dim(obs_space):
    if len(obs_space.shape) != 1:
        raise Exception("only support N-d observation space. ")
    return obs_space.shape[0]

def _get_action_dim(action_space):
    if isinstance(action_space, Discrete):
        return action_space.n
    elif isinstance(action_space, Box):
        if len(action_space.shape) != 1:
            raise Exception("only support N-d action space. ")
        return action_space.shape[0]

class CartPoleEnv():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.env.seed(RANDOM_SEED)
        self.obs_dim=self.env.observation_space.shape[0],
        self.act_dim=self.env.action_space.n,
        print("cartpole env obs_dim", self.env.observation_space, "act_dim", self.env.action_space)
        self.reset()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_states, rewards, dones, info = self.env.step(action)
        return np.reshape(next_states, (-1, 1)), \
               np.reshape(rewards, (-1, 1)), np.reshape(dones, (-1, 1)), info