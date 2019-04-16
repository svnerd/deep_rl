# this sets up the unity env and
from unityagents import UnityEnvironment
import numpy as np


class BananaEnv:
    def __init__(self):
        self.env = UnityEnvironment(
            file_name="/Users/chenyuan/project/ipython/deep_rl/dqn/projects/navigation/Banana.app")
        self.brain_name = self.env.brain_names[0] # 0 is the default brain
        self.brain = self.env.brains[self.brain_name]
        self.act_dim = self.brain.vector_action_space_size
        info = self.env.reset(train_mode=True)[self.brain_name]
        self.obs_dim = len(info.vector_observations[0]) # current observation
        print("env created with act_dim", self.act_dim, "obs_dim", self.obs_dim)

    def reset(self, train_mode=True):
        info = self.env.reset(train_mode)[self.brain_name]
        return info.vector_observations[0].reshape(1, -1)

    def step(self, action):
        info = self.env.step(action)[self.brain_name]
        reward = info.rewards[0]
        state = info.vector_observations[0]
        done = info.local_done[0]
        return state.reshape(1, -1), \
               np.reshape(reward, -1), \
               np.reshape(done, -1)

    def close(self):
        self.env.close()