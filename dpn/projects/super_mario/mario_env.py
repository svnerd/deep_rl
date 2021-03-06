from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import numpy as np
import torch, cv2, time
import torchvision.transforms as transforms
from PIL import Image
from deep_rl.util.device import DEVICE


class MarioEnv:
    def __init__(self, os='mac', display=False):
        self.display = display
        if os == 'mac' or os == 'linux':
            env = gym_super_mario_bros.make('SuperMarioBros-v0')
            self.env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
        else:
            raise Exception("bad os")
        self.act_dim = self.env.action_space.n
        self.obs_dim = (1, 128, 128)
        print("env created with act_dim", self.act_dim, "obs_dim", self.obs_dim)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), # chain 2 transforms together using list.
             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def reset(self):
        state = self.env.reset()
        return self.__resize_image(state)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if reward == 0:
            reward = -0.5
        state_t = self.__resize_image(state)
        return state_t, \
               np.reshape(reward, -1), \
               np.reshape(done, -1)

    def close(self):
        self.env.close()

    def __resize_image(self, state):
        state_new = cv2.resize(state, (128, 128))
        img = Image.fromarray(state_new)
        state_t = self.transform(img)[0,:,:].unsqueeze(0)
        state_t = state_t.float().to(DEVICE)
        return state_t.unsqueeze(0)

    def render(self):
        if self.display:
            self.env.render()
