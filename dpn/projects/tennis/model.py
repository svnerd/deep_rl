
import torch
import torch.nn as nn

from deep_rl.network.fc_net import FCNet, FCNetInjectX
from deep_rl.util.device import DEVICE
from .constant import RANDOM_SEED

class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
        super(Actor, self).__init__()
        self.net = FCNet(state_size, action_size, [fc1_units, fc2_units],
                         random_seed=RANDOM_SEED,
                         output_func=torch.tanh)
        self.to(DEVICE)

    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, state):
        return self.net.forward(state)


class Critic(nn.Module):
    def __init__(self, state_size, fc1_units=256, fc2_units=128):
        super(Critic, self).__init__()
        self.net = FCNet(state_size, 1, [fc1_units, fc2_units],
                         random_seed=RANDOM_SEED)
        self.to(DEVICE)

    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, state_action):
        return self.net.forward(state_action)
