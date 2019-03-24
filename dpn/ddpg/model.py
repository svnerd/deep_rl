"""
Example Neural Network Model for Vector Observation DDPG (Actor-Critic) Agent
DDPG Model for Unity ML-Agents Environments using PyTorch

Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code expanded and adapted from code examples provided by Udacity DRL Team, 2018.
"""


import torch
import torch.nn as nn
from drl.network.fc_net import FCNet, FCNetInjectX
from drl.util.device import DEVICE


class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
        super(Actor, self).__init__()
        self.net = FCNet(state_size, action_size, [fc1_units, fc2_units], output_func=torch.tanh)
        self.to(DEVICE)

    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, state):
        return self.net.forward(state)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, fcs1_units=256, fc2_units=128):
        super(Critic, self).__init__()
        self.net = FCNetInjectX(
            input_size=state_size,
            output_size=1,
            hidden_sizes=[fcs1_units, fc2_units],
            layer_x=0, size_x=action_size
        )
        self.to(DEVICE)

    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, state, action):
        return self.net.forward(state, action)
