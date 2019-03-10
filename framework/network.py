import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from drl.util.device import DEVICE
"""
NOTE:
all class variables should be either Module or Module list so that
the class object's parameters() function can recursively includes 
all modules' parameters(). 
"""


def _hidden_init(layer, scale):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim * scale, lim * scale)


def _init_layers(layers, scale=1.0):
    for layer in layers:
        layer.weight.data.uniform_(*_hidden_init(layer, scale))


def _init_output_layer(layer, scale=1.0):
    layer.weight.data.uniform_(-3e-3 * scale, 3e-3 * scale)


def _make_hidden_layers(input_dim, hidden_units):
    dims = [input_dim] + hidden_units
    layers = []
    for i in range(1, len(dims)):
        print("nn.Linear({}, {})".format(dims[i-1], dims[i]))
        layers.append(nn.Linear(dims[i-1], dims[i]))
    _init_layers(layers)
    return nn.ModuleList(layers)


class FCNet(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(FCNet, self).__init__()
        self.layers = _make_hidden_layers(input_dim, hidden_units)
        self.feature_dim = hidden_units[-1]

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


class FCNetOutputLayer(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(FCNetOutputLayer, self).__init__()
        self.layers = _make_hidden_layers(input_dim, hidden_units)
        self.output_layer = nn.Linear(hidden_units[-1], output_dim)
        self.to(DEVICE)

    def forward(self, x):
        for layers in self.layers:
            x = F.relu(layers(x))
        return self.output_layer(x)


def _make_hidden_layers_with_action_input(input_dim, action_dim, hidden_units):
    if (len(hidden_units) < 2):
        raise Exception("it need to contain 2 layers at least!")
    layers = []
    layers.append(nn.Linear(input_dim, hidden_units[0]))
    layers.append(nn.Linear(hidden_units[0] + action_dim, hidden_units[1]))
    dims = hidden_units[2:]
    for i in range(1, len(dims)):
        layers.append(nn.Linear(dims[i-1], dims[i]))
    _init_layers(layers)
    return nn.ModuleList(layers)


def _forward_with_action(layers, x, action):
    cnt = 0
    for layer in layers:
        if cnt == 1:
            x = torch.cat([x, action], dim=1)
        x = F.relu(layer(x))
        cnt += 1
    return x


class FCActInjected1Net(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_units):
        super(FCActInjected1Net, self).__init__()
        self.layers = _make_hidden_layers_with_action_input(input_dim, action_dim, hidden_units)
        self.feature_dim = hidden_units[-1]

    def forward(self, x, action):
        return _forward_with_action(self.layers, x, action)


class FCActInjected1NetOutputLayer(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_units, output_dim):
        super(FCActInjected1NetOutputLayer, self).__init__()
        self.layers = _make_hidden_layers_with_action_input(input_dim, action_dim, hidden_units)
        self.output_layer = nn.Linear(hidden_units[-1], output_dim)
        _init_output_layer(self.output_layer)
        self.to(DEVICE)

    def forward(self, x, action):
        x = _forward_with_action(self.layers, x, action)
        return self.output_layer(x)


class PassthroughNet(nn.Module):
    def __init__(self, dim):
        super(PassthroughNet, self).__init__()
        self.feature_dim = dim

    def forward(self, x):
        return x