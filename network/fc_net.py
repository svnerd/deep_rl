import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def _make_fc_net(input_size, output_size, hidden_sizes, layer_x=-1, x_size=-1):
    if len(hidden_sizes) < 1 or hidden_sizes is None:
        raise Exception("no hidden layer specified")
    input_layer = nn.Linear(input_size, hidden_sizes[0])
    layers = []
    for i in range(0, len(hidden_sizes)-1):
        h0 = hidden_sizes[i]
        h1 = hidden_sizes[i+1]
        if i == layer_x:
            if (x_size < 0):
                raise Exception("bad inject size")
            h0 += x_size
        print("nn.Linear({}, {})".format(h0, h1))
        layers.append(nn.Linear(h0, h1))

    hidden_layers = nn.ModuleList(layers)
    output_layer = nn.Linear(hidden_sizes[-1], output_size)
    return input_layer, hidden_layers, output_layer


def _init_fc_net(input_layer, hidden_layers, output_layer):
    input_layer.weight.data.uniform_(*_hidden_init(input_layer))
    for layer in hidden_layers:
        layer.weight.data.uniform_(*_hidden_init(layer))
    output_layer.weight.data.uniform_(-3e-3, 3e-3)


def _dummy_func(x):
    return x


def _fc_net_forward(input_layer, hidden_layers, output_layer, input,
                    output_func=_dummy_func, layer_x=-1, input_x=None):
    x = F.relu(input_layer(input))
    hidden_layer_id = 0
    for layer in hidden_layers:
        if hidden_layer_id == layer_x:
            x = torch.cat((x, input_x), dim=1)
        x = F.relu(layer(x))
    return output_func(output_layer(x))

class FCNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, random_seed,
                 output_func=_dummy_func):
        super(FCNet, self).__init__()
        torch.manual_seed(random_seed)
        self.input_layer, self.hidden_layers, self.output_layer = _make_fc_net(
            input_size, output_size, hidden_sizes
        )
        self.output_func = output_func
        self.reset_parameters()

    def reset_parameters(self):
        _init_fc_net(self.input_layer, self.hidden_layers, self.output_layer)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.input_layer(state))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return self.output_func(x)


class FCNetInjectX(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, layer_x, size_x, random_seed,
                 output_func=_dummy_func):
        super(FCNetInjectX, self).__init__()
        torch.manual_seed(random_seed)
        self.layer_x = layer_x
        self.input_layer, self.hidden_layers, self.output_layer = _make_fc_net(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            layer_x=layer_x,
            x_size=size_x
        )
        self.reset_parameters()
        self.output_func = output_func

    def reset_parameters(self):
        _init_fc_net(self.input_layer, self.hidden_layers, self.output_layer)

    def forward(self, state, action):
        return _fc_net_forward(
            input_layer=self.input_layer,
            hidden_layers=self.hidden_layers,
            output_layer=self.output_layer,
            input=state,
            output_func=self.output_func,
            layer_x=self.layer_x,
            input_x=action
        )