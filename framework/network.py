import torch.nn as nn
import torch.nn.functional as F
import torch

"""
NOTE:
all class variables should be either Module or Module list so that
the class object's parameters() function can recursively includes 
all modules' parameters(). 
"""
class FCNet(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(FCNet, self).__init__()
        dims = [input_dim] + hidden_units
        self.layers = []
        for i in range(1, len(dims)):
            print("nn.Linear({}, {})".format(dims[i-1], dims[i]))
            self.layers.append(nn.Linear(dims[i-1], dims[i]))
        self.layers = nn.ModuleList(self.layers)
        self.feature_dim = hidden_units[-1]

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

class FCActInjected1Net(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_units):
        super(FCActInjected1Net, self).__init__()
        if (len(hidden_units) < 2):
            raise Exception("it need to contain 2 layers at least!")
        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_units[0]))
        self.layers.append(nn.Linear(hidden_units[0] + action_dim, hidden_units[1]))
        dims = hidden_units[2:]
        for i in range(1, len(dims)):
            self.layers.append(nn.Linear(dims[i-1], dims[i]))
        self.layers = nn.ModuleList(self.layers)
        self.feature_dim = hidden_units[-1]

    def forward(self, x, action):
        cnt = 0
        for layer in self.layers:
            if cnt == 1:
                x = torch.cat([x, action], dim=1)
            x = F.relu(layer(x))
            cnt += 1
        return x

class PassthroughNet(nn.Module):
    def __init__(self, dim):
        super(PassthroughNet, self).__init__()
        self.feature_dim = dim

    def forward(self, x):
        return x