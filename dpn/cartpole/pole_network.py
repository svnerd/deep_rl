import torch
import torch.nn as nn
import torch.nn.functional as F
import drl.util.device as D

LR = 5e-4

class PoleNetwork:
    def __init__(self, state_size, action_size, seed=6):
        self.network = TorchNetwork(state_size, action_size, seed)



    def estimate(self, output_tensor):
        pass

    def forward(self, state):
        return self.network.forward(D.float_to_device(state))


class TorchNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(TorchNetwork, self).__init__()
        torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        internal_state = [128, 32]
        self.fc1 = nn.Linear(state_size, internal_state[0])
        self.fc2 = nn.Linear(internal_state[0], internal_state[1])
        self.fc3 = nn.Linear(internal_state[1], action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)