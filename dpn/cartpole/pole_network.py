import torch
import torch.nn as nn
import torch.nn.functional as F
import drl.util.device as D
from torch.distributions import Categorical
LR = 5e-4

class PoleNetwork:
    def __init__(self, state_size, action_size, seed=6):
        self.network = TorchNetwork(state_size, action_size, seed).to(D.DEVICE)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

    def forward(self, state):
        probs = self.network.forward(D.float_to_device(state)).cpu()
        cat_probs = Categorical(probs)
        action = cat_probs.sample()
        print("action", action)
        print("action_item", action.item())
        return action.item(), cat_probs.log_prob(action)

    def correct(self, losses):
        policy_loss = torch.cat(losses).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


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
        return F.softmax(x)