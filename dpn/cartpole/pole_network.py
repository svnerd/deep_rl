import torch
import torch.nn as nn
import torch.nn.functional as F
import drl.util.device as D
from torch.distributions import Categorical
LR = 1e-3

class PoleNetwork:
    def __init__(self, state_size, action_size, seed=6):
        self.network = TorchNetwork(state_size, action_size, seed).to(D.DEVICE)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

    def forward(self, state, action_int=None):
        state = D.float_to_device(state)
        probs = self.network.forward(state).cpu()
        cat_probs = Categorical(probs)
        if action_int is None:
            action = cat_probs.sample()
            action_int = action.item()
        else:
            action = torch.tensor(action_int)
        return action_int, probs.gather(1, action.reshape(-1, 1)), cat_probs.log_prob(action)

    def correct(self, losses):
        policy_loss = torch.cat(losses).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def manual_update_param(self, new_params, tau=1.0):
        for target_param, new_params in zip(self.network.parameters(), new_params):
            target_param.data.copy_(tau*new_params.data + (1.0-tau)*target_param.data)

    def get_params(self):
        return self.network.parameters()

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