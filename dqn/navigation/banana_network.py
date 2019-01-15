import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 5e-4

def float_to_device(np_arr):
    return torch.from_numpy(np_arr).float().to(DEVICE)

def long_to_device(np_arr):
    return torch.from_numpy(np_arr).long().to(DEVICE)


class BananaNetwork:
    def __init__(self, state_size, action_size, need_optimizer=True, seed=58):
        self.network = TorchNetwork(state_size, action_size, seed).to(DEVICE)
        if need_optimizer:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        else:
            self.optimizer = None

    def forward(self, states):
        states_on_device = float_to_device(states)
        return self.network.forward(states_on_device)

    def estimate(self, states):
        states_on_device = float_to_device(states)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network.forward(states_on_device)
        self.network.train()
        return action_values, action_values.cpu().data.numpy()

    def correct(self, now_v, target_v):
        loss = F.mse_loss(now_v, target_v)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def manual_update_param(self, new_params, tau=1.0):
        for target_param, new_params in zip(self.network.parameters(), new_params):
            target_param.data.copy_(tau*new_params.data + (1.0-tau)*target_param.data)

    def get_params(self):
        return self.network.parameters()

# this is a super simple 3-layer NN
# function approximator to map from (state to action probability)
class TorchNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(TorchNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = [256, 64]
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(in_features=state_size, out_features=self.hidden_sizes[0])
        self.fc2 = nn.Linear(in_features=self.hidden_sizes[0], out_features=self.hidden_sizes[1])
        self.fc3 = nn.Linear(in_features=self.hidden_sizes[1], out_features=action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

    def save(self, check_point):
        model_state = {
            'state_size'    : self.state_size,
            'action_size'   : self.action_size,
            'state_dict'    : self.state_dict()
        }
        torch.save(model_state, check_point)

    @staticmethod
    def load(self, check_point):
        model_state = torch.load(check_point)
        func = TorchNetwork(model_state['state_size'], model_state['action_size'], seed=58)
        func.load_state_dict(model_state['state_dict'])
        return func