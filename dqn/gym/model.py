import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        hidden_layer_size = 64
        self.fc1 = nn.Linear(in_features=state_size, out_features=hidden_layer_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=hidden_layer_size, out_features=action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return self.fc3(x)

