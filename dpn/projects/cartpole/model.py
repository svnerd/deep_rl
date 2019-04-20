import torch.nn as nn
from deep_rl.network.fc_net import FCNet
from deep_rl.network.softmax import Softmax
from .constant import RANDOM_SEED, MAX_STEPS
from deep_rl.util.device import DEVICE

import torch

class CategoricalActorCritic(nn.Module):
    def __init__(self, state_size, action_size, shared_feature_size, fc_units):
        super(CategoricalActorCritic, self).__init__()

        self.shared_net = FCNet(
            input_size=state_size,
            output_size=shared_feature_size,
            hidden_sizes=fc_units,
            random_seed=RANDOM_SEED
        )
        self.actor = Softmax(feature_size=shared_feature_size, action_size=action_size)
        self.critic = nn.Linear(shared_feature_size, 1)
        self.to(DEVICE)

    def forward(self, states):
        features = self.shared_net.forward(states)
        actions, log_prob = self.actor.forward(features=features)
        critic_v = self.critic.forward(features)
        return actions, log_prob, critic_v


class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=32):
        super(Actor, self).__init__()
        self.net = FCNet(state_size, action_size, [fc1_units, fc2_units],
                         random_seed=RANDOM_SEED)
        self.to(DEVICE)

    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, state):
        logits = self.net.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() # action.shape = (N)
        log_prob = dist.log_prob(action) # log_prob.shape = (N)
        #entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return action.unsqueeze(-1), log_prob.unsqueeze(-1)

class Critic(nn.Module):
    def __init__(self, state_size, fc1_units=64, fc2_units=32):
        super(Critic, self).__init__()
        self.net = FCNet(state_size, 1, [fc1_units, fc2_units],
                         random_seed=RANDOM_SEED)
        self.to(DEVICE)

    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, state_action):
        return self.net.forward(state_action)
