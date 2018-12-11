import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import deque, namedtuple

EVERY_4_STEPS = 4
LR = 5e-4
TAU = 1e-3              # for soft update of target parameters

class NavigationBrainAgent():
    def __init__(self, state_size, action_size):
        self.target_model = NavigationDQNModel(state_size, action_size).to(DEVICE)
        self.local_model = NavigationDQNModel(state_size, action_size).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=LR)
        self.action_size = action_size
        self.memory = ExperienceMemory()
        self.steps = 0

    def observe(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.steps += 1
        if self.steps % EVERY_4_STEPS == 0:
            experiences = self.memory.sample()
            if experiences == None:
                return
            self.__learn(experiences)

    def act(self, state, esp=0.0):
        if random.random() > esp:
            state_on_device = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            self.local_model.eval()
            with torch.no_grad():
                action = self.local_model.forward(state_on_device).max(dim=1)[0].float()
            self.local_model.train()
        else:
            action = random.choice(np.arange(self.action_size))
        return action

    def __learn(self, experiences, gamma=0.99): # nothing fancy, just the 2015 dqn
        states, actions, rewards, next_states, dones = experiences
        # use local to calculate expected Q
        max_next_state_q = self.target_model.forward(next_states).max(dim=1)[0].unsqueeze(-1)
        target_q = rewards + gamma * max_next_state_q * (1-dones)
        estimate_q = self.local_model.forward(states).gather(1, actions)
        loss = F.mse_loss(estimate_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.__soft_update()

    def __soft_update(self, tau=TAU):
        for target_param, local_param in zip(self.target_model.parameters(), self.local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def float_to_device(np_arr):
    return torch.from_numpy(np_arr).float().to(DEVICE)

def long_to_device(np_arr):
    return torch.from_numpy(np_arr).long().to(DEVICE)

class ExperienceMemory:
    def __init__(self, msize=int(1e5)):
        self.memory = deque(maxlen=msize)
        random.seed(78)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None
        samples = random.sample(self.memory, k=batch_size)
        return float_to_device(np.vstack([s.state for s in samples])), \
               long_to_device(np.vstack([s.action for s in samples])), \
               float_to_device(np.vstack([s.reward for s in samples])), \
               float_to_device(np.vstack([s.next_state for s in samples])), \
               float_to_device(np.vstack([s.done for s in samples]).astype(np.uint8))


# this is a super simple 3-layer NN
class NavigationDQNModel(nn.Module):
    def __init__(self, state_size, action_size, seed=58):
        super(NavigationDQNModel, self).__init__()
        self.seed = torch.manual_seed(seed)

        hidden_size1 = 256
        self.fc1 = nn.Linear(in_features=state_size, out_features=hidden_size1)
        self.relu1 = nn.ReLU()

        hidden_size2 = 64
        self.fc2 = nn.Linear(in_features=hidden_size1, out_features=hidden_size2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(in_features=hidden_size2, out_features=action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return self.fc3(x)
