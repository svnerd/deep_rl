import random
import numpy as np
from torch import optim
import torch.nn.functional as F

from deep_rl.util.replay_buffer import ExperienceMemory
from deep_rl.util.device import tensor_long
from deep_rl.network.param import soft_update
from deep_rl.network.fc_net import FCNet
from deep_rl.util.eps_decay import SoftEpsilonDecay
from .constant import RANDOM_SEED, BATCH_SIZE

BUFFER_SIZE = int(1e5)  # replay buffer size
NAV_LR = 5e-4
DISCOUNT_RATE = 0.99

class NavAgent():
    def __init__(self, env, dim_maker):
        self.memory = ExperienceMemory(batch_size=BATCH_SIZE, msize=BUFFER_SIZE)
        self.env = env
        self.dim_maker = dim_maker
        self.eps_handler = SoftEpsilonDecay(1.0, 1e-3, 0.995)
        self.local_network = FCNet(env.obs_dim, env.act_dim, [128, 64], random_seed=RANDOM_SEED)
        self.target_network = FCNet(env.obs_dim, env.act_dim, [128, 64], random_seed=RANDOM_SEED)
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=NAV_LR)
        soft_update(target_net=self.target_network, local_net=self.local_network, tau=1.0)

    def act(self, obs):
        if random.random() >= self.eps_handler.eps:
            estimate_v_t = self.local_network.forward(obs)
            estimate_v = self.dim_maker.agent_out_to_np(estimate_v_t)
            action = np.argmax(estimate_v, axis=1)
        else:
            action = random.choice(np.arange(self.env.act_dim))
        action = np.reshape(action, (-1, 1))
        return action

    def update(self, states, actions, rewards, next_states, dones):
        self.eps_handler.decay()
        self.memory.add(states, actions, rewards, next_states, dones)
        if len(self.memory) < BATCH_SIZE:
            return
        b_states_t, b_actions_t, b_rewards_t, b_next_states_t, b_dones_t = self.memory.sample(self.dim_maker)
        vnext_target_t = self.target_network.forward(b_next_states_t)
        max_vnext_target_t = vnext_target_t.max(1)[0].reshape(-1, 1)
        vtarget_t = b_rewards_t + DISCOUNT_RATE * (1-b_dones_t) * max_vnext_target_t

        vlocal_t = self.local_network.forward(b_states_t)
        vlocal_t = vlocal_t.gather(1, (b_actions_t.long()))
        loss = F.mse_loss(vlocal_t, vtarget_t.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

