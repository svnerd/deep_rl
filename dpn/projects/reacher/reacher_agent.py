
import numpy as np
import torch
import torch.nn.functional as F

from deep_rl.dpn.projects.reacher.model import Actor, Critic
from deep_rl.network.param import soft_update
from deep_rl.util.noise import OUNoise
from deep_rl.util.replay_buffer import ExperienceMemory
from deep_rl.util.save_restore import SaveRestoreService

from .constant import RANDOM_SEED, BATCH_SIZE

BUFFER_SIZE = int(1e5)  # replay buffer size
DISCOUNT_RATE = 0.99
TAU = 1e-3              # for soft update of target parameters
WEIGHT_DECAY = 0.0      # L2 weight decay
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4       # learning rate of the critic

FC1 = 256
FC2 = 128


def _make_actor_critic_net(env):
    actor_net =  Actor(
        state_size=env.obs_dim,
        action_size=env.act_dim,
        fc1_units=FC1, fc2_units=FC2,
    )
    critic_net = Critic(
        state_size=env.obs_dim,
        action_size=env.act_dim
    )
    return actor_net, critic_net


class ReacherAgent:
    def __init__(self, reacher_env, dim_tensor_maker, batch_size, record_dir):
        self.actor_net, self.critic_net = _make_actor_critic_net(reacher_env)
        self.actor_target, self.critic_target = _make_actor_critic_net(reacher_env)
        #soft_update(self.actor_target, self.actor_net)
        #soft_update(self.critic_target, self.critic_net)

        self.env = reacher_env
        self.batch_size = batch_size
        self.dtm = dim_tensor_maker
        self.critic_optimizer = torch.optim.Adam(params=self.critic_net.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.actor_optimizer = torch.optim.Adam(params=self.actor_net.parameters(), lr=LR_ACTOR)
        self.noise = OUNoise((reacher_env.num_agents, reacher_env.act_dim), seed=RANDOM_SEED)
        self.memory = ExperienceMemory(batch_size=batch_size, msize=BUFFER_SIZE)

        network_dict = {
            "actor_net": self.actor_net,
            "actor_target": self.actor_target,
            "critic_net": self.critic_net,
            "critic_target": self.critic_target,
        }
        self.sr_service = SaveRestoreService(record_dir, network_dict)
        self.sr_service.restore()

    def act(self, obs):
        num_agents = self.env.num_agents
        action_size = self.env.act_dim
        actions = np.zeros((num_agents, action_size))
        self.actor_net.eval()
        with torch.no_grad():
            for agent in range(num_agents):
                actions[agent,:] = self.actor_net(obs[agent,:]).cpu().data.numpy()
            #actions = self.actor_net(obs).cpu().data.numpy()
        self.actor_net.train()
        actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def __learn(self):
        experiences = self.memory.sample(self.dtm)
        if experiences is None:
            return
        b_states_t, b_actions_t, b_rewards_t, b_next_states_t, b_dones_t = experiences

        # ----- establish critic baseline --------
        a_next_t = self.actor_target.forward(b_next_states_t)
        self.dtm.check_agent_out(a_next_t)
        q_next_t = self.critic_target.forward(b_next_states_t, a_next_t)
        q_target_t = b_rewards_t + DISCOUNT_RATE * (1-b_dones_t) * q_next_t
        assert(q_target_t.shape[0]==BATCH_SIZE and q_target_t.shape[1]==1)
        q_target_t.detach()

        # ----- optimize away critic loss ---------
        q_local_t = self.critic_net.forward(b_states_t, b_actions_t)
        self.dtm.check_agent_out(q_local_t)
        critic_loss = F.mse_loss(q_local_t, q_target_t)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 1)
        self.critic_optimizer.step()

        # ----- according to the paper, the critic networks' deriv is to be used to update actor network
        actions_t = self.actor_net.forward(b_states_t)
        actor_loss = -self.critic_net.forward(b_states_t, actions_t).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.1)
        self.actor_optimizer.step()
        soft_update(self.actor_target, self.actor_net, tau=TAU)
        soft_update(self.critic_target, self.critic_net, tau=TAU)

    def update(self, states, actions, next_states, rewards, dones, episode):
        for agent in range(self.env.num_agents):
            self.memory.add(states[agent,:], actions[agent,:], rewards[agent], next_states[agent,:], dones[agent])
        self.__learn()
        if np.any(dones) and episode % 20 == 0:
            self.sr_service.save()
