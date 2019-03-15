from drl.framework.network import FCNetOutputLayer, FCActInjected1NetOutputLayer
from drl.framework.buffer import ExperienceMemory
from drl.framework.buffer import ReplayBuffer

from drl.util.noise import OUNoise
from drl.dpn.ddpg.model import Actor, Critic
import torch.nn.functional as F
import torch
import numpy as np


BUFFER_SIZE = int(1e5)  # replay buffer size
DISCOUNT_RATE = 0.99
TAU = 1e-3              # for soft update of target parameters

def _make_actor_critic_net(env):
    actor_net =  FCNetOutputLayer(
        input_dim=env.obs_dim, hidden_units=[400, 300],
        output_dim=env.act_dim
    )
    critic_net = FCActInjected1NetOutputLayer(
        input_dim=env.obs_dim, action_dim=env.act_dim,
        hidden_units=[400, 300], output_dim=1
    )
    return actor_net, critic_net

def _make_actor_critic_net_udacity(env):
    actor_net =  Actor(state_size=env.obs_dim,
                       action_size=env.act_dim,
                       seed=15, fc1_units=256,
                       fc2_units=128)
    critic_net = Critic(
        state_size=env.obs_dim,
        action_size=env.act_dim, seed=16,
        fcs1_units=256, fc2_units=128
    )
    return actor_net, critic_net

def _soft_update(target_net, local_net, tau=1.0):
    for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0-tau) * target_param.data)


class ReacherAgent:
    def __init__(self, reacher_env, dim_tensor_maker, batch_size):
        self.actor_net, self.critic_net = _make_actor_critic_net_udacity(reacher_env)
        self.actor_target, self.critic_target = _make_actor_critic_net_udacity(reacher_env)
        _soft_update(self.actor_target, self.actor_net)
        _soft_update(self.critic_net, self.critic_target)
        self.env = reacher_env
        self.batch_size = batch_size
        #self.memory = ExperienceMemory(BUFFER_SIZE)
        self.dtm = dim_tensor_maker
        self.critic_optimizer = torch.optim.Adam(params=self.critic_net.parameters(), lr=1e-4)
        self.actor_optimizer = torch.optim.Adam(params=self.actor_net.parameters(), lr=1e-4)
        self.noise = [OUNoise(reacher_env.act_dim)] * reacher_env.num_agents
        self.memory = ReplayBuffer(reacher_env.act_dim, BUFFER_SIZE, batch_size, seed=15)

    def act(self, obs):
        action_t = self.actor_net.forward(obs)
        self.dtm.check_agent_out(action_t)
        actions = self.dtm.agent_out_to_np(action_t)
        noise_explore = np.array([n.sample() for n in self.noise])
        actions += noise_explore
        return np.clip(actions, -1, 1)

    def reset(self):
        for n in self.noise:
            n.reset()

    def update(self, states, actions, next_states, rewards, dones):
        # ----- establish critic baseline --------

        for s, a, ns, r, d in zip(*[states, actions, next_states, rewards, dones]):
            #self.memory.add([s, a, ns, r, d])
            self.memory.add(s, a, r, ns, d)

        if (len(self.memory) < self.batch_size):
            return

        tmp = self.memory.sample(self.batch_size)

        b_states, b_action, b_rewards, b_next_states, b_dones = tmp

        #experiences = self.memory.sample(self.batch_size)
        #if experiences is None:
        #    return
        #b_states, b_action, b_next_states, b_rewards, b_dones = experiences

        # ----- establish critic baseline --------
        b_next_states_t = self.dtm.agent_in(obs=b_next_states)
        a_next_t = self.actor_target.forward(b_next_states_t)
        self.dtm.check_agent_out(a_next_t)
        q_next_t = self.critic_target.forward(b_next_states_t, a_next_t)
        q_target_t = self.dtm.rewards_dones_to_tensor(rewards) + DISCOUNT_RATE * self.dtm.rewards_dones_to_tensor(1-dones) * q_next_t
        q_target_t.detach()

        # ----- optimize away critic loss ---------
        b_states_t = self.dtm.agent_in(obs=b_states)
        q_local_t = self.critic_net.forward(b_states_t, self.dtm.agent_out_to_tensor(b_action))
        self.dtm.check_agent_out(q_local_t)
        critic_loss = F.mse_loss(q_local_t, q_target_t)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 1)
        self.critic_optimizer.step()

        # ----- according to the paper, the critic networks' deriv is to be used to update actor network
        actions_t = self.actor_net.forward(b_states_t)
        actor_loss = -self.critic_net.forward(b_states_t, actions_t).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.1)
        self.actor_optimizer.step()

        _soft_update(self.actor_target, self.actor_net, tau=TAU)
        _soft_update(self.critic_target, self.critic_net, tau=TAU)
