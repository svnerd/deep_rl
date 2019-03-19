from drl.framework.network import FCNetOutputLayer, FCActInjected1NetOutputLayer
from drl.dpn.ddpg.replay_buffer import ReplayBuffer

from drl.util.noise import OUNoise
from drl.dpn.ddpg.model import Actor, Critic
import torch.nn.functional as F
import torch, random
import numpy as np


BUFFER_SIZE = int(1e5)  # replay buffer size
DISCOUNT_RATE = 0.99
TAU = 1e-3              # for soft update of target parameters
WEIGHT_DECAY = 0.0      # L2 weight decay
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4       # learning rate of the critic

FC1 = 256
FC2 = 128

def _make_actor_critic_net(env):
    actor_net =  FCNetOutputLayer(
        input_dim=env.obs_dim, hidden_units=[FC1, FC2],
        output_dim=env.act_dim
    )
    critic_net = FCActInjected1NetOutputLayer(
        input_dim=env.obs_dim, action_dim=env.act_dim,
        hidden_units=[FC1, FC2], output_dim=1
    )
    return actor_net, critic_net

def _make_actor_critic_net_udacity(env):
    actor_net =  Actor(state_size=env.obs_dim,
                       action_size=env.act_dim,
                       fc1_units=FC1,
                       fc2_units=FC2)
    critic_net = Critic(
        state_size=env.obs_dim,
        action_size=env.act_dim,
        fcs1_units=FC1, fc2_units=128
    )
    return actor_net, critic_net

def _soft_update(target_net, local_net, tau=1.0):
    for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReacherAgent:
    def __init__(self, reacher_env, dim_tensor_maker, batch_size):
        self.actor_net, self.critic_net = _make_actor_critic_net_udacity(reacher_env)
        self.actor_target, self.critic_target = _make_actor_critic_net_udacity(reacher_env)
        _soft_update(self.actor_target, self.actor_net)
        _soft_update(self.critic_target, self.critic_net)

        self.env = reacher_env
        self.batch_size = batch_size
        #self.memory = ExperienceMemory(BUFFER_SIZE)
        self.dtm = dim_tensor_maker
        self.critic_optimizer = torch.optim.Adam(params=self.critic_net.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.actor_optimizer = torch.optim.Adam(params=self.actor_net.parameters(), lr=LR_ACTOR)
        self.noise = OUNoise((reacher_env.num_agents, reacher_env.act_dim))
        self.memory = ReplayBuffer(reacher_env.act_dim, BUFFER_SIZE, batch_size)

    def act(self, obs):
        self.actor_net.eval()
        with torch.no_grad():
            action_t = self.actor_net.forward(obs)
        self.actor_net.train()
        self.dtm.check_agent_out(action_t)
        actions = self.dtm.agent_out_to_np(action_t)
        actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()


    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (DISCOUNT_RATE * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_net(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_net(states)
        actor_loss = -self.critic_net(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        _soft_update(self.actor_target, self.actor_net, tau=TAU)
        _soft_update(self.critic_target, self.critic_net, tau=TAU)

    def learn_my(self, experiences):

        b_states, b_action, b_rewards, b_next_states, b_dones = experiences
        #experiences = self.memory.sample(self.batch_size)
        #if experiences is None:
        #    return
        #b_states, b_action, b_next_states, b_rewards, b_dones = experiences

        # ----- establish critic baseline --------
        #b_next_states_t = self.dtm.agent_in(obs=b_next_states)
        a_next_t = self.actor_target.forward(b_next_states)
        #self.dtm.check_agent_out(a_next_t)
        q_next_t = self.critic_target.forward(b_next_states, a_next_t)
        b_rewards_t = self.dtm.rewards_dones_to_tensor(b_rewards)
        b_dones_t = self.dtm.rewards_dones_to_tensor(1-b_dones)
        #print(b_rewards_t.shape, b_rewards.shape, b_dones_t.shape, b_dones.shape)
        q_target_t = b_rewards_t + DISCOUNT_RATE * b_dones_t * q_next_t
        #q_target_t.detach()

        # ----- optimize away critic loss ---------
        b_states_t = self.dtm.agent_in(obs=b_states)
        #print(b_states.shape, b_states_t.shape)
        #self.dtm.agent_out_to_tensor
        q_local_t = self.critic_net.forward(b_states_t, b_action)
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
        _soft_update(self.actor_target, self.actor_net, tau=TAU)
        _soft_update(self.critic_target, self.critic_net, tau=TAU)

    def update(self, states, actions, next_states, rewards, dones):
        for agent in range(self.env.num_agents):
            self.memory.add(states[agent,:], actions[agent,:], rewards[agent], next_states[agent,:], dones[agent])

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
