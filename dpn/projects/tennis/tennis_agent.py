import torch
import numpy as np
import torch.nn.functional as F

from torch import optim
from deep_rl.util.noise import OUNoise
from deep_rl.util.replay_buffer import ExperienceMemory
from deep_rl.network.param import soft_update
from deep_rl.util.device import to_np

from .model import Actor, Critic
from .constant import RANDOM_SEED, BATCH_SIZE

BUFFER_SIZE = int(1e5)  # replay buffer size
DISCOUNT_RATE = 0.99
TAU = 1e-3              # for soft update of target parameters
WEIGHT_DECAY = 0.0      # L2 weight decay
LR_ACTOR = 4e-4         # learning rate of the actor
LR_CRITIC = 4e-4       # learning rate of the critic

FC1 = 256
FC2 = 128


class TennisAgent:
    def __init__(self, env_driver, dim_maker):
        act_dim = env_driver.act_dim
        self.dim_maker = dim_maker

        self.actor_local = Actor(
            state_size=env_driver.obs_dim,
            action_size=act_dim,
            fc1_units=FC1,
            fc2_units=FC2
        )
        self.actor_target = Actor(
            state_size=env_driver.obs_dim,
            action_size=act_dim,
            fc1_units=FC1,
            fc2_units=FC2
        )

        num_agents = env_driver.num_agents
        total_act_dim = env_driver.obs_dim * num_agents + act_dim * num_agents
        self.critic_local = Critic(
            state_size=total_act_dim,
            fc1_units=FC1,
            fc2_units=FC2
        )
        self.critic_target = Critic(
            state_size=total_act_dim,
            fc1_units=FC1,
            fc2_units=FC2
        )
        soft_update(self.critic_target, self.critic_local, tau=1.0)
        soft_update(self.actor_target, self.actor_local, tau=1.0)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

    def act_for_env(self, obs):
        self.actor_local.eval()
        with torch.no_grad():
            act_t = self.actor_local.forward(obs)
            act = self.dim_maker.agent_out_to_np(act_t)
        self.actor_local.train()
        return act


class TennisMultiAgent():
    def __init__(self, env_driver, dim_maker):
        super(TennisMultiAgent, self).__init__()
        self.env_driver = env_driver
        self.noise = OUNoise((env_driver.num_agents, env_driver.act_dim), sigma_decay=0.9995, seed=RANDOM_SEED)
        self.memory = ExperienceMemory(batch_size=BATCH_SIZE, msize=BUFFER_SIZE)
        self.ddpg_agent_list = [TennisAgent(env_driver, dim_maker) for i in range(env_driver.num_agents)]
        self.dim_maker = dim_maker

    def reset(self):
        self.noise.reset()

    def act_for_env(self, agent_obs):
        env_driver = self.env_driver
        actions = np.zeros((env_driver.num_agents, env_driver.act_dim))
        for i, agent in enumerate(self.ddpg_agent_list):
            obs_t = self.dim_maker.agent_in(agent_obs[i, :].reshape(1, -1))

            actions[i, :] = agent.act_for_env(obs_t)
        noise = self.noise.sample()
        actions += noise
        return np.clip(actions, -1, 1)                  # all actions between -1 and 1

    def update(self, states, actions, rewards, next_states, dones):
        self.memory.add(state=states, action=actions, reward=rewards,
                        next_state=next_states, done=dones)
        if len(self.memory) >= BATCH_SIZE:
            for a_idx in range(self.env_driver.num_agents):
                self.__learn(self.memory.sample(self.dim_maker), a_idx)

    def __learn(self, samples, agent_id):
        env_driver = self.env_driver
        b_states_t, b_actions_t, b_rewards_t, b_next_states_t, b_dones_t = samples
        this_agent = self.ddpg_agent_list[agent_id]

        # --------------- the agent's baseline ---------
        # step 1.  get all agents next move from target network.
        a_next_t_list = []
        for i in range(env_driver.num_agents):
            a_next_t_list.append(
                this_agent.actor_target.forward(b_next_states_t[:, i, :])
            )
        a_next_t = torch.cat(a_next_t_list, dim=1)

        q_next_t = this_agent.critic_target.forward(
            torch.cat([b_next_states_t.reshape(BATCH_SIZE, -1), a_next_t], dim=1)
        )
        r = b_rewards_t[:, agent_id].reshape(-1, 1)
        d_1 = 1 - b_dones_t[:, agent_id].reshape(-1, 1)
        q_target_t = r + DISCOUNT_RATE * q_next_t * (d_1)
        assert(q_target_t.shape[0]==BATCH_SIZE and q_target_t.shape[1]==1)

        critic_input = torch.cat([
            b_states_t.reshape(BATCH_SIZE, -1), b_actions_t.reshape(BATCH_SIZE, -1)
        ], dim=1)
        assert(critic_input.shape[0]==BATCH_SIZE)
        v = this_agent.critic_local.forward(critic_input)
        assert(v.shape[0]==BATCH_SIZE and v.shape[1]==1)

        #huber_loss = torch.nn.SmoothL1Loss()
        #critic_loss = huber_loss(v, q_target_t)
        critic_loss = F.mse_loss(v, q_target_t)
        this_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        this_agent.critic_optimizer.step()

        # optimize single agent.
        this_agent.actor_optimizer.zero_grad()
        action_list = []
        for i, agent in enumerate(self.ddpg_agent_list):
            obs = b_states_t[:, i, :]
            assert(obs.shape[0]==BATCH_SIZE and obs.shape[1]==env_driver.obs_dim)
            action = agent.actor_local.forward(obs)
            if i != agent_id:
                action.detach()
            action_list.append(action)
        actions_t = torch.cat(action_list, dim=1)
        assert(actions_t.shape[0]==BATCH_SIZE and actions_t.shape[1] == env_driver.act_dim * env_driver.num_agents)

        critic_input = torch.cat([b_states_t.reshape(BATCH_SIZE, -1), actions_t.reshape(BATCH_SIZE, -1)], dim=1)
        assert(critic_input.shape[0]==BATCH_SIZE)
        actor_loss = -this_agent.critic_local.forward(critic_input).mean()
        actor_loss.backward()
        this_agent.actor_optimizer.step()
        self.__update_target()

    def __update_target(self):
        for agent in self.ddpg_agent_list:
            soft_update(agent.actor_target, agent.actor_local, tau=TAU)
            soft_update(agent.critic_target, agent.critic_local, tau=TAU)
