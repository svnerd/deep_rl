from drl.framework.agent import Agent
from drl.framework.network import FCNetOutputLayer
from drl.util.torch_util import soft_update
from drl.util.device import tensor_float
from drl.util.noise import OUNoise
from torch import optim
import torch

TAU = 1e-3
GAMMA = 0.99
LR_CRITIC = 1e-3
LR_ACTOR = 1e-3
ACTOR_HIDDEN_LAYER=[16, 8]
CRITIC_HIDDEN_LAYER=[32, 16]


class GlobalDDPGAgent:
    def __init__(self, env_info, num_agents):
        act_dim = env_info.act_dim

        self.actor = FCNetOutputLayer(
            input_dim=env_info.obs_dim,
            hidden_units=ACTOR_HIDDEN_LAYER,
            output_dim=act_dim
        )
        self.actor_target = FCNetOutputLayer(
            input_dim=env_info.obs_dim,
            hidden_units=ACTOR_HIDDEN_LAYER,
            output_dim=act_dim
        )

        total_act_dim = env_info.obs_dim + act_dim * num_agents
        self.critic = FCNetOutputLayer(
            input_dim=total_act_dim,
            hidden_units=CRITIC_HIDDEN_LAYER,
            output_dim=1
        )
        self.critic_target = FCNetOutputLayer(
            input_dim=total_act_dim,
            hidden_units=CRITIC_HIDDEN_LAYER,
            output_dim=1
        )
        soft_update(self.critic_target, self.critic, tau=1.0)
        soft_update(self.actor_target, self.actor, tau=1.0)
        self.noise = OUNoise(act_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

    def __act(self, actor, obs, noise_scale):
        obs = tensor_float(obs)
        return actor.forward(obs) + noise_scale * self.noise.sample()

    def act(self, obs, noise_scale=0.0):
        self.__act(self.actor, obs, noise_scale)

    def target_act(self, obs, noise_scale=0.0):
        self.__act(self.actor_target, obs, noise_scale)


class MDDPGAgent(Agent):
    def __init__(self, env_info, num_agents):
        super(MDDPGAgent, self).__init__()
        self.ddpg_agent_list = [GlobalDDPGAgent(env_info, num_agents) for i in range(num_agents)]

    def __target_act(self, agent_obs_list, noise_scale=0.0):
        actions = []
        for obs, agent in zip(agent_obs_list, self.ddpg_agent_list):
            actions.append(agent.target_act(obs, noise_scale))
        return actions

    def __get_next_target_q(self, agent, next_obs_list, next_obs_full):
        next_actions = self.__target_act(next_obs_list)
        next_target_critic_input = torch.cat([next_obs_full, next_actions], dim=1)
        with torch.no_grad():
            return agent.critic_target.forward(next_target_critic_input)

    def act(self, agent_obs_list, noise_scale=0.0):
        actions = []
        for obs, agent in zip(agent_obs_list, self.ddpg_agent_list):
            actions.append(agent.act(obs, noise_scale))
        return actions

    def update(self, samples, agent_id):
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(tensor_float, samples)
        this_agent = self.ddpg_agent_list[agent_id]

        # global optimization of critics.
        this_agent.critic_optimizer.zero_grad()
        critic_input = torch.cat([obs_full, action], dim=1)
        v = this_agent.critic.forward(critic_input)
        q_next = self.__get_next_target_q(this_agent, next_obs, next_obs_full)
        y = reward[agent_id] + GAMMA * q_next * (1-done)
        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(v, y)
        critic_loss.backward()
        this_agent.critic_optimizer.step()

        # optimize single agent.
        this_agent.actor_optimizer.zero_grad()
        action_list = []
        for i, agent in enumerate(self.ddpg_agent_list):
            action = agent.actor.forward(obs[i])
            if i == agent_id:
                action.detach()
            action_list.append(action)
        actions = torch.cat(action_list, dim=1)
        q_input = torch.cat([obs_full, actions], dim=1)
        actor_loss = - this_agent.critic.forward(q_input).mean()
        actor_loss.backward()
        this_agent.actor_optimizer.step()

    def update_target(self):
        for agent in self.ddpg_agent_list:
            soft_update(agent.actor_target, agent.actor)
            soft_update(agent.critic_target, agent.critic)
