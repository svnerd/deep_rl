from drl.framework.agent import Agent
from drl.framework.network import FCNetOutputLayer
from drl.util.torch_util import soft_update


TAU = 1e-3
GAMMA = 0.99
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


class MDDPGAgent(Agent):
    def __init__(self, env_info, num_agents):
        super(MDDPGAgent, self).__init__()
        self.ddpg_agent_list = [GlobalDDPGAgent(env_info, num_agents) for i in range(num_agents)]

    def
