from drl.framework.agent import Agent
from drl.framework.network import FCNet


TAU = 1e-3
GAMMA = 0.99


class GlobalDDPGAgent:
    def __init__(self, env_info, num_agents):
        act_dim = env_info.act_dim
        total_act_dim = act_dim * num_agents
        self.actor = FCNet(input_dim=env_info.obs_dim,
                           hidden_units=[300, 400],
                           output_dim=act_dim)


class MDDPGAgent(Agent):
    def __init__(self, env_info, num_agents):
        super(MDDPGAgent, self).__init__()
