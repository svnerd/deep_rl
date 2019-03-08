
# with N batch size, M parallel env, single agent per env
# and observation space of P and action space of Q.

# dim input of single env with single agent step(): (1, 1, Q)
# dim input of multi env with single agent step(): (M, 1, Q)
# dim input of multi env with multi agent step(): (M, K, Q)
# dim input of agent's step(): (N, P)

# dim output of single env with multi agent step(): (M, K, P)
# dim output of multi env with multi agent step(): (M, K, Q)
# dim output of agent's step(): (N, Q)

import numpy as np
import torch

class SingleAgentDimTensorChecker:
    def __init__(self, N, M, P, Q):
        self.batch_size = N
        self.num_env = M
        self.obs_space = P
        self.act_space = Q

    def check_env_step_in(self, t):
        assert isinstance(t, np.ndarray)
        assert t.shape[0] == self.num_env
        assert t.shape[1] == self.act_space

    def check_env_step_out(self, t):
        assert isinstance(t, np.ndarray)
        assert t.shape[0] == self.num_env
        assert t.shape[1] == self.obs_space

    def check_agent_network_in(self, t):
        assert isinstance(t, torch.Tensor)
        assert t.shape[0] == self.num_env or t.shape[0] == self.batch_size
        assert t.shape[1] == self.obs_space

    def check_agent_network_out(self, t):
        assert isinstance(t, torch.Tensor)
        assert t.shape[0] == self.num_env or t.shape[0] == self.batch_size
        assert t.shape[1] == self.act_space

class MultiAgentDimTensorChecker:
    def __init__(self, N, M, K, P, Q):
        self.batch_size = N
        self.num_env = M
        self.num_agent = K
        self.obs_space = P
        self.act_space = Q

    def check_env_step_in(self, t):
        assert isinstance(t, np.ndarray)
        assert t.shape[0] == self.num_env
        assert t.shape[1] == self.num_agent
        assert t.shape[2] == self.act_space

    def check_env_step_out_obs(self, t):
        assert isinstance(t, np.ndarray)
        assert t.shape[0] == self.num_env
        assert t.shape[1] == self.num_agent
        assert t.shape[2] == self.obs_space

    def check_env_step_out_obs_full(self, t):
        assert isinstance(t, np.ndarray)
        assert t.shape[0] == self.num_env
        assert t.shape[1] == self.obs_space

    def check_agent_network_in(self, t):
        assert isinstance(t, torch.Tensor)
        assert t.shape[0] == self.num_env or t.shape[0] == self.batch_size
        assert t.shape[1] == self.obs_space

    def check_agent_network_out(self, t):
        assert isinstance(t, torch.Tensor)
        assert t.shape[0] == self.num_env or t.shape[0] == self.batch_size
        assert t.shape[1] == self.act_space