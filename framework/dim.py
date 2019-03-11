
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
from drl.util.device import to_np, tensor_float

class SingleAgentDimTensorMaker:
    def __init__(self, batch_size, num_env, obs_space, act_space):
        self.batch_size = batch_size
        self.num_env = num_env
        self.obs_space = obs_space
        self.act_space = act_space

    def env_in(self, act_t):
        act = to_np(act_t)
        assert act.shape[0] == self.num_env
        assert act.shape[1] == self.act_space
        return act

    def rewards_dones_to_tensor(self, rd):
        assert len(rd.shape) == 1
        assert rd.shape[0] == self.num_env or rd.shape[0] == self.batch_size
        return tensor_float(rd)

    def check_env_out(self, obs, rewards, dones):
        assert isinstance(obs, np.ndarray)
        assert isinstance(rewards, np.ndarray)
        assert isinstance(dones, np.ndarray)
        assert obs.shape[0] == self.num_env and rewards.shape[0] == self.num_env and dones.shape[0] == self.num_env
        assert obs.shape[1] == self.obs_space and len(rewards.shape) == 1 and len(dones.shape) == 1

    def agent_in(self, obs):
        obs_t = tensor_float(obs)
        assert obs_t.shape[0] == self.num_env or obs_t.shape[0] == self.batch_size
        assert obs_t.shape[1] == self.obs_space
        return obs_t

    def check_agent_out(self, t):
        assert isinstance(t, torch.Tensor)
        assert t.shape[0] == self.num_env or t.shape[0] == self.batch_size
        assert t.shape[1] == self.act_space or t.shape[1] == 1 # for actor/critic

    def agent_out_to_np(self, t):
        return to_np(t)

    def agent_out_to_tensor(self, o):
        return tensor_float(o)

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