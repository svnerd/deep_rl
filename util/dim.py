
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

from deep_rl.util.device import to_np, tensor_float


class SingleAgentDimTensorMaker:
    def __init__(self, batch_size, num_env, state_size, act_size):
        self.batch_size = batch_size
        self.num_env = num_env
        self.state_size = state_size
        self.act_size = act_size

    def env_in(self, act_t):
        act = to_np(act_t)
        assert act.shape[0] == self.num_env
        assert act.shape[1] == self.act_size or act.shape[1] == 1 # for categorical actions
        return act

    def rewards_dones_to_tensor(self, rd):
        rd_t = tensor_float(rd)
        assert rd_t.shape[0] == self.num_env or rd_t.shape[0] == self.batch_size
        assert rd_t.shape[1] == 1
        return rd_t

    def actions_to_tensor(self, rd):
        rd_t = tensor_float(rd)
        assert rd_t.shape[0] == self.num_env or rd_t.shape[0] == self.batch_size
        assert rd_t.shape[1] == self.act_size or rd_t.shape[1] == 1 # for discrete action
        return rd_t

    def check_env_out(self, obs, rewards, dones):
        assert isinstance(obs, np.ndarray)
        assert isinstance(rewards, np.ndarray)
        assert isinstance(dones, np.ndarray)
        assert obs.shape[0] == self.num_env and rewards.shape[0] == self.num_env and dones.shape[0] == self.num_env
        assert obs.shape[1] == self.state_size and len(rewards.shape) == 1 and len(dones.shape) == 1

    def agent_in(self, obs):
        obs_t = tensor_float(obs)
        assert obs_t.shape[0] == self.num_env or obs_t.shape[0] == self.batch_size
        assert obs_t.shape[1] == self.state_size
        return obs_t

    def check_agent_out(self, t):
        assert isinstance(t, torch.Tensor)
        assert t.shape[0] == self.num_env or t.shape[0] == self.batch_size
        assert t.shape[1] == self.act_size or t.shape[1] == 1 # for actor/critic

    def check_loss(self, loss):
        assert isinstance(loss, torch.Tensor)
        assert loss.shape[0] == self.num_env or loss.shape[0] == self.batch_size or loss.shape[0] == 1
        assert loss.shape[1] == 1

    def agent_out_to_np(self, t):
        return to_np(t)

    def agent_out_to_tensor(self, o):
        return tensor_float(o)

class MultiAgentDimTensorChecker:
    def __init__(self, batch_size, num_env, num_agent, state_size, act_size):
        self.batch_size = batch_size
        self.num_env = num_env
        self.num_agent = num_agent
        self.state_size = state_size
        self.act_size = act_size

    def env_in(self, act_t):
        act = to_np(act_t)
        assert act.shape[0] == self.num_env or act.shape[0] == self.num_agent
        assert act.shape[1] == self.act_size
        return act

    def rewards_dones_to_tensor(self, rd):
        rd_t = tensor_float(rd)
        assert rd_t.shape[0] == self.num_env or rd_t.shape[0] == self.batch_size
        return rd_t

    def actions_to_tensor(self, rd):
        rd_t = tensor_float(rd)
        assert rd_t.shape[0] == self.num_env or rd_t.shape[0] == self.batch_size
        assert rd_t.shape[1] == self.act_size or rd_t.shape[1] == 1 # for discrete action
        return rd_t

    def check_env_out(self, obs, rewards, dones):
        assert isinstance(obs, np.ndarray)
        assert isinstance(rewards, np.ndarray)
        assert isinstance(dones, np.ndarray)
        assert obs.shape[0] == self.num_env and rewards.shape[0] == self.num_env and dones.shape[0] == self.num_env
        assert obs.shape[1] == self.state_size and len(rewards.shape) == 1 and len(dones.shape) == 1

    def agent_in(self, obs):
        obs_t = tensor_float(obs)
        if len(obs_t.shape) == 3:
            assert obs_t.shape[0] == self.num_env or obs_t.shape[0] == self.batch_size
            assert obs_t.shape[1] == self.num_agent
            assert obs_t.shape[2] == self.state_size
        elif len(obs_t.shape) == 2:
            assert obs_t.shape[0] == self.num_env or obs_t.shape[0] == self.batch_size
            assert obs_t.shape[1] == self.state_size
        else:
            assert False
        return obs_t

    def check_agent_out(self, t):
        assert isinstance(t, torch.Tensor)
        assert t.shape[0] == self.num_env or t.shape[0] == self.batch_size
        assert t.shape[1] == self.act_size or t.shape[1] == 1 # for actor/critic

    def agent_out_to_np(self, t):
        return to_np(t)

    def agent_out_to_tensor(self, o):
        return tensor_float(o)