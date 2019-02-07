from drl.framework.agent import Agent
from drl.framework.buffer import StepStorage
from drl.util.device import to_np, tensor_float
import torch

class A2CAgent(Agent):
    def __init__(self, config, nstep):
        super(A2CAgent, self).__init__()
        self.config = config
        self.env_states = config.env_driver.reset()
        self.nstep = nstep
        self.sample_eposide = 0
        self.score_per_episode = 0

    def step(self):
        config = self.config
        storage = StepStorage(self.nstep, batch_size=config.batch_size)
        for _ in range(self.nstep):
            prediction = config.network.forward(self.env_states)
            storage.add(prediction)
            next_states, rewards, dones, _ = config.env_driver.step(to_np(prediction['a']))
            self.score_per_episode += rewards[0]
            if dones[0]:
                config.score_tracker.score_tracking(self.sample_eposide, self.score_per_episode)
                self.score_per_episode = 0
                self.sample_eposide += 1

            storage.add({
                'r': tensor_float(rewards),
                'c': tensor_float(1- dones)
            })
            self.env_states = next_states
        prediction = config.network.forward(self.env_states)
        returns = prediction['v'].detach()
        advantages_list = [None] * self.nstep
        value_loss_list = [None] * self.nstep
        # adv = r + discount* v(next_state) - v (this state)
        for i in reversed(range(self.nstep)):
            returns = storage.r[i] + config.discount * storage.c[i] * returns
            advantages_list[i] = returns.detach() - storage.v[i].detach()
            value_loss_list[i] = returns.detach() - storage.v[i].detach()
            # if gae is used, value_loss will be different from advantage.
        log_pi_a = list(storage.concat(['log_pi_a']))[0]
        advantages_tensor = torch.cat(advantages_list, dim=1)
        policy_loss = -(log_pi_a * advantages_tensor).mean()
        value_loss_tensor = torch.cat(value_loss_list, dim=1)
        #print("vvalue_loss_tensor", value_loss_tensor.shape)
        value_loss = 0.5 * (value_loss_tensor).pow(2).mean()
        total_loss = policy_loss + value_loss
        config.optimizer.zero_grad()
        total_loss.backward()
        config.optimizer.step()
        return config.score_tracker.is_good()