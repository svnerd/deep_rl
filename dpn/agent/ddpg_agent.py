from drl.framework.agent import Agent
from drl.framework.buffer import ExperienceMemory
from drl.util.device import to_np, tensor_float
import torch.nn.functional as F
import numpy as np
import torch

# note should only have 1 active tensor(requires derivative) in the loss.

class DDPGAgent(Agent):
    def __init__(self, config):
        super(DDPGAgent, self).__init__()
        self.config = config
        self.states = config.env_driver.reset()
        self.memory = ExperienceMemory(int(1e5))
        self.batch_size = 128
        self.episode_reward = 0
        self.episode_cnt = 0

    def __soft_update(self):
        config = self.config
        tau = config.soft_update_tau
        for target_param, local_param in zip(config.target_network.get_actor_params(),
                                             config.network.get_actor_params()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

        for target_param, local_param in zip(config.target_network.get_critic_params(),
                                             config.network.get_critic_params()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def step(self):
        config = self.config
        env_driver = config.env_driver
        config.network.eval()
        with torch.no_grad():
            actions = to_np(config.network.actor(self.states))
        actions += np.array([n.sample() for n in config.noise])
        actions = np.clip(actions, -1, 1)
        next_states, rs, dones, _ = env_driver.step(actions)
        if dones[0]:
            config.score_tracker.score_tracking(self.episode_cnt, self.episode_reward)
            self.episode_reward = 0
            self.episode_cnt += 1
            [n.reset() for n in config.noise]
        else:
            self.episode_reward += rs[0]

        for s, a, ns, r, d in zip(*[self.states, actions, next_states, rs, dones]):
            self.memory.add([s, a, ns, r, d])
        self.states = next_states

        experiences = self.memory.sample(self.batch_size)
        if experiences is None:
            return
        states, actions, next_states, rewards, dones = experiences
        a_next = config.target_network.actor(next_states)
        q_next = config.target_network.critic(next_states, a_next)
        q_target = tensor_float(rewards) + config.discount * tensor_float(1 - dones) * q_next
        # doesn't need derivative
        q_target.detach()
        q = config.network.critic(states, actions)
        config.network.train()

        critic_loss = F.mse_loss(q, q_target)
        config.critic_optimizer.zero_grad()
        critic_loss.backward()
        config.critic_optimizer.step()

        a = config.network.actor(states)
        states = tensor_float(states).detach()
        actor_loss = -config.network.critic(states, a).mean()
        config.actor_optimizer.zero_grad()
        actor_loss.backward()
        config.actor_optimizer.step()
        self.__soft_update()
        return config.score_tracker.is_good()
