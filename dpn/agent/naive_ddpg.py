from drl.framework.agent import Agent
from drl.framework.buffer import ExperienceMemory
from drl.util.device import to_np, tensor_float, DEVICE
import torch.nn.functional as F
import numpy as np
import torch


from .naive_model import Actor, Critic
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
import torch.optim as optim

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
        state_size=3
        action_size=1
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, 2).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, 2).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, 2).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, 2).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=0)
        print(self.critic_local, self.critic_target, self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.config.noise.sample()
        return np.clip(action, -1, 1)

    def step(self):
        config = self.config
        env_driver = config.env_driver
        actions = self.act(self.states)
        next_states, rs, dones, _ = env_driver.step(actions)
        if dones:
            config.score_tracker.score_tracking(self.episode_cnt, self.episode_reward)
            self.episode_reward = 0
            self.episode_cnt += 1
            next_states = env_driver.reset()
        else:
            self.episode_reward += rs

        self.memory.add([self.states, actions, next_states, rs, dones])
        self.states = next_states

        experiences = self.memory.sample(self.batch_size)
        if experiences is None:
            return
        states, actions, next_states, rewards, dones = experiences
        a_next = self.actor_target(next_states)
        q_next = self.critic_target(next_states, a_next)
        q_target = tensor_float(rewards) + config.discount * tensor_float(1 - dones) * q_next
        # doesn't need derivative
        q_target.detach()
        q = self.critic_local(states, actions)

        critic_loss = F.mse_loss(q, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        a = self.actor_local(states)
        states = tensor_float(states).detach()
        actor_loss = -self.critic_local(states, a).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.critic_local, self.critic_target, config.soft_update_tau)
        self.soft_update(self.actor_local, self.actor_target, config.soft_update_tau)
        return config.score_tracker.is_good()
