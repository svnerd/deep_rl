from drl.framework.agent import Agent
from drl.framework.buffer import ExperienceMemory
from drl.util.device import to_np, tensor_float, DEVICE
import torch.nn.functional as F
import numpy as np
import torch, random
from collections import namedtuple, deque

# note should only have 1 active tensor(requires derivative) in the loss.
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size

class DDPGAgent(Agent):
    def __init__(self, config):
        super(DDPGAgent, self).__init__()
        self.config = config
        self.states = config.env_driver.reset()
        self.memory = ReplayBuffer(1, BUFFER_SIZE, BATCH_SIZE, 2)#ExperienceMemory(BUFFER_SIZE)#
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
        config.network.train()
        actions += np.array([n.sample() for n in config.noise])
        next_states, rs, dones, _ = env_driver.step(actions)
        if dones[0]:
            config.score_tracker.score_tracking(self.episode_cnt, self.episode_reward)
            self.episode_reward = 0
            self.episode_cnt += 1
            [n.reset() for n in config.noise]
        else:
            self.episode_reward += rs[0]

        #for s, a, ns, r, d in zip(*[self.states, actions, next_states, rs, dones]):
        #    self.memory.add([s, a, r, ns, d])
        self.memory.add(self.states, actions, rs, next_states, dones)

        self.states = next_states
        #experiences = self.memory.sample(BATCH_SIZE)
        #if experiences is None:
        #    return
        if len(self.memory) <= BATCH_SIZE:
            return
        experiences = self.memory.sample()

        states, actions, rewards, next_states, dones = experiences
        a_next = config.target_network.actor(next_states)
        q_next = config.target_network.critic(next_states, a_next)
        q_target = tensor_float(rewards) + config.discount * tensor_float(1 - dones) * q_next
        # doesn't need derivative
        q_target.detach()
        q = config.network.critic(states, actions)

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


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)