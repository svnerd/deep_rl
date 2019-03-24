import numpy as np

from collections import deque, namedtuple
import random


Experience = namedtuple("Experience", field_names=[
    "state", "action", "reward", "next_state", "done"
])


class ExperienceMemory:
    def __init__(self, batch_size, msize=int(1e5)):
        self.memory = deque(maxlen=msize)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append(Experience(
            state, action, reward, next_state, done
        ))

    def sample(self, dim_tensor_maker):
        if len(self.memory) < self.batch_size:
            return None
        samples = random.sample(self.memory, k=self.batch_size)
        b_states = np.vstack([s.state for s in samples])
        b_actions = np.vstack([s.action for s in samples])
        b_rewards = np.vstack([s.reward for s in samples])
        b_next_states = np.vstack([s.next_state for s in samples])
        b_dones = np.vstack([s.done for s in samples]).astype(np.uint8)

        b_next_states_t = dim_tensor_maker.agent_in(obs=b_next_states)
        b_rewards_t = dim_tensor_maker.rewards_dones_to_tensor(b_rewards)
        b_dones_t = dim_tensor_maker.rewards_dones_to_tensor(b_dones)
        b_states_t = dim_tensor_maker.agent_in(obs=b_states)
        b_actions_t = dim_tensor_maker.actions_to_tensor(b_actions)
        return b_states_t, b_actions_t, b_rewards_t, b_next_states_t, b_dones_t


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)