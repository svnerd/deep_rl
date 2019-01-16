import numpy as np

from collections import deque, namedtuple
import random


Experience = namedtuple("Experience", field_names=["prev_state", "action", "reward", "state", "done"])


class ExperienceMemory:
    def __init__(self, msize=int(1e5)):
        self.memory = deque(maxlen=msize)
        random.seed(78)

    def add(self, prev_state, action, reward, state, done):
        self.memory.append(Experience(prev_state, action, reward, state, done))

    def sample(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None
        samples = random.sample(self.memory, k=batch_size)
        return (np.vstack([s.prev_state for s in samples])), \
               (np.vstack([s.action for s in samples])), \
               (np.vstack([s.reward for s in samples])), \
               (np.vstack([s.state for s in samples])), \
               (np.vstack([s.done for s in samples]).astype(np.uint8))