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

    def sample(self):
        if len(self.memory) < self.batch_size:
            return None
        samples = random.sample(self.memory, k=self.batch_size)
        return (np.vstack([s.state for s in samples])), \
               (np.vstack([s.action for s in samples])), \
               (np.vstack([s.reward for s in samples])), \
               (np.vstack([s.next_state for s in samples])), \
               (np.vstack([s.done for s in samples]).astype(np.uint8))

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)