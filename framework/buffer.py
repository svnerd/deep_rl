import torch
from drl.util.device import tensor_float
from collections import deque, namedtuple
import random
import numpy as np


class StepStorage:
    def __init__(self, nsteps, batch_size):
        self.keys = [
            'a', # action
            'v', # value
            'log_pi_a', # log prob of action a
            'r', # reward
            'c', # continue = 1.0, stop = 0.0
        ]
        self.reset()
        self.nsteps = nsteps
        self.batch_size = batch_size

    def add(self, data):
        for k, v in data.items():
            assert k in self.keys
            v = tensor_float(v)
            if len(v.shape) == 1 and v.shape[0] == self.batch_size:
                v = v.unsqueeze(-1)
            elif len(v.shape) == 2 and v.shape[0] == self.batch_size and v.shape[1] == 1:
                v = v
            else:
                raise Exception(
                    "storage takes shape ({}) or ({}, 1), but instead got {} of {}".format(
                    self.batch_size, self.batch_size, k, str(v.shape)
                    ))
            getattr(self, k).append(v)

    def concat(self, keys=None):
        data = []
        if keys is None:
            keys = self.keys
        for k in keys:
            vs = getattr(self, k)
            len_vs = len(vs)
            if len_vs == 0:
                vs = [None] * self.nsteps
            elif len_vs != len_vs:
                raise Exception("not enough steps taken!")
            data.append(vs)
        # think of the concat result is (batch_size, nsteps)
        # i.e. each step is a feature. 
        return map(lambda x: torch.cat(x, dim=1), data)

    def reset(self):
        for k in self.keys:
            setattr(self, k, [])

class ExperienceMemory:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)
        random.seed(2)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        batch = random.sample(self.memory, k=batch_size)
        return (np.asarray(batch))


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

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


if __name__ == '__main__':
    pool = ExperienceMemory(100)
    pool.add([1, 2, 3])
    pool.add([4, 5, 6])
    pool.add([7, 8, 9])
    for a in pool.sample(batch_size=2):
        print(a)

'''
    s = StepStorage(3, 2)
    data = {
        'a': tensor_float(np.array([1, 2])),
        'v': tensor_float([2, 3]),
        'log_pi_a': tensor_float([-1, -2]), # log prob of action a
        'r': tensor_float([0.2, 0.3]), # reward
        'c': tensor_float([1, 0]), # continue = 1.0, stop = 0.0
    }
    s.add(data)
    s.add(data)
    s.add(data)
    a, v, log_pi_a, r = s.concat(['a', 'v', 'log_pi_a', 'r'])
    print(a)
    print(v)
    print(log_pi_a)
    print(r)

'''
