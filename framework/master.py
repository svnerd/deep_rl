'''
from .env import Env
from .network import Network
from .policy import Policy
from .sampler import Sampler
'''

# this is basically a wrapper to NN's forward and backward.
class Env:
    def observe(self):
        pass


# this is basically a wrapper to NN's forward(approximate) and backward(correct).
class Network:
    def approximate(self):
        pass
    def correct(self, observations):
        pass

class Policy:
    def __init__(self, network):
        self.network = network

    def decide(self):
        return self.network.approximate()

    def update(self, observations):
        self.network.correct(observations)

    def is_good(self):
        return False


class Sampler:
    def __init__(self, env):
        self.env = env
    def generate(self, policy):
        pass


env = Env()
network = Network()
policy = Policy(network)
sampler = Sampler(env)

while True:
    observations = sampler.generate(policy)
    policy.update(observations)
    if policy.is_good():
        break