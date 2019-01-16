from drl.dpn.cartpole.pole_network import PoleNetwork
import numpy as np

# this implements REINFORCE policy
class PolePolicy:
    def __init__(self, env, discount_rate=1.0):
        self.env = env
        state_size, action_size = env.state_action_size()
        self.network = PoleNetwork(state_size, action_size)
        self.reward_buffer = []
        self.gamma = discount_rate

    def decide(self):
        experience = self.env.response()
        return self.network.forward(experience.state)

    def update(self, experience):
        if experience.done:
            discount_rates = np.array([self.gamma ** i for i in range(len(self.reward_buffer))])
            discounted_reward = np.dot(np.array(self.reward_buffer), discount_rates)

