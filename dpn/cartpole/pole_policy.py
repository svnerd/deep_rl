from drl.dpn.cartpole.pole_network import PoleNetwork
import numpy as np
import drl.util.device as D

# this implements REINFORCE policy
class PolePolicy:
    def __init__(self, env, discount_rate=1.0):
        self.env = env
        state_size, action_size = env.state_action_size()
        print(state_size, action_size)
        self.network = PoleNetwork(state_size, action_size)
        self.reward_buffer = []
        self.log_prob_buffer = []
        self.gamma = discount_rate

    def decide(self):
        experience = self.env.response()
        return self.network.forward(experience.state)

    def update(self, experience, log_prob):
        self.reward_buffer.append(experience.reward)
        self.log_prob_buffer.append(log_prob)

        if experience.done:
            discount_rates = np.array([self.gamma ** i for i in range(len(self.reward_buffer))])
            discounted_reward = np.dot(np.array(self.reward_buffer), discount_rates)
            policy_loss = []
            for log_prob in self.log_prob_buffer:
                policy_loss.append(-log_prob * discounted_reward)
            self.network.correct(policy_loss)