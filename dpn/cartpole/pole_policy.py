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
        self.__clean_buffer()
        self.gamma = discount_rate

    def decide(self):
        experience = self.env.response()
        return self.network.forward(experience.state)

    def update(self, experience, log_prob):
        nsteps = len(self.reward_buffer)
        for i in range(nsteps):
            self.reward_buffer[i] += self.gamma ** (nsteps-i) * experience.reward
        self.reward_buffer.append(experience.reward)
        self.log_prob_buffer.append(log_prob)

        if experience.done:
            policy_loss = []
            for log_prob, r in zip(self.log_prob_buffer, self.reward_buffer):
                policy_loss.append(-log_prob * r)
            self.network.correct(policy_loss)
            self.__clean_buffer()

    def __clean_buffer(self):
        self.reward_buffer = []
        self.log_prob_buffer = []