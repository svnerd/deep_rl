from drl.dpn.cartpole.pole_network import PoleNetwork
import numpy as np
import drl.util.device as D
import torch

# this implements REINFORCE policy
class PolePPOPolicy:
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

    def update(self, experience, prob):
        nsteps = len(self.state_buffer)
        for i in range(nsteps):
            self.reward_buffer[i] += self.gamma ** (nsteps-i) * experience.reward
        self.state_buffer.append(experience)
        self.reward_buffer.append(experience.reward)
        self.old_prob_buffer.append(prob.detach())
        eps = 0.1
        if experience.done:
            for i in range(3):
                policy_loss = []
                for old_prob, s, r in zip(*[self.old_prob_buffer, self.state_buffer, self.reward_buffer]):
                    _, updated_prob, _ = self.network.forward(s.state, s.action)
                    g = updated_prob / old_prob
                    surrogate = torch.min(g * r, g.clamp(min=1-eps, max=1 + eps) * r)
                    policy_loss.append(-surrogate)
                self.network.correct(policy_loss)
            self.__clean_buffer()
            
    def __clean_buffer(self):
        self.state_buffer = []
        self.reward_buffer = []
        self.old_prob_buffer = []