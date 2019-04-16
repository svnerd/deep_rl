
from dqn.navigation.view import RobotView
from dqn.navigation.model import NavigationBrainAgent

from collections import deque
import numpy as np
import torch

class RobotController:
    def __init__(self):
        self.view = RobotView()
        self.agent = NavigationBrainAgent(self.view.state_size, self.view.action_size)
        self.eps = SoftEpsilonDecay(1.0, 1e-3, 0.995)
        self.dropout = SoftEpsilonDecay(0.5, 0.1, 0.995)

    def train(self, train_target=13, episode_cnt=2000):
        eps = self.eps.eps
        dropout = self.dropout.eps
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        for i in range(episode_cnt):
            self.view.reset_env(train_model=True)
            s, _, done = self.view.get_state_reward_done()
            score = 0
            while True:
                action = self.agent.act(s, eps)
                self.view.step(action)
                next_s, r, done = self.view.get_state_reward_done()
                self.agent.observe(s, action, r, next_s, done, dropout)
                score += r
                if done:
                    break
            eps = self.eps.decay()
            dropout = self.dropout.decay()
            scores_window.append(score)
            scores.append(score)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)), end="")
            if i % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
            if np.mean(scores_window)>=train_target:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i-100, np.mean(scores_window)))
                torch.save(self.agent.local_model.state_dict(), 'checkpoint.pth')
                break
        return scores

    def show(self):
        self.agent.local_model.load_state_dict(torch.load('checkpoint.pth'))
        for i in range(5):
            state = self.view.reset_env(train_model=False)
            for j in range(200):
                action = self.agent.act(state)
                _, _, done = self.view.step(action)
                if done:
                    break

    def close(self):
        self.view.unity_env.close()


class SoftEpsilonDecay:
    def __init__(self, start_epsilon, end_epsilon, decay_coeff):
        self.eps = start_epsilon
        self.end = end_epsilon
        self.decay_coeff = decay_coeff

    def decay(self):
        self.eps *= self.decay_coeff
        return max(self.end, self.eps)

if __name__ == '__main__':
    controller = RobotController()
    controller.train()