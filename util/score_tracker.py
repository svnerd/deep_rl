
import numpy as np
from collections import deque

class ScoreTracker:
    def __init__(self, good_target, window_len=100):
        self.scores_window = deque(maxlen=window_len)  # last 100 scores
        self.good_target = good_target

    def score_tracking(self, episode, score):
        self.scores_window.append(score)
        mean_score = np.mean(self.scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score))
        if self.is_good():
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, mean_score))
        if episode > 2000:
            exit(0)

    def is_good(self):
        return np.mean(self.scores_window) >= self.good_target