import sys
import numpy as np
from collections import deque
import os

class ScoreTracker:
    def __init__(self, good_target, record_dir, window_len=100):
        self.scores_window = deque(maxlen=window_len)  # last 100 scores
        self.good_target = good_target
        os.makedirs(record_dir, exist_ok=True)
        self.record_f = open(os.path.join(record_dir, "record.csv"), "w")
        self.record_f.write("episode,mean_score,pit_score\n")
        self.record_f.flush()

    def score_tracking(self, episode, score, report_frequency=100):
        self.scores_window.append(score)
        mean_score = np.mean(self.scores_window)
        print('Episode {}\tAverage Score: {:.3f}\tThis Score: {:.3f}'.format(episode, mean_score, score), end="\n")
        sys.stdout.flush()
        if episode % report_frequency == 0:
            print('\rEpisode {}\tAverage Score: {:.3f}'.format(episode, mean_score))
        if self.is_good():
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(episode, mean_score))
        if episode > 2000:
            exit(0)
        self.record_f.write("{},{},{}\n".format(episode, mean_score, score))
        self.record_f.flush()

    def is_good(self):
        return np.mean(self.scores_window) >= self.good_target