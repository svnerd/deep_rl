from argparse import ArgumentParser

import numpy as np

from deep_rl.dpn.projects.tennis.tennis_env import TennisEnv
from deep_rl.util.dim import MultiAgentDimTensorChecker
from deep_rl.util.score_tracker import ScoreTracker

from .tennis_agent import TennisMultiAgent
from .constant import seed_it, BATCH_SIZE

parser = ArgumentParser()
parser.add_argument("--os", default="linux", help="os")
parser.add_argument("--display", action="store_true")
parser.add_argument("--save")
args = parser.parse_args()

seed_it()
env = TennisEnv(os=args.os, display=args.display)
dim_maker = MultiAgentDimTensorChecker(
    batch_size=BATCH_SIZE, num_env=1, num_agent=env.num_agents,
    state_size=env.obs_dim, act_size=env.act_dim
)
agent = TennisMultiAgent(env, dim_maker)
score_tracker = ScoreTracker(good_target=1.0, window_len=100)
save_dir = args.save

for i in range(0, 5000):
    states, _, _ = env.reset()
    agent.reset()
    scores = np.zeros(env.num_agents)
    while True:
        actions = agent.act_for_env(states)
        next_states, rewards, dones = env.step(actions)
        agent.update(
            states=states, actions=actions,
            rewards=rewards, next_states=next_states, dones=dones
        )
        scores += rewards
        states = next_states
        if np.any(dones):
            score_tracker.score_tracking(i, np.max(scores))
            break
    if score_tracker.is_good():
        if save_dir is not None:
            agent.save(save_dir)
env.close()