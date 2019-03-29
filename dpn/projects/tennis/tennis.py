from argparse import ArgumentParser

import numpy as np

from deep_rl.dpn.projects.tennis.tennis_env import TennisEnv
from deep_rl.util.dim import MultiAgentDimTensorChecker
from .tennis_agent import TennisMultiAgent
from .constant import seed_it, BATCH_SIZE

parser = ArgumentParser()
parser.add_argument("--os", default="linux", help="os")
parser.add_argument("--display", action="store_true")
args = parser.parse_args()

seed_it()
env = TennisEnv(os=args.os, display=args.display)
dim_maker = MultiAgentDimTensorChecker(
    batch_size=BATCH_SIZE, num_env=1, num_agent=env.num_agents,
    state_size=env.obs_dim, act_size=env.act_dim
)
agent = TennisMultiAgent(env, dim_maker)
score_tracker = ScoreTracker(good_target=30, window_len=100)

for i in range(0, 5000):                                      # play game for 5 episodes
    states, _, _ = env.reset()
    #agent.reset()
    scores = np.zeros(env.num_agents)
    while True:
        actions = agent.act_for_env(states)
        next_states, rewards, dones = env.step(actions)           # send all actions to tne environment
        agent.update(
            states=states, actions=actions,
            rewards=rewards, next_states=next_states, dones=dones
        )
        scores += rewards
        states = next_states                               # roll over states to next time step
        if np.any(dones):
            break
    print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))