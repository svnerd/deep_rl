from drl.dpn.ddpg.reacher_env import ReacherEnv
from drl.dpn.ddpg.reacher_agent import ReacherAgent
from drl.framework.dim import SingleAgentDimTensorMaker
from drl.util.score_tracker import ScoreTracker
from drl.util.constant import RANDOM_SEED, BATCH_SIZE

import random
import numpy as np
from argparse import ArgumentParser
import torch


parser = ArgumentParser()
parser.add_argument("--os", default="linux", help="os")
parser.add_argument("--display", action="store_true")
parser.add_argument("--good", action="store_true")
args = parser.parse_args()

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

env = ReacherEnv(os=args.os, display=args.display)
dim_tensor_maker = SingleAgentDimTensorMaker(
    batch_size=BATCH_SIZE,
    num_env=env.num_agents,
    state_size=env.obs_dim,
    act_size=env.act_dim
)

agent = ReacherAgent(env, dim_tensor_maker, BATCH_SIZE)

score_tracker = ScoreTracker(good_target=100, window_len=100)
for e in range(200):
    states, r, dones = env.reset()
    scores = np.zeros(env.num_agents)
    agent.reset()
    while True:
        actions = agent.act(dim_tensor_maker.agent_in(states))
        next_states, rewards, dones = env.step(actions)
        dim_tensor_maker.check_env_out(
            next_states, rewards, dones
        )
        agent.update(states, actions, next_states, rewards, dones)
        scores += rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            score_tracker.score_tracking(e, np.mean(scores))
            break
env.close()