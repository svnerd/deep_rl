from drl.dpn.ddpg.reacher_env import ReacherEnv
from drl.dpn.ddpg.reacher_agent import ReacherAgent
from drl.framework.dim import SingleAgentDimTensorMaker
from drl.util.score_tracker import ScoreTracker
from drl.dpn.ddpg.ddpg_agent import Agent

import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--os", default="linux", help="os")
parser.add_argument("--graph", action="store_true")
parser.add_argument("--good", action="store_true")
args = parser.parse_args()

BATCH_SIZE=128

env = ReacherEnv(os=args.os, display=args.graph)
dim_tensor_maker = SingleAgentDimTensorMaker(
    batch_size=BATCH_SIZE,
    num_env=env.num_agents,
    obs_space=env.obs_dim,
    act_space=env.act_dim
)

agent_good = Agent(state_size=env.obs_dim, action_size=env.act_dim, num_agents=1, random_seed=0)
agent_bad = ReacherAgent(env, dim_tensor_maker, BATCH_SIZE,random_seed=0)

score_tracker = ScoreTracker(good_target=100, window_len=100)
for e in range(200):
    states, r, dones = env.reset()
    scores = np.zeros(env.num_agents)
    agent_good.reset()
    agent_bad.reset()
    while True:
        # all actions between -1 and 1
        actions = agent_bad.act(dim_tensor_maker.agent_in(obs=states))
        print("bad action", actions)
        actions = agent_good.act(states)
        print("good action", actions)
        next_states, rewards, dones = env.step(actions)
        dim_tensor_maker.check_env_out(
            next_states, rewards, dones
        )
        agent_good.step(states, actions, rewards, next_states, dones)
        agent_bad.update(states, actions, next_states, rewards, dones)
        scores += rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            score_tracker.score_tracking(e, np.mean(scores))
            break
env.close()