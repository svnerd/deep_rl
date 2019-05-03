import numpy as np
from argparse import ArgumentParser

from deep_rl.dpn.projects.cartpole.constant import seed_it, MAX_STEPS, RANDOM_SEED

from deep_rl.util.score_tracker import ScoreTracker
from deep_rl.util.dim import SingleAgentDimTensorMaker
from deep_rl.env.openai.single_agent_env import SingleAgentEnv
from deep_rl.dpn.projects.cartpole.q_agent import QAgent
parser = ArgumentParser()
parser.add_argument("--record-dir")
args = parser.parse_args()

seed_it()
NUM_ENV = 1
env = SingleAgentEnv(seed=RANDOM_SEED, name='CartPole-v0', num_envs=NUM_ENV)

dim_tensor_maker = SingleAgentDimTensorMaker(
    batch_size=128,
    num_env=NUM_ENV,
    state_size=env.obs_dim,
    act_size=env.act_dim
)
agent = QAgent(env, dim_tensor_maker, args.record_dir)
score_tracker = ScoreTracker(good_target=195, window_len=100, record_dir=args.record_dir)


for e in range(2000):
    state = env.reset()
    cnt = 0
    score = 0
    while True:
        state_t = dim_tensor_maker.agent_in(state)
        action = agent.act(state_t)
        cnt += 1
        next_state, reward, done, _ = env.step(action)
        agent.update(
            state[0], action[0], reward[0], next_state[0], done[0], e
        )
        score += reward[0][0]
        if done[0][0]:
            score_tracker.score_tracking(e, score=score)
            break
        state = next_state
