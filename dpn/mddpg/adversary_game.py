import numpy as np
import torch
from drl.framework.env import EnvDriver
from drl.util.score_tracker import ScoreTracker
from drl.dpn.agent.mddpg_agent import MDDPGAgent
import random
import multiagent as mt
import make_env

NOISE_DECAY=0.99
NUM_AGENTS=3

def seeding(seed=2):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def create_env_driver(num_env):
    return EnvDriver(
        name='Pendulum-v0',
        num_envs= num_env,
        single_process=True
    )


if __name__ == '__main__':
    score_tracker = ScoreTracker(good_target=100, window_len=100)
    env = create_env_driver(1)
    obs, obs_full = env.reset()
    mddpg_agent = MDDPGAgent(env, NUM_AGENTS)
    noise_amp = 2.0
    while True:
        actions = mddpg_agent.act(obs, noise_scale=noise_amp)
        noise_amp *= NOISE_DECAY
        next_obs, next_obs_full, reward, done, _ = env.step(actions)
        