import random

import numpy as np
import torch

from drl.dpn.agent.mddpg_agent import MDDPGAgent
from drl.framework.env.mutli_agent_env import MEnvDriver, make_env
from drl.util.score_tracker import ScoreTracker
from drl.framework.buffer import ExperienceMemory

NOISE_DECAY=0.99
NUM_AGENTS=3
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
EPISODE_LENGTH = 80

NUM_PARALLEL_ENV = 5

def seeding(seed=2):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def create_env_driver(num_env):
    return MEnvDriver(
        name='simple_adversary',
        make_env_fn=make_env,
        num_envs= num_env,
        single_process=True
    )


if __name__ == '__main__':
    score_tracker = ScoreTracker(good_target=100, window_len=100)
    replay_buffer = ExperienceMemory(BUFFER_SIZE)
    env = create_env_driver(num_env=NUM_PARALLEL_ENV)

    obs, obs_full = env.reset()
    mddpg_agent = MDDPGAgent(env, NUM_AGENTS)
    noise_amp = 2.0
    episode_cnt = 0
    per_episod_reward = 0
    step_per_episode = 0
    while True:
        actions = mddpg_agent.act(obs, noise_scale=noise_amp)
        noise_amp *= NOISE_DECAY
        actions_array = torch.stack(actions).detach().numpy()
        # rollaxis
        # actions_array.shape
        # Out[19]: (3, 5, 2)
        # np.rollaxis(actions_array,0).shape
        # Out[21]: (3, 5, 2)
        # np.rollaxis(actions_array,1).shape
        # Out[20]: (5, 3, 2)
        # np.rollaxis(actions_array,2).shape
        # Out[22]: (2, 3, 5)

        actions_for_env = np.rollaxis(actions_array,1)
        next_obs, next_obs_full, reward, done, _ = env.step(actions_for_env)
        step_per_episode += 1
        for i in range(NUM_PARALLEL_ENV):
            replay_buffer.add([
                obs[i], obs_full[i], actions_for_env[i],
                reward[i], next_obs[i], next_obs_full[i],
                done.astype(int)[i]
            ])
        need_to_update = False
        for a_idx in range(NUM_AGENTS):
            samples = replay_buffer.sample(BATCH_SIZE)
            if samples is None:
                break
            need_to_update = True
            mddpg_agent.update(samples, a_idx)
        if need_to_update:
            mddpg_agent.update_target()
        obs = next_obs
        obs_full = next_obs_full
        if step_per_episode > EPISODE_LENGTH:
            episode_cnt += 1
            step_per_episode = 0
            per_episod_reward += reward[0][2]
            score_tracker.score_tracking(episode_cnt, per_episod_reward)
            per_episod_reward = 0
            obs, obs_full = env.reset()