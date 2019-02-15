import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from drl.dpn.agent.naive_ddpg import DDPGAgent
from drl.framework.config import DDPGConfig
from drl.util.score_tracker import ScoreTracker
from drl.util.noise import OUNoise

env = gym.make('Pendulum-v0')
env.seed(2)
config = DDPGConfig()
config.env_driver = env

config.soft_update_tau = 1e-3
config.discount = 0.99
config.score_tracker = None

score_tracker = ScoreTracker(good_target=100, window_len=100)
config.noise = OUNoise(1, seed=2)


from drl.dpn.pendulum.udacity.ddpg_agent import Agent
#agent = Agent(state_size=3, action_size=1, random_seed=2)
agent = DDPGAgent(config)

def ddpg(n_episodes=1000, max_t=300, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        #agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores

scores = ddpg()
