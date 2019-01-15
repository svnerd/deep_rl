from drl.dqn.navigation.banana_network import BananaNetwork, long_to_device, float_to_device
from drl.util.replay_buffer import ExperienceMemory
import numpy as np
import random
from collections import deque

EVERY_4_STEPS = 4

class BananaPolicy:
    def __init__(self, env, good_target):
        self.env = env
        state_size, action_size = env.state_action_size()
        print("action, state size", action_size, state_size)
        self.target_network = BananaNetwork(state_size, action_size, need_optimizer=False)
        self.local_network = BananaNetwork(state_size, action_size, need_optimizer=True)
        self.action_size = action_size
        self.memory = ExperienceMemory()
        self.steps = 0
        self.discount_factor = 0.99
        self.scores_window = deque(maxlen=100)  # last 100 scores
        self.good_target = good_target

    def decide(self, eps=0.0):
        _, _, state, _ = self.env.response()
        if random.random() >= eps:
            _, estimate_v = self.local_network.estimate(state)
            action = np.argmax(estimate_v)
        else:
            action = random.choice(np.arange(self.action_size))
        return action

    def update(self, prev_state, action, reward, state, done):
        self.memory.add(prev_state, action, reward, state, done)
        self.steps += 1
        if self.steps % EVERY_4_STEPS == 0:
            experiences = self.memory.sample()
            if experiences == None:
                return
            self.__learn(experiences)

    def score_tracking(self, episode, score):
        self.scores_window.append(score)
        mean_score = np.mean(self.scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score))
        if self.is_good():
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, mean_score))

    def is_good(self):
        return np.mean(self.scores_window) >= self.good_target

    def __learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        #next_state_estimate_tensor = self.target_network.forward(next_states)
        #max_next_state_q = next_state_estimate_tensor.detach().max(1)[0].unsqueeze(1)
        _, next_state_estimate = self.target_network.estimate(next_states)
        max_next_state_q = np.amax(next_state_estimate, axis=1).reshape(-1, 1)
        # if next state is done state, don't count the rewards.
        target_qs = float_to_device(rewards) +\
                    self.discount_factor * max_next_state_q * (1-float_to_device(dones))
        # use local to calculate expected Q
        estimate_qs_tensors = self.local_network.forward(states)
        estimate_qs_by_actions = estimate_qs_tensors.gather(1, long_to_device(actions))
        self.local_network.correct(estimate_qs_by_actions, (target_qs))
        self.target_network.manual_update_param(self.local_network.get_params())

