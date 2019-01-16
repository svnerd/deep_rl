from drl.dqn.navigation.banana_network import BananaNetwork
from drl.util.replay_buffer import ExperienceMemory
import drl.util.device as D
import numpy as np
import random


EVERY_4_STEPS = 4

class BananaPolicy:
    def __init__(self, env):
        self.env = env
        state_size, action_size = env.state_action_size()
        print("action, state size", action_size, state_size)
        self.target_network = BananaNetwork(state_size, action_size, need_optimizer=False)
        self.local_network = BananaNetwork(state_size, action_size, need_optimizer=True)
        self.action_size = action_size
        self.memory = ExperienceMemory()
        self.steps = 0
        self.discount_factor = 0.99

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

    def __learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_state_estimate_tensor = self.target_network.forward(next_states)
        max_next_state_q = next_state_estimate_tensor.detach().max(1)[0].unsqueeze(1)
        # if next state is done state, don't count the rewards.
        target_qs = D.float_to_device(rewards) + self.discount_factor * max_next_state_q * (1-D.float_to_device(dones))
        # use local to calculate expected Q
        estimate_qs_tensors = self.local_network.forward(states)
        estimate_qs_by_actions = estimate_qs_tensors.gather(1, D.long_to_device(actions))
        self.local_network.correct(estimate_qs_by_actions, (target_qs))
        self.target_network.manual_update_param(self.local_network.get_params())

