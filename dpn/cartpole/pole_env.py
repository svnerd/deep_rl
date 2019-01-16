import gym
from drl.util.replay_buffer import Experience



class PoleEnv():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.env.seed(0)
        self.response_buffer = Experience(prev_state=None, action=None, reward=None, state=self.reset(), done=False)

    def reset(self):
        return self.env.reset()

    def state_action_size(self):
        return (self.env.observation_space, self.env.action_space)

    def response(self, action=None):
        if action is None:
            return self.response_buffer
        state, reward, done, _ = self.env.step(action)
        self.response_buffer = Experience(
            prev_state=self.response_buffer.state,
            action=action, reward=reward, state=state, done=done
        )
        return self.response_buffers