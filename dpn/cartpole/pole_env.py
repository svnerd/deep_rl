import gym
from drl.util.replay_buffer import Experience



class PoleEnv():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.env.seed(0)
        self.reset()

    def reset(self):
        state = self.env.reset()
        self.response_buffer = Experience(prev_state=None, action=None, reward=None, state=state, done=False)
        return

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