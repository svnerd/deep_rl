import gym
from drl.util.replay_buffer import Experience
from drl.util.matrix import to_2d


class PoleEnv():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.env.seed(0)
        self.reset()

    def reset(self):
        state = self.env.reset()
        self.response_buffer = Experience(prev_state=None, action=None, reward=None,
                                          state=to_2d(state), done=False)
        return

    def state_action_size(self):
        return (int(self.env.observation_space.shape[0]), int(self.env.action_space.n))

    def response(self, action=None):
        if action is None:
            return self.response_buffer
        state, reward, done, _ = self.env.step(action)
        self.response_buffer = Experience(
            prev_state=self.response_buffer.state,
            action=action, reward=reward, state=to_2d(state), done=done
        )
        return self.response_buffer