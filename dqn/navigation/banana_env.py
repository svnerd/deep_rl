# this sets up the unity env and
from unityagents import UnityEnvironment


class BananaEnv:
    def __init__(self):
        self.unity_env = UnityEnvironment(file_name="Banana.app")
        self.brain_name = self.unity_env.brain_names[0] # 0 is the default brain
        self.brain = self.unity_env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        # reset the environment
        self.reset_env(train_mode=True)
        self.state_size = len(self.env.vector_observations[0]) # current observation

    def state_action_size(self):
        return (self.state_size, self.action_size)

    def reset_env(self, train_mode=True):
        self.env = self.unity_env.reset(train_mode=train_mode)[self.brain_name]

    def response(self, action=None):
        prev_state = self.env.vector_observations[0]
        if action != None:
            # the env after the action is taken and what the brain sees.
            self.env = self.unity_env.step(action)[self.brain_name]
        reward = self.env.rewards[0]
        state = self.env.vector_observations[0]
        done = self.env.local_done[0]
        return prev_state, reward, state, done

    def close(self):
        self.unity_env.close()

