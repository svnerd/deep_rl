# this sets up the unity env and
from unityagents import UnityEnvironment


class RobotView:
    def __init__(self):
        self.unity_env = UnityEnvironment(file_name="Banana.app")
        self.brain_name = self.unity_env.brain_names[0] # 0 is the default brain
        self.brain = self.unity_env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size

        # reset the environment
        self.reset_env()
        self.state_size = len(self.env.vector_observations[0]) # current observation

    def reset_env(self, train_model=True):
        self.env = self.unity_env.reset(train_mode=train_model)[self.brain_name]

    def step(self, action):
        # the env after the action is taken and what the brain sees.
        self.env = self.unity_env.step(action)[self.brain_name]

    def get_state_reward_done(self):
        next_state = self.env.vector_observations[0]
        reward = self.env.rewards[0]
        done = self.env.local_done[0]
        return next_state, reward, done
