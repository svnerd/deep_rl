from unityagents import UnityEnvironment
import numpy as np

class ReacherEnv:
    def __init__(self, os='linux', display=False):
        if os == 'linux':
            env = UnityEnvironment(
                file_name='/home/seiya/projects/reinforce/drl/dpn/ddpg/Reacher_Linux_multi/Reacher.x86_64',
                no_graphics=(not display)
            )
        elif os == 'mac':
            env = UnityEnvironment(
                file_name='/Users/chenyuan/project/ipython/drl/dpn/ddpg/Reacher_28',
                no_graphics=(not display)
            )
        else:
            raise Exception("failed to find env.")
        # get the default brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        self.num_agents = len(env_info.agents)
        self.act_dim = brain.vector_action_space_size
        self.obs_dim = states.shape[1]
        self.env = env
        self.brain_name = brain_name

    def __get_obs_r_d(self, env_info):

        return np.array(env_info.vector_observations), \
               np.array(env_info.rewards).reshape(-1), \
               np.array(env_info.local_done).reshape(-1)

    def reset(self):
        return self.__get_obs_r_d(self.env.reset(train_mode=True)[self.brain_name])

    def step(self, actions):
        return self.__get_obs_r_d(self.env.step(actions)[self.brain_name])

    def close(self):
        self.env.close()