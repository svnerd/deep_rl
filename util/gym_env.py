from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

class BaseParallelEnvs(ABC):
    def __init__(self, num_envs, obs_space, action_space):
        self.num_env = num_envs
        self.obs_space = obs_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    @abstractmethod
    def close(self):
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        '''
        just render the env like display.
        '''
        pass

def worker(job_term, scheduler_term, env_fn_wrapper):
    scheduler_term.close()
    #env = env_fn_wrapper.x()
    env = env_fn_wrapper
    while True:
        cmd, action = job_term.recv()
        if cmd == 'step':
            result_tuple = env.step(action)
            done = result_tuple[-2]
            if done:
                s = env.reset()
                result_tuple = tuple(list(s) + list(result_tuple[len(s):]))
            job_term.send(result_tuple)
        elif cmd == 'reset':
            job_term.send = env.reset()
        elif cmd == 'close':
            job_term.close()
            break
        elif cmd == 'get_spaces':
            job_term.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class MSerialEnv(BaseParallelEnvs):
    def __init__(self, env_fns):
        self.envs = env_fns
        env = self.envs[0]
        print(env.action_space[0].low)
        BaseParallelEnvs.__init__(self, len(env_fns), env.observation_space[0], env.action_space[0])
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_env):
            obs, obs_full, r, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs, obs_full = self.envs[i].reset()
            data.append([obs, obs_full, r, done, info])
        obs, obs_full, r, done, info = zip(*data)
        return np.stack(obs), np.stack(obs_full), np.stack(r), np.stack(done), info

    def reset(self):
        data = []
        for env in self.envs:
            obs, obs_full = env.reset()
            data.append([obs, obs_full])
        obs, obs_full = zip(*data)
        return np.stack(obs), np.stack(obs_full)

    def close(self):
        return


class MSubprocessEnv(BaseParallelEnvs):
    def __init__(self, env_fns):
        nenvs = len(env_fns)
        self.job_terms, self.scheduler_terms = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(job_term, scheduler_term, env_fn))
                   for job_term, scheduler_term, env_fn in
                   zip(*[self.job_terms, self.scheduler_terms, env_fns])]
        for p in self.ps:
            p.daemon = True
            p.start()
        # pipe has 4 terminals  p_master, p_child --- p_master, p_child.
        # after close, it's like this:
        # p_master --- p_child
        for t in self.job_terms:
            t.close()
        self.scheduler_terms[0].send(('get_spaces', None))
        obs_space, action_space = self.scheduler_terms[0].recv()
        BaseParallelEnvs.__init__(self, nenvs, obs_space, action_space)

    def reset(self):
        for t in self.scheduler_terms:
            t.send(('reset', None))
        results = [t.recv() for t in self.scheduler_terms]
        obs, obs_full = zip(*results)
        return np.stack(obs), np.stack(obs_full)

    def step_async(self, actions):
        for t, action in zip(self.scheduler_terms, actions):
            t.send(('step', action))

    def step_wait(self):
        results = [ t.recv() for t in self.scheduler_terms]
        obs, obs_full, r, done, info = zip(*results)
        return np.stack(obs), np.stack(obs_full), np.stack(r), np.stack(done), info

    def close(self):
        for t in self.scheduler_terms:
            t.send(('close', None))

        for p in self.ps:
            p.join()