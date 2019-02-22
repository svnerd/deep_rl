from abc import ABC, abstractmethod
import numpy as np
from multiprocessing import Process, Pipe
from drl.util.device import tensor_float
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import gym

class Envs(ABC):
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

class SerialEnv(Envs):
    def __init__(self, env_fns):
        #self.envs = [fn() for fn in env_fns]
        self.envs = env_fns
        env = self.envs[0]
        Envs.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_env):
            s, r, done, info = self.envs[i].step(self.actions[i])
            if done:
                s = self.envs[i].reset()
            data.append([s, r, done, info])
        s, r, done, info = zip(*data)
        return np.stack(s), np.stack(r), np.stack(done), info

    def reset(self):
        return np.stack([env.reset() for env in self.envs])

    def close(self):
        return

def worker(job_term, scheduler_term, env_fn_wrapper):
    scheduler_term.close()
    #env = env_fn_wrapper.x()
    env = env_fn_wrapper
    while True:
        cmd, action = job_term.recv()
        if cmd == 'step':
            s, r, done, info = env.step(action)
            if done:
                s = env.reset()
            job_term.send((s, r, done, info))
        elif cmd == 'reset':
            s = env.reset()
            job_term.send(s)
        elif cmd == 'close':
            job_term.close()
            break
        elif cmd == 'get_spaces':
            job_term.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class SubprocessEnv(Envs):
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
        Envs.__init__(self, nenvs, obs_space, action_space)

    def reset(self):
        for t in self.scheduler_terms:
            t.send(('reset', None))
        return np.stack([t.recv() for t in self.scheduler_terms])

    def step_async(self, actions):
        for t, action in zip(self.scheduler_terms, actions):
            t.send(('step', action))

    def step_wait(self):
        results = [ t.recv() for t in self.scheduler_terms]
        s, r, done, info = zip(*results)
        return np.stack(s), np.stack(r), np.stack(done), info

    def close(self):
        for t in self.scheduler_terms:
            t.send(('close', None))

        for p in self.ps:
            p.join()

def _get_observation_dim(obs_space):
    if len(obs_space.shape) != 1:
        raise Exception("only support N-d observation space. ")
    return obs_space.shape[0]

def _get_action_dim(action_space):
    if isinstance(action_space, Discrete):
        return action_space.n
    elif isinstance(action_space, Box):
        if len(action_space.shape) != 1:
            raise Exception("only support N-d action space. ")
        return action_space.shape[0]

class EnvDriver:
    def __init__(self, name='CartPole-v0', num_envs=1, single_process=True):
        envs = [gym.make(name) for _ in range(num_envs)]
        [env.seed(2) for env in envs]
        if single_process:
            self.env = SerialEnv(envs)
        else:
            self.env = SubprocessEnv(envs)
        self.name = name
        self.obs_dim = _get_observation_dim(self.env.obs_space)
        self.action_space = self.env.action_space
        self.action_dim = _get_action_dim(self.action_space)
        print(name, "obs_dim =", self.obs_dim, "action_dim =", self.action_dim)
        if isinstance(self.action_space, Box):
            print("action_low =", self.action_space.low, "action_high =", self.action_space.high)

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)

if __name__ == '__main__':
    driver = EnvDriver(num_envs=5, single_process=True)
    print(driver.reset())
    print("====================")
    s, r, done, _ = driver.step([0, 1, 0, 1, 0, 1])
    print(s, tensor_float(r).unsqueeze(-1), done)
