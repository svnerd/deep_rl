from drl.dpn.cartpole.pole_env import PoleEnv
from drl.dpn.cartpole.pole_policy import PolePolicy

if __name__ == '__main__':
    env = PoleEnv()
    env.reset()

    policy = PolePolicy()
    while True:
        action = policy.decide()
        experience = env.response(action)
        if experience.done:
            break
