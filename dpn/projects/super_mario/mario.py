from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    actions=env.action_space.sample()
    print(actions)
    state, reward, done, info = env.step(actions)
    print(state, state.shape, reward, done, info)
    exit(0)
    env.render()

env.close()

