from drl.dpn.ddpg.reacher_env import ReacherEnv
import numpy as np

env = ReacherEnv()

scores = np.zeros(env.num_agents)
while True:
    actions = np.random.randn(env.num_agents, env.action_dim) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    next_states, rewards, dones = env.step(actions)
    scores += rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

env.close()