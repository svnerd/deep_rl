
from .nav_agent import NavAgent
from .constant import BATCH_SIZE

from deep_rl.dqn.projects.navigation.banana_env import BananaEnv
from deep_rl.util.score_tracker import ScoreTracker
from deep_rl.util.dim import SingleAgentDimTensorMaker

if __name__ == '__main__':
    env = BananaEnv()
    dim_maker = SingleAgentDimTensorMaker(
        batch_size=BATCH_SIZE, num_env=1,
        state_size=env.obs_dim, act_size=env.act_dim
    )
    agent = NavAgent(env, dim_maker)

    score_tracker = ScoreTracker(good_target=13, window_len=100)
    episode = 0
    score_per_episode = 0
    state = env.reset()
    while True:
        state_t = dim_maker.agent_in(state)
        action = agent.act(state_t)
        next_state, reward, done = env.step(action)
        score_per_episode += reward[0]
        if done:
            score_tracker.score_tracking(episode, score_per_episode)
            score_per_episode = 0
            episode += 1
            state = env.reset()
        agent.update(state.reshape(-1), action.reshape(-1), reward, next_state.reshape(-1), done)
        if score_tracker.is_good():
            break
    env.close()
