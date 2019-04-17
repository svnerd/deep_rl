
from .nav_agent import NavAgent
from .constant import BATCH_SIZE, seed_it
from argparse import ArgumentParser

from deep_rl.dqn.projects.navigation.banana_env import BananaEnv
from deep_rl.util.score_tracker import ScoreTracker
from deep_rl.util.dim import SingleAgentDimTensorMaker
seed_it()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--os", default="linux", help="os")
    parser.add_argument("--record-dir", help="record dir", required=True)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    env = BananaEnv(os=args.os, display=args.display)
    dim_maker = SingleAgentDimTensorMaker(
        batch_size=BATCH_SIZE, num_env=1,
        state_size=env.obs_dim, act_size=env.act_dim
    )
    agent = NavAgent(env, dim_maker, args.record_dir)

    score_tracker = ScoreTracker(good_target=13, record_dir=args.record_dir, window_len=100)
    episode = 0
    score_per_episode = 0
    state = env.reset()
    while True:
        state_t = dim_maker.agent_in(state)
        action = agent.act(state_t)
        next_state, reward, done = env.step(action)
        score_per_episode += reward[0]
        agent.update(state.reshape(-1), action.reshape(-1), reward, next_state.reshape(-1), done, episode)
        if done:
            score_tracker.score_tracking(episode, score_per_episode)
            score_per_episode = 0
            episode += 1
            state = env.reset()
        state = next_state
        if score_tracker.is_good():
            break
    env.close()