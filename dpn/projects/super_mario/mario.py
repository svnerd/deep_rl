from deep_rl.dpn.projects.super_mario.mario_env import MarioEnv
from deep_rl.dpn.projects.super_mario.q_agent_mario import MarioAgent
from deep_rl.dpn.projects.super_mario.constant import BATCH_SIZE, seed_it
from argparse import ArgumentParser
import time
from deep_rl.util.score_tracker import ScoreTracker
from deep_rl.util.dim import SingleAgentDimTensorMaker
seed_it()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--os", default="linux", help="os")
    parser.add_argument("--record-dir", help="record dir", required=True)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    env = MarioEnv(args.os, args.display)
    dim_maker = SingleAgentDimTensorMaker(
        batch_size=BATCH_SIZE, num_env=1,
        state_size=env.obs_dim, act_size=env.act_dim
    )
    agent = MarioAgent(env, dim_maker, args.record_dir)
    score_tracker = ScoreTracker(good_target=1e6, record_dir=args.record_dir, window_len=100)
    episode = 0
    score_per_episode = 0
    state_t = env.reset()
    while True:
        action = agent.act(state_t)
        next_state_t, reward, done = env.step(action)
        score_per_episode += reward[0]
        if (reward[0] < -10):
            print("died. make it more painful")
            print(score_per_episode)

        agent.update(state_t.squeeze(0), action.reshape(-1), reward, next_state_t.squeeze(0), done, episode)
        if done:
            score_tracker.score_tracking(episode, score_per_episode)
            score_per_episode = 0
            episode += 1
            state = env.reset()
        state_t = next_state_t
        if score_tracker.is_good():
            break
        env.render()
    env.close()
