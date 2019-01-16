
from drl.dqn.navigation.banana_env import BananaEnv
from drl.dqn.navigation.banana_policy import BananaPolicy
from drl.util.eps_decay import SoftEpsilonDecay
from drl.util.score_tracker import ScoreTracker

if __name__ == '__main__':
    env = BananaEnv()
    policy = BananaPolicy(env)
    score_tracker = ScoreTracker(good_target=13, window_len=100)
    eps_handler = SoftEpsilonDecay(1.0, 1e-3, 0.995)
    eps = eps_handler.eps
    episode = 0
    score_per_episode = 0
    while True:
        action = policy.decide(eps)
        prev_state, reward, state, done = env.response(action)
        score_per_episode += reward
        if done:
            eps = eps_handler.decay()
            score_tracker.score_tracking(episode, score_per_episode)
            score_per_episode = 0
            episode += 1
            env.reset_env()
        policy.update(prev_state, action, reward, state, done)
        if score_tracker.is_good():
            break
    env.close()
