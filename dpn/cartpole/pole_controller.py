from drl.dpn.cartpole.pole_env import PoleEnv
from drl.dpn.cartpole.pole_policy import PolePolicy
from drl.dpn.cartpole.pole_ppo_policy  import PolePPOPolicy
from drl.util.score_tracker import ScoreTracker

if __name__ == '__main__':
    env = PoleEnv()
    env.reset()
    score_tracker = ScoreTracker(good_target=193, window_len=100)
    policy = PolePolicy(env)
    #policy = PolePPOPolicy(env)
    episode = 0
    score_per_episode = 0
    while True:
        action, prob, log_prob = policy.decide()
        experience = env.response(action)
        score_per_episode += experience.reward
        if experience.done:
            score_tracker.score_tracking(episode, score_per_episode)
            score_per_episode = 0
            episode += 1
            env.reset()
        policy.update(experience, log_prob)

        if score_tracker.is_good():
            break