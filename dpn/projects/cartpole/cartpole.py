import numpy as np
from argparse import ArgumentParser

from deep_rl.dpn.projects.cartpole.constant import seed_it, MAX_STEPS, RANDOM_SEED
from deep_rl.dpn.projects.cartpole.cartpole_sep_a2c_agent import CartpoleSepA2CAgent

from deep_rl.util.score_tracker import ScoreTracker
from deep_rl.util.dim import SingleAgentDimTensorMaker
from deep_rl.env.openai.single_agent_env import SingleAgentEnv

parser = ArgumentParser()
parser.add_argument("--record-dir")
args = parser.parse_args()

seed_it()
NUM_ENV = 1
env = SingleAgentEnv(seed=RANDOM_SEED, name='CartPole-v0', num_envs=NUM_ENV)

dim_tensor_maker = SingleAgentDimTensorMaker(
    batch_size=1,
    num_env=NUM_ENV,
    state_size=env.obs_dim,
    act_size=env.act_dim
)
agent = CartpoleSepA2CAgent(env, dim_tensor_maker)
score_tracker = ScoreTracker(good_target=195, window_len=100, record_dir=args.record_dir)


for e in range(2000):
    states = env.reset()
    cnt = 0
    agent.reset()
    score_per_episode = np.zeros(NUM_ENV)
    round_per_env = np.zeros(NUM_ENV)
    dones_cnt = np.zeros(NUM_ENV)
    while True:
        entropy_t, actions_t, log_probs_t, critic_v_t = agent.act(states)
        cnt += 1
        actions = dim_tensor_maker.env_in(actions_t)
        next_states, rewards, dones, _ = env.step(actions)
        ready_to_update = cnt >= 300
        agent.update(
            entropy_t=entropy_t,
            log_probs_t=log_probs_t,
            critic_v_t=critic_v_t,
            rewards=rewards,
            dones=dones,
            step_cnt=cnt,
            ready=ready_to_update
        )
        score_per_episode += rewards[:, 0]
        dones_cnt += dones[:, 0]
        if ready_to_update:
            scores = np.sum(score_per_episode/dones_cnt) / NUM_ENV
            score_tracker.score_tracking(e, score=scores)
            break
        states = next_states
