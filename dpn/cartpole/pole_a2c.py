
from drl.framework.config import Config
from drl.framework.env import EnvDriver
from drl.dpn.network.actor_critic import CategoricalActorCriticNet
from drl.framework.network import FCNet
from drl.dpn.agent.a2c_agent import A2CAgent
from drl.util.score_tracker import ScoreTracker
import torch

if __name__ == '__main__':
    config = Config()
    config.env_driver = EnvDriver(
        name='CartPole-v0',
        num_envs=64,
        single_process=True
    )
    action_dim = config.env_driver.action_dim
    config.network = CategoricalActorCriticNet(
        action_dim=action_dim,
        shared_net=FCNet(input_dim=config.env_driver.obs_dim, hidden_units=[64,32]),
        actor_net=None, critic_net=None
    )
    config.optimizer = torch.optim.Adam(params=config.network.parameters(), lr=3e-4)
    config.score_tracker = ScoreTracker(good_target=193, window_len=100)
    agent = A2CAgent(config, nstep=20)
    while True:
        if agent.step():
            break