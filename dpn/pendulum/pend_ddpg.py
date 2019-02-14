
from drl.framework.config import DDPGConfig
from drl.framework.env import EnvDriver
from drl.dpn.network.actor_critic import DeterministicActorCriticNet
from drl.framework.network import FCNet, PassthroughNet, FCActInjected1Net
from drl.dpn.agent.naive_ddpg import DDPGAgent
from drl.util.score_tracker import ScoreTracker
from drl.util.noise import OUNoise
from drl.util.device import DEVICE
import torch

def make_ddpg_net(config):
    action_dim = config.env_driver.action_dim
    obs_dim = config.env_driver.obs_dim
    return DeterministicActorCriticNet(
        action_dim=action_dim,
        shared_net=PassthroughNet(dim=obs_dim),
        actor_net=FCNet(input_dim=obs_dim, hidden_units=[400, 300]),
        critic_net=FCActInjected1Net(input_dim=obs_dim, action_dim=action_dim, hidden_units=[400, 300])
    ).to(DEVICE)

if __name__ == '__main__':
    config = DDPGConfig()
    num_env = 1
    config.env_driver = EnvDriver(
        name='Pendulum-v0',
        num_envs= num_env,
        single_process=True
    )
    config.soft_update_tau = 1e-3
    config.discount = 0.99
    action_dim = config.env_driver.action_dim
    config.noise = [OUNoise(action_dim, seed=2)] * num_env
    obs_dim = config.env_driver.obs_dim
    config.network = make_ddpg_net(config)
    config.optimizer = None
    config.actor_optimizer = torch.optim.Adam(params=config.network.get_actor_params(), lr=1e-4)
    config.critic_optimizer = torch.optim.Adam(params=config.network.get_critic_params(), lr=1e-3)
    config.target_network = make_ddpg_net(config)
    config.target_network.load_state_dict(config.network.state_dict())
    config.score_tracker = ScoreTracker(good_target=100, window_len=100)
    agent = DDPGAgent(config)
    while True:
        if agent.step():
            break
