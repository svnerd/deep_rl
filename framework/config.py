# config is responsible to define hyper parameters
from drl.util.noise import OUNoise


class Config:
    def __init__(self):
        self.lr = 1e-3
        self.optimization_epochs = 4
        self.soft_update_tau = 0.01
        self.discount = 1.0
        self.env_driver = None
        self.score_tracker = None

class DDPGConfig(Config):
    def __init__(self):
        super(DDPGConfig, self).__init__()
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.target_network = None
        self.noise = None
