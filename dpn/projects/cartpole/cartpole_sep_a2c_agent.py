from .model import CategoricalActorCritic, Actor, Critic, DummyNet, FCNet
from .constant import MAX_STEPS, DISCOUNT_RATE

import torch
from torch import nn

WEIGHT_DECAY = 0.0      # L2 weight decay
LR_CRITIC = 3e-4       # learning rate of the critic

class CartpoleSepA2CAgent():
    def __init__(self, cartpole_env, dim_maker):
        self.env = cartpole_env
        '''
        feature_size = 8
        self.shared_net = FCNet(
            input_size=cartpole_env.obs_dim,
            output_size=feature_size,
            hidden_sizes=[8, 4], random_seed=0
        )
        '''
        feature_size = cartpole_env.obs_dim
        self.shared_net = DummyNet(cartpole_env.obs_dim)
        self.actor_net = Actor(
            state_size=feature_size,
            action_size=cartpole_env.act_dim,
            fc1_units=32, fc2_units=16
        )
        self.critic_net  = Critic(
            state_size=feature_size,
            fc1_units=32, fc2_units=16
        )
        self.dim_maker = dim_maker
        self.optimizer = torch.optim.Adam(
            params=list(self.actor_net.parameters()) + list(self.critic_net.parameters()) + list(self.shared_net.parameters()),
            lr=LR_CRITIC, weight_decay=WEIGHT_DECAY
        )
        self.reset()

    def act(self, states):
        states_t = self.dim_maker.agent_in(states)
        feature_t = self.shared_net.forward(states_t)
        entropy_t, actions_t, log_probs_t = self.actor_net.forward(feature_t)
        critic_v_t = self.critic_net.forward(feature_t)
        return entropy_t, actions_t, log_probs_t, critic_v_t

    def reset(self):
        self.storage = []

    def update(self, entropy_t, log_probs_t, critic_v_t,
               rewards, dones,
               step_cnt, ready=False):
        if not ready:
            self.storage.append((log_probs_t, critic_v_t, rewards, dones))
            return

        critic_true = critic_v_t
        returns = [None] * (step_cnt-1)
        values = [None] *(step_cnt -1)
        advantages = [None]*(step_cnt - 1)
        log_probs = [None]*(step_cnt-1)
        entropys = [None] *(step_cnt-1)

        for i in reversed(range(step_cnt-1)):
            log_p_t, critic_estimate_t, r, ds = self.storage[i]
            r_t = self.dim_maker.rewards_dones_to_tensor(r)
            d_t = self.dim_maker.rewards_dones_to_tensor(1-ds)
            critic_true = r_t + DISCOUNT_RATE * (d_t) * critic_true
            advantage = critic_true - critic_estimate_t
            advantages[i] = advantage.detach()
            returns[i] = critic_true.detach()
            values[i] = critic_estimate_t
            log_probs[i] = log_p_t
            entropys[i] = entropy_t

        policy_loss = -(torch.cat(log_probs, dim=1) * torch.cat(advantages, dim=1)).mean()
        value_loss = 0.5 * (torch.cat(returns, dim=1) - torch.cat(values, dim=1)).pow(2).mean()
        entropy_loss = 1e-4 * torch.cat(entropys).mean()
        total_loss = value_loss + policy_loss  - entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm(self.actor_net.parameters(), 0.5)
        nn.utils.clip_grad_norm(self.shared_net.parameters(), 0.5)
        nn.utils.clip_grad_norm(self.critic_net.parameters(), 0.5)
        self.optimizer.step()