from .model import CategoricalActorCritic, Actor, Critic
from .constant import MAX_STEPS, DISCOUNT_RATE
from deep_rl.util.device import tensor_float

import torch
from torch import nn

WEIGHT_DECAY = 0.0      # L2 weight decay
LR_CRITIC = 1e-3       # learning rate of the critic

class CartpoleA2CAgent():
    def __init__(self, cartpole_env, dim_maker):
        self.env = cartpole_env
        self.actor_critic  = CategoricalActorCritic(
            state_size=cartpole_env.obs_dim,
            action_size=cartpole_env.act_dim,
            shared_feature_size=8,
            fc_units=[64,32]
        )

        self.dim_maker = dim_maker
        self.optimizer = torch.optim.Adam(params=self.actor_critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.reset()

    def act(self, states):
        states_t = self.dim_maker.agent_in(states)
        actions_t, log_probs_t, critic_v_t = self.actor_critic.forward(states_t)
        return actions_t, log_probs_t, critic_v_t

    def reset(self):
        self.storage = []

    def update(self, states, actions_t, log_probs_t, critic_v_t,
               rewards, next_states, dones,
               step_cnt, ready=False):
        if not ready:
            self.storage.append((states, actions_t, log_probs_t, critic_v_t, rewards, next_states, dones))
            return
        actor_loss_list = [None] * (step_cnt-1)
        critic_loss_list = [None] * (step_cnt-1)
        #next_states_t = self.dim_maker.agent_in(next_states)
        #_, _, critic_v_td_t = self.actor_critic.forward(next_states_t)
        critic_true = critic_v_t
        for i in reversed(range(step_cnt-1)):
            _, _, log_p_t, critic_estimate_t, r, _, ds = self.storage[i]
            r_t = self.dim_maker.rewards_dones_to_tensor(r)
            d_t = self.dim_maker.rewards_dones_to_tensor(1-ds)
            critic_true = r_t + DISCOUNT_RATE * (d_t) * critic_true
            advantage = critic_true.detach() - critic_estimate_t
            actor_loss = -(advantage.detach() * log_p_t)
            self.dim_maker.check_loss(actor_loss)
            actor_loss_list[i] = (actor_loss)
            critic_loss = (advantage.pow(2))
            self.dim_maker.check_loss(critic_loss)
            critic_loss_list[i] = (critic_loss)
        actor_loss = torch.cat(actor_loss_list, dim=1).mean()
        critic_loss = torch.cat(critic_loss_list, dim=1).mean() * 0.5
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()