from .model import CategoricalActorCritic, Actor, Critic
from .constant import MAX_STEPS, DISCOUNT_RATE
from deep_rl.util.device import tensor_float

import torch
from torch import nn

WEIGHT_DECAY = 0.0      # L2 weight decay
LR_CRITIC = 5e-4       # learning rate of the critic

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
        entropy, actions_t, log_probs_t, critic_v_t = self.actor_critic.forward(states_t)
        return entropy, actions_t, log_probs_t, critic_v_t

    def reset(self):
        self.storage = []

    def update(self, entropy, log_probs_t, critic_v_t,
               rewards, dones,
               step_cnt, ready=False):
        if not ready:
            self.storage.append((entropy, log_probs_t, critic_v_t, rewards, dones))
            return
        actor_loss_list = [None] * (step_cnt-1)
        critic_loss_list = [None] * (step_cnt-1)
        #next_states_t = self.dim_maker.agent_in(next_states)
        #_, _, critic_v_td_t = self.actor_critic.forward(next_states_t)
        critic_true = critic_v_t
        returns = [None] * (step_cnt-1)
        values = [None] *(step_cnt -1)
        advantages = [None]*(step_cnt - 1)
        log_probs = [None]*(step_cnt-1)
        entropys = [None] *(step_cnt-1)
        for i in reversed(range(step_cnt-1)):
            entropy_t, log_p_t, critic_estimate_t, r, ds = self.storage[i]
            r_t = self.dim_maker.rewards_dones_to_tensor(r)
            d_t = self.dim_maker.rewards_dones_to_tensor(1-ds)
            critic_true = r_t + DISCOUNT_RATE * (d_t) * critic_true
            advantage = critic_true - critic_estimate_t
            advantages[i] = advantage.detach()
            returns[i] = critic_true.detach()
            values[i] = critic_estimate_t
            log_probs[i] = log_p_t
            entropys[i] = entropy_t
            #actor_loss = -(advantage.detach() * log_p_t)
            #self.dim_maker.check_loss(actor_loss)
            #actor_loss_list[i] = (actor_loss)
            #returns = critic_true.detach() - critic_estimate_t
            #critic_loss = (returns.pow(2))
            #self.dim_maker.check_loss(critic_loss)
            #critic_loss_list[i] = (critic_loss)
        #actor_loss = torch.cat(actor_loss_list, dim=1).mean()
        #critic_loss = torch.cat(critic_loss_list, dim=1).mean() * 0.5

        policy_loss = -(torch.cat(log_probs, dim=1) * torch.cat(advantages, dim=1)).mean()
        value_loss = 0.5 * (torch.cat(returns, dim=1) - torch.cat(values, dim=1)).pow(2).mean()
        entropy_loss = 1e-4 * torch.cat(entropys, dim=1).mean()
        total_loss = policy_loss + value_loss - entropy_loss
        #self.actor_optimizer.zero_grad()
        #policy_loss.backward()
        #self.actor_optimizer.step()

        #self.critic_optimizer.zero_grad()
        #value_loss.backward()
        #self.critic_optimizer.step()
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm(self.actor_critic.parameters(), 0.5)

        self.optimizer.step()
        print(policy_loss, value_loss)