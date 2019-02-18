import torch
import torch.nn as nn
import torch.nn.functional as F
from drl.util.device import tensor_float
from drl.framework.network import PassthroughNet, FCNet
'''
ActorCriticNet has 3 components:
1. the shared DNN
2. Actor DNN (can be passthrough net)
   Critic DNN (can be passthrough net)
3. Actor linear NN output (output size is the action size)
   Critic linear NN output (output size 1 -- the value)
'''

class ActorCriticNet(nn.Module):
    def __init__(self, action_dim, shared_net,
                 actor_net=None, critic_net=None):
        super(ActorCriticNet, self).__init__()
        torch.manual_seed(2)
        if shared_net is None:
            raise Exception("shared net cannot be none")
        if actor_net is None:
            actor_net = PassthroughNet(shared_net.feature_dim)
        if critic_net is None:
            critic_net = PassthroughNet(shared_net.feature_dim)
        self.shared_net = shared_net
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.fc_actor_out = nn.Linear(actor_net.feature_dim, action_dim)
        self.fc_actor_out.weight.data.uniform_(-3e-3, 3e-3)
        self.fc_critic_out = nn.Linear(critic_net.feature_dim, 1)
        self.fc_critic_out.weight.data.uniform_(-3e-3, 3e-3)


class DeterministicActorCriticNet(nn.Module):
    def __init__(self, action_dim, shared_net, actor_net, critic_net):
        super(DeterministicActorCriticNet, self).__init__()
        self.net = ActorCriticNet(action_dim, shared_net, actor_net, critic_net)

    def get_actor_params(self):
        return list(self.net.actor_net.parameters()) + list(self.net.fc_actor_out.parameters())

    def get_critic_params(self):
        return list(self.net.critic_net.parameters()) + list(self.net.fc_critic_out.parameters())

    def forward(self, obs):
        raise Exception("no clear func defined by forward function")

    def __shared_forward(self, obs):
        obs = tensor_float(obs)
        return self.net.shared_net.forward(obs)

    def actor(self, obs):
        x = self.__shared_forward(obs)
        x = self.net.actor_net.forward(x)
        x = self.net.fc_actor_out(x)
        return F.tanh(x)

    def critic(self, obs, action):
        x = self.__shared_forward(obs)
        action = tensor_float(action)
        x = self.net.critic_net.forward(x, action)
        return self.net.fc_critic_out.forward(x)

class CategoricalActorCriticNet(nn.Module):
    def __init__(self, action_dim, shared_net,
                 actor_net=None, critic_net=None):
        super(CategoricalActorCriticNet, self).__init__()
        self.net = ActorCriticNet(action_dim, shared_net, actor_net, critic_net)


    def forward(self, x):
        x = tensor_float(x)
        shared_x = self.net.shared_net.forward(x)
        critic_x = self.net.critic_net.forward(shared_x)
        v = self.net.fc_critic_out.forward(critic_x) # v.shape = (N, 1)

        actor_x = self.net.actor_net.forward(shared_x)
        logits = self.net.fc_actor_out.forward(actor_x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() # action.shape = (N, 1)
        #print("action", action.shape, action)
        log_prob = dist.log_prob(action).unsqueeze(-1) # log_prob.shape = (N, 1)
        #print(log_prob)
        return {
            'a': action,
            'log_pi_a': log_prob,
            'v': v
        }

if __name__ == '__main__':
    net = CategoricalActorCriticNet(4, FCNet(6, [16, 64, 10]))
    for p in net.parameters():
        print(p.shape)
    pred = net.forward(torch.Tensor(5, 6))
    for k, v in pred.items():
        print(k, v.shape)