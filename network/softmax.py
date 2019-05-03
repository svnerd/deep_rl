import torch.nn as nn
import torch
import torch.nn.functional as F


class Softmax(nn.Module):
    def __init__(self, feature_size, action_size):
        super(Softmax, self).__init__()
        self.fc1 = nn.Linear(feature_size, action_size)
        self.fc1.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, features):
        logits = self.fc1.forward(features)
        probs = F.softmax(logits)
        action = probs.multinomial(1).data
        log_prob = F.log_softmax(logits) # log_prob.shape = (N)
        log_action_prob = log_prob.gather(1, action)
        return 0, action, log_action_prob