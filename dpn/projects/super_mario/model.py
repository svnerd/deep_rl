from torch import nn
import torch.nn.functional as F
from deep_rl.util.device import DEVICE

def inception_k5():
    conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5)
    return [conv1, pool, conv2]

def inception_k3():
    conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
    return [conv1, pool, conv2]


def forward(conv1, pool, conv2, images):
    t = conv1.forward(images)
    t = pool.forward(t)
    t = conv2.forward(t)
    return t

class SimpleCNN(nn.Module):
    def __init__(self, action_size):
        super(SimpleCNN, self).__init__()
        self.conv1, self.pool, self.conv2 = inception_k5()
        self.fc1 = nn.Linear(3*26*26, 16*16)
        self.fc2 = nn.Linear(16*16, action_size)
        self.to(DEVICE)

    def forward(self, images):
        o5 = forward(self.conv1, self.pool, self.conv2, images)
        t = o5.reshape(-1, 3*26*26)
        t = self.fc1.forward(t)
        t = F.relu(t)
        t = self.fc2.forward(t)
        return t