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


def _forward(conv1, pool, conv2, images):
    t = conv1.forward(images)
    t = pool.forward(t)
    t = conv2.forward(t)
    return t

class SimpleCNN(nn.Module):
    def __init__(self, action_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5)
        self.fc1 = nn.Linear(1*21*21, 8*8)
        self.fc2 = nn.Linear(8*8, action_size)
        self.to(DEVICE)

    def forward(self, images):
        t = self.conv1.forward(images)
        t = self.pool.forward(t)
        t = self.conv2.forward(t)
        t = self.pool2.forward(t)
        t = self.conv3.forward(t)
        t = t.reshape(-1, 1*21*21)
        t = self.fc1.forward(t)
        t = F.relu(t)
        t = self.fc2.forward(t)
        return t