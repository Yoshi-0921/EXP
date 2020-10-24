# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F

class DQN_Conv(nn.Module):
    """
    Simple MLP network
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden: size of hidden layers
    """

    def __init__(self, n_actions: int, hidden1: int = 32, hidden2: int = 64, hidden3: int = 100):
        super(DQN_Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden1, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=hidden1, out_channels=hidden2, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(hidden2, hidden3)
        self.fc2 = nn.Linear(hidden3, n_actions)

    def forward(self, x):
        out = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        out = F.max_pool2d(self.conv2(out), kernel_size=2, stride=2)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out
