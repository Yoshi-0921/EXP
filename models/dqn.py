# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F

class DQN(nn.Module):
    """
    Simple MLP network
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, hidden: int = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out