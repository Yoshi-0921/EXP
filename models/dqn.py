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
        hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        out = F.tanh(self.fc1(x))
        out = F.tanh(self.fc2(out))
        out = F.tanh(self.fc3(out))

        return out