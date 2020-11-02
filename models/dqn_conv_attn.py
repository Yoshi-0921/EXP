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

    def __init__(self, obs_size: int, n_actions: int, hidden1: int = 32, hidden2: int = 64, hidden3: int = 100):
        super(DQN_Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden1, kernel_size=2, stride=1)
        self.attn1 = Self_Attention(in_dimm=hidden1)
        self.conv2 = nn.Conv2d(in_channels=hidden1, out_channels=hidden2, kernel_size=2, stride=1)
        self.attn2 = Self_Attention(in_dimm=hidden2)
        self.fc1 = nn.Linear(hidden2*(obs_size**2), hidden3)
        self.fc2 = nn.Linear(hidden3, n_actions)

    def forward(self, x):
        out = self.conv1(x)
        out, attn_hm1 = self.attn1(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.conv2(out)
        out, attn_hm2 = self.attn2(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out, attn_hm1, attn_hm2

class Self_Attention(nn.Module):
    def __init__(self, in_dimm):
        super(Self_Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dimm, out_channels=in_dimm//4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dimm, out_channels=in_dimm//4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dimm, out_channels=in_dimm, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        X = x
        proj_query = self.query_conv(X).view(X.shape[0], -1, X.shape[2]*X.shape[3])
        proj_query = proj_query.permute(0, 2, 1)
        proj_key = self.key_conv(X).view(X.shape[0], -1, X.shape[2]*X.shape[3])

        S = torch.bmm(proj_query, proj_key)

        attention_map_T = self.softmax(S)
        attention_map = attention_map_T.permute(0, 2, 1)

        proj_value = self.value_conv(X).view(X.shape[0], -1, X.shape[2]*X.shape[3])
        o = torch.bmm(proj_value, attention_map.permute(0, 2, 1))

        o = o.view(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
        out = x + self.gamma * o

        return out, attention_map