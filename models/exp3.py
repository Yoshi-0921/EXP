# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_size, n_actions, hidden1=400, hidden2=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out))
        return out

class Critic(nn.Module):
    def __init__(self, obs_size, n_actions, num_agents, hidden1=400, hidden2=300):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_size*num_agents, hidden1)
        self.fc2 = nn.Linear(hidden1 + n_actions*num_agents, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x, a):
        out = F.relu(self.fc1(x))
        xa_cat = torch.cat([out, a], 1)
        out = F.relu(self.fc2(xa_cat))
        out = self.fc3(out)
        return out