# -*- coding: utf-8 -*-

from random import random
from typing import Tuple

import numpy as np
import torch
from torch import nn

from models.dqn import DQN
from utils.agent import Agent
from utils.buffer import Experience


class DQNAgent(Agent):
    def __init__(self, obs_size, act_size):
        super(DQNAgent, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = DQN(obs_size, act_size, hidden_size=16).to(self.device)
        self.target_net = DQN(obs_size, act_size, hidden_size=16).to(self.device)
        self.criterion = nn.MSELoss()
        self.target_update()

    def get_action(self, state, epsilon):
        self.net.eval()
        q_values = [0]
        if np.random.random() < epsilon:
            action = self.random_action()
        else:
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)

                q_values = self.net(state)
                _, action = torch.max(q_values, dim=1)
                action = int(action.item())

        return action, q_values

    def target_update(self):

        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(param.data)

    def mse_loss(self, state, action, reward, done, next_state):
        self.net.eval()
        self.target_net.eval()
        state_action_values = self.net(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.target_net(next_state).max(1)[0]
            next_state_values[done] = 0.0
            next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * 0.99 + reward

        self.net.train()
        loss = self.criterion(state_action_values, expected_state_action_values)

        return loss