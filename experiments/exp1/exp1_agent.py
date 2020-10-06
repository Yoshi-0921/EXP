# -*- coding: utf-8 -*-

from random import random
from typing import Tuple

import numpy as np
import torch
<<<<<<< HEAD
from torch import nn, optim
=======
from torch import nn
>>>>>>> 3887ac7a7f59978e499b606ebdb04026d3575832

from models.dqn import DQN
from utils.agent import Agent
from utils.buffer import Experience
<<<<<<< HEAD
from utils.tools import hard_update
=======
>>>>>>> 3887ac7a7f59978e499b606ebdb04026d3575832


class DQNAgent(Agent):
    def __init__(self, obs_size, act_size):
        super(DQNAgent, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
<<<<<<< HEAD
        self.dqn = DQN(obs_size, act_size, hidden_size=16).to(self.device)
        self.target_dqn = DQN(obs_size, act_size, hidden_size=16).to(self.device)
        self.criterion = nn.MSELoss()
        hard_update(self.target_dqn, self.dqn)

        # configure optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), 1e-3)

    def get_action(self, state, epsilon):
        self.dqn.eval()
=======
        self.net = DQN(obs_size, act_size, hidden_size=16).to(self.device)
        self.target_net = DQN(obs_size, act_size, hidden_size=16).to(self.device)
        self.criterion = nn.MSELoss()
        self.target_update()

    def get_action(self, state, epsilon):
        self.net.eval()
>>>>>>> 3887ac7a7f59978e499b606ebdb04026d3575832
        q_values = [0]
        if np.random.random() < epsilon:
            action = self.random_action()
        else:
<<<<<<< HEAD
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)

                q_values = self.dqn(state)
                _, action = torch.max(q_values, dim=1)
                action = int(action.item())

        return action, q_values

    def update(self, state, action, reward, done, next_state):
        self.dqn.eval()
        self.target_dqn.eval()
        state_action_values = self.dqn(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.target_dqn(next_state).max(1)[0]
            next_state_values[done] = 0.0
            next_state_values = next_state_values.detach()
        expected_state_action_values = reward + 0.99 * (1 - done) * next_state_values

        self.dqn.train()
        self.optimizer.zero_grad()
        loss = self.criterion(state_action_values, expected_state_action_values)
        loss.backward()
        nn.utils.clip_grad_norm_(self.dqn.parameters(), 0.1)
        self.optimizer.step()
=======
            state = state.unsqueeze(0).to(self.device)

            q_values = self.net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action, q_values

    def target_update(self):

        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(param.data)

    def mse_loss(self, state, action, reward, done, next_state):
        self.net.train()
        self.target_net.eval()
        state_action_values = self.net(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.target_net(next_state).max(1)[0]
            next_state_values[done] = 0.0
            next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * 0.99 + reward

        loss = self.criterion(state_action_values, expected_state_action_values)
>>>>>>> 3887ac7a7f59978e499b606ebdb04026d3575832

        return loss