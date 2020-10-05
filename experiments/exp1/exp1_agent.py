# -*- coding: utf-8 -*-

from random import random
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim

from models.dqn import DQN
from utils.agent import Agent
from utils.buffer import Experience
from utils.tools import hard_update


class DQNAgent(Agent):
    def __init__(self, obs_size, act_size):
        super(DQNAgent, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dqn = DQN(obs_size, act_size, hidden_size=16).to(self.device)
        self.target_dqn = DQN(obs_size, act_size, hidden_size=16).to(self.device)
        self.criterion = nn.MSELoss()
        hard_update(self.target_dqn, self.dqn)

        # configure optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), 1e-3)

    def get_action(self, state, epsilon):
        self.dqn.eval()
        q_values = [0]
        if np.random.random() < epsilon:
            action = self.random_action()
        else:
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
        expected_state_action_values = next_state_values * 0.99 + reward

        self.dqn.train()
        self.optimizer.zero_grad()
        loss = self.criterion(state_action_values, expected_state_action_values)
        loss.backward()
        nn.utils.clip_grad_norm_(self.dqn.parameters(), 0.1)
        self.optimizer.step()

        return loss