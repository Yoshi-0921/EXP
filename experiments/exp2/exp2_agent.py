# -*- coding: utf-8 -*-

from random import random
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

from models.ddpg import Actor, Critic
from utils.agent import Agent
from utils.buffer import Experience
from utils.tools import soft_update, hard_update


class DDPGAgent(Agent):
    def __init__(self, obs_size, act_size, config):
        super(DDPGAgent, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # set neural networks
        self.actor = Actor(obs_size, act_size, hidden1=config.hidden1, hidden2=config.hidden2).to(self.device)
        self.actor_target = Actor(obs_size, act_size, hidden1=config.hidden1, hidden2=config.hidden2).to(self.device)
        self.critic = Critic(obs_size, act_size, hidden1=config.hidden1, hidden2=config.hidden2).to(self.device)
        self.critic_target = Critic(obs_size, act_size, hidden1=config.hidden1, hidden2=config.hidden2).to(self.device)
        self.criterion = nn.MSELoss()

        # configure optimizer
        self.actor_optimizer  = optim.Adam(params=self.actor.parameters(),
                                           lr=config.learning_rate,
                                           betas=config.betas,
                                           eps=config.eps)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(),
                                           lr=config.learning_rate,
                                           betas=config.betas,
                                           eps=config.eps)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.gamma = config.gamma
        self.tau = config.tau
        self.obs_size = obs_size
        self.act_size = act_size

    def get_action(self, state, epsilon):
        self.actor.eval()
        if np.random.random() < epsilon:
            action = self.random_action()
            logit = self.onehot_from_action(action)
            logit = logit.detach().cpu().numpy()
        else:
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)

                logit = self.actor(state)
                logit = F.softmax(logit)
                _, action = torch.max(logit, dim=1)
                logit = logit[0].detach().cpu().numpy()
                action = int(action.item())

        return action, logit

    def onehot_from_logits(self, logits, eps=0.0):
        # get best (according to current policy) actions in one-hot form
        argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
        if eps == 0.0:
            return argmax_acs

        # get random actions in one-hot form
        rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
            range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
        # chooses between best and random actions using epsilon greedy
        return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in enumerate(torch.rand(logits.shape[0]))])

    def onehot_from_action(self, action):
        action = Variable(torch.eye(self.act_size)[action], requires_grad=False).to(self.device)
        return action

    def update(self, state, logit, reward, done, next_state):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

        with torch.no_grad():
            # Get the actions and the state values to compute the targets
            next_action_batch = self.actor_target(next_state)
            next_action_batch = F.softmax(next_action_batch)
            next_state_action_values = self.critic_target(next_state, next_action_batch.detach())

            # Compute the target
            reward = reward.unsqueeze(-1)
            done = done.unsqueeze(-1)
            expected_values = reward + self.gamma * (1 - done) * next_state_action_values

        self.critic.train()
        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state, logit)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        self.critic.eval()
        self.actor.train()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state, F.softmax(self.actor(state)))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()