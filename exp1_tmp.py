# -*- coding: utf-8 -*-

from random import random
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from experiments.exp1.exp1_agent import DQNAgent
from experiments.exp1.exp1_env import Exp1_Env
from utils.buffer import Experience, ReplayBuffer
from utils.dataset import RLDataset


class Exp1:
    def __init__(self):
        super(Exp1, self).__init__()
        self.env = Exp1_Env()
        obs_size = self.env.observation_space
        act_size = self.env.action_space
        # initialize for agents
        self.buffer = ReplayBuffer(10000)
        self.agents = [DQNAgent(obs_size[agent_id], act_size[agent_id]) for agent_id in range(self.env.num_agents)]

        self.total_reward = 0

        self.states = self.env.reset()
        self.populate()
        self.reset()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter()

    def populate(self, steps=1000):
        for i in range(steps):
            _, _ = self.play_step(epsilon=1.0)

    def reset(self):
        self.states = self.env.reset()
        self.episode_reward = 0

    def forward(self, x):
        return self.agents[0].net(x)

    def loss_calculation(self, batch):
        loss = list()
        states, actions, rewards, dones, next_states = batch
        for agent_id, agent in enumerate(self.agents):
            state = states[agent_id].float().to(self.device)
            action = actions[agent_id].to(self.device)
            reward = rewards[agent_id].to(self.device)
            done = dones[agent_id].to(self.device)
            next_state = next_states[agent_id].float().to(self.device)

            # normalize states and rewards in range of [0, 1.0]
            state[:, 0::2] /= self.env.world.map.SIZE_X
            state[:, 1::2] /= self.env.world.map.SIZE_Y

            loss.append(agent.mse_loss(state, action, reward, done, next_state))

        return loss

    def fit(self):
        # hard coding
        max_epochs = 1000
        global_step = 0

        # set dataloader
        dataset = RLDataset(self.buffer, 200)
        dataloader = DataLoader(dataset=dataset, batch_size=16)

        # configure optimizer
        optim_list = list()
        for agent in self.agents:
            optimizer = optim.Adam(agent.net.parameters(), 1e-3)
            optim_list.extend([optimizer])

        # put models on GPU and change to training mode
        for agent in self.agents:
            agent.net.to(self.device)
            agent.target_net.to(self.device)
            agent.net.train()
            agent.target_net.train()

        # training loop
        for epoch in tqdm(range(max_epochs)):
            for batch in dataloader:
                # execute in environment
                epsilon = max(0.01, 1.0 - (global_step+1)/10000)
                rewards, dones = self.play_step(epsilon)
                self.episode_reward += np.sum(rewards)

                if all(dones):
                    self.total_reward += self.episode_reward
                    self.reset()

                if global_step % 100 == 0:
                    for agent in self.agents:
                        agent.target_update()

                # train based on experiments
                loss_list = self.loss_calculation(batch)
                for optimizer, loss in zip(optim_list, loss_list):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # log
                self.writer.add_scalar('total_reward', torch.tensor(self.total_reward), global_step)
                self.writer.add_scalar('reward', torch.tensor(rewards).mean(), global_step)
                self.writer.add_scalar('steps', torch.tensor(global_step), global_step)
                self.writer.add_scalar('loss', loss, global_step)

                global_step += 1
        self.writer.close()

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, 200)
        return DataLoader(dataset=dataset, batch_size=16)

    def configure_optimizers(self):
        optim_list = list()
        for agent in self.agents:
            optimizer = optim.Adam(agent.net.parameters(), 1e-3)
            optim_list.extend([optimizer])

        return optim_list

    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):

        actions = list()
        for agent_id, agent in enumerate(self.agents):
             action = agent.get_action(self.states[agent_id], epsilon)
             actions.append(action)

        new_states, rewards, dones = self.env.step(actions)

        #print(f'action: {actions[0]}')
        #print(f'reward: {rewards[0]}')

        exp = Experience(self.states, actions, rewards, dones, new_states)

        self.buffer.append(exp)

        self.states = new_states

        return rewards, dones


if __name__ == '__main__':
    torch.manual_seed(921)
    np.random.seed(921)

    model = Exp1()
    model.fit()