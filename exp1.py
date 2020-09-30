# -*- coding: utf-8 -*-

from random import random
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, optim
from torch.utils.data import DataLoader

from experiments.exp1.exp1_agent import DQNAgent
from experiments.exp1.exp1_env import Exp1_Env
from utils.buffer import Experience, ReplayBuffer
from utils.dataset import RLDataset


class Exp1(pl.LightningModule):
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

    def populate(self, steps=1000):
        for i in range(steps):
            _, _ = self.play_step(epsilon=1.0)

    def reset(self):
        self.states = self.env.reset()
        self.episode_reward = 0

    def forward(self, x):
        return self.agents[0].net(x)

    def loss_calculation(self, batch):
        loss = 0
        states, actions, rewards, dones, next_states = batch
        for agent_id, agent in enumerate(self.agents):
            state = states[agent_id].float()
            action = actions[agent_id]
            reward = rewards[agent_id]
            done = dones[agent_id]
            next_state = next_states[agent_id].float()

            # normalize states and rewards in range of [0, 1.0]
            state[:, 0::2] /= self.env.world.map.SIZE_X
            state[:, 1::2] /= self.env.world.map.SIZE_Y

            loss += agent.mse_loss(state, action, reward, done, next_state)

        return loss

    def training_step(self, batch, batch_idx):
        #print()
        #print(f'step: {self.global_step}')
        #print(f'agent: {self.env.agents[0].state.p_pos}')
        #print(f'landmark: {self.env.world.landmarks[0].state.p_pos}')
        epsilon = max(0.01, 1.0 - (self.global_step+1)/10000)
        #print(f'epsilon: {epsilon}')
        rewards, dones = self.play_step(epsilon)
        self.episode_reward += np.sum(rewards)
        loss = self.loss_calculation(batch)

        if all(dones):
            self.total_reward += self.episode_reward
            self.reset()

        if self.global_step % 10000 == 0:
            for agent in self.agents:
                agent.target_update()

        log = {'total_reward': torch.tensor(self.total_reward),
               'reward': torch.tensor(rewards).mean(),
               'steps': torch.tensor(self.global_step),
               'loss': loss}

        return {'loss': loss, 'log': log}

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

    logger = TensorBoardLogger('logs/exp1/')
    model = Exp1()

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100000,
        logger=logger,
        checkpoint_callback=None)

    trainer.fit(model)