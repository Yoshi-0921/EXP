# -*- coding: utf-8 -*-

import os
import pathlib
import sys
import warnings
from random import random

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../../')
warnings.simplefilter('ignore')

import hydra
import numpy as np
import torch
from experiments.exp4.exp4_agent import DQNAgent
from experiments.exp4.exp4_env import Exp4_Env
from omegaconf import DictConfig
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision.utils import make_grid
from tqdm import tqdm

from utils.buffer import Experience, ReplayBuffer
from utils.dataset import RLDataset
from utils.tools import hard_update


class Exp4:
    def __init__(self, config):
        super(Exp4, self).__init__()
        self.cfg = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.env = Exp4_Env(config)
        obs_size = self.env.observation_space
        act_size = self.env.action_space
        # initialize for agents
        self.buffer = ReplayBuffer(config.capacity, state_conv=True)
        self.agents = [DQNAgent(obs_size[agent_id], act_size[agent_id], config) for agent_id in range(self.env.num_agents)]

        self.total_reward = 0
        self.global_step = 0
        self.episode_count = 0
        self.validation_count = 0
        self.episode_events_left = 0
        self.epsilon = 1.0
        self.heatmap_agents = torch.zeros(self.env.num_agents, 3, self.env.world.map.SIZE_X, self.env.world.map.SIZE_Y)
        self.heatmap_agents[:, 0, ...] = torch.tensor(self.env.world.map.matrix[..., 0])# + (torch.tensor(self.env.world.map.matrix_probs[...] * 3))
        self.heatmap_agents[:, 1, ...] = torch.tensor(self.env.world.map.matrix[..., 0])
        self.heatmap_events = torch.zeros(self.env.num_agents, 2, self.env.world.map.SIZE_X, self.env.world.map.SIZE_Y)
        # 3 (blue)はagentの軌跡

        self.states = self.env.reset()
        self.populate(1000)
        self.reset()
        self.writer = SummaryWriter('exp4')

        # describe network
        print("""
================================================================
DQN Network Summary:""")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        summary(self.agents[0].dqn, (3, 7, 7), batch_size=self.cfg.batch_size, device=device)

    def populate(self, steps: int):
        with tqdm(total=steps) as pbar:
            pbar.set_description('Populating buffer')
            for i in range(steps):
                _, _, _ = self.play_step(epsilon=1.0)
                pbar.update(1)
            pbar.close()

    def reset(self):
        self.states = self.env.reset()
        self.episode_reward = 0
        self.episode_step = 0
        self.episode_events_left = 0
        self.heatmap_agents[:, 0, ...] = torch.tensor(self.env.world.map.matrix[..., 0])
        self.heatmap_agents[:, 1, ...] = torch.tensor(self.env.world.map.matrix[..., 0])
        self.heatmap_agents[:, 2, ...] = torch.zeros(self.env.num_agents, self.env.world.map.SIZE_X, self.env.world.map.SIZE_Y)
        self.heatmap_events = torch.zeros(self.env.num_agents, 2, self.env.world.map.SIZE_X, self.env.world.map.SIZE_Y)

    def loss_and_update(self, batch):
        loss = list()
        states, actions, rewards, dones, next_states = batch
        for agent_id, agent in enumerate(self.agents):
            state = states[agent_id].float().to(self.device)
            action = actions[agent_id].to(self.device)
            reward = rewards[agent_id].to(self.device)
            done = dones[agent_id].to(self.device)
            next_state = next_states[agent_id].float().to(self.device)

            loss.append(agent.update(state, action, reward, done, next_state))

        return loss

    def fit(self):
        # set dataloader
        dataset = RLDataset(self.buffer, self.cfg.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.cfg.batch_size, pin_memory=True)

        # put models on GPU and change to training mode
        for agent in self.agents:
            agent.dqn.to(self.device)
            agent.dqn_target.to(self.device)
            agent.dqn.train()
            agent.dqn_target.eval()

        # training loop
        torch.backends.cudnn.benchmark = True
        with tqdm(total=self.cfg.max_epochs) as pbar:
            for epoch in range(self.cfg.max_epochs):
                # training phase
                for step in range(200):
                    self.global_step += 1
                    self.episode_step += 1
                    total_loss_sum = 0.0

                    # train based on experiments
                    for batch in dataloader:
                        loss_list = self.loss_and_update(batch)

                        for loss in loss_list:
                            total_loss_sum += loss.item()

                    # update target network
                    if self.global_step % 200 == 0:#:self.cfg.synch_epochs == 0:
                        for agent in self.agents:
                            hard_update(agent.dqn_target, agent.dqn)

                    # execute in environment
                    #epsilon = max(0.1, 1.0 - (epoch+1)/self.cfg.decay_epochs)
                    actions, rewards, dones = self.play_step(self.epsilon)
                    self.episode_reward += np.sum(rewards)

                    # log
                    self.writer.add_scalar('training/epsilon', torch.tensor(self.epsilon), self.global_step)
                    self.writer.add_scalar('training/reward', torch.tensor(rewards).mean(), self.global_step)
                    self.writer.add_scalar('training/total_loss', torch.tensor(total_loss_sum), self.global_step)

                    # print on terminal
                    if self.cfg.logs_on_termial and epoch % (self.cfg.max_epochs//10) == 0:
                        print(f"""
    q_values: {self.q[0]}
    actions: {actions}
    rewards: {rewards}

    agent: {self.env.agents[0].state.p_pos}
    landmark: {self.env.world.landmarks[0].state.p_pos}""")

                self.episode_count += 1
                self.epsilon *= 0.999
                self.epsilon = max(0.05, self.epsilon)

                self.writer.add_scalar('episode/episode_reward', torch.tensor(self.episode_reward), self.episode_count)
                self.writer.add_scalar('episode/episode_step', torch.tensor(self.episode_step), self.episode_count)
                self.writer.add_scalar('episode/global_step', torch.tensor(self.global_step), self.episode_count)
                self.writer.add_scalar('env/events_left', torch.tensor(self.env.events_generated-self.env.events_completed), self.episode_count)
                self.writer.add_scalar('env/events_completed', torch.tensor(self.env.events_completed), self.episode_count)
                self.writer.add_scalar('env/agents_collided', torch.tensor(self.env.agents_collided), self.episode_count)
                self.writer.add_scalar('env/walls_collided', torch.tensor(self.env.walls_collided), self.episode_count)
                self.log_heatmaps()
                self.reset()

                # updates pbar
                pbar.set_description(f'[Step {self.global_step}]')
                pbar.set_postfix({'loss': total_loss_sum})
                pbar.update(1)

        self.writer.close()
        pbar.close()

    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):

        actions = list()
        for agent_id, agent in enumerate(self.agents):
            # normalize states [0, map.SIZE] -> [0, 1.0]
            states = torch.tensor(self.states).float()

            action, self.q = agent.get_action(states[agent_id], epsilon)
            actions.append(action)

            # heatmap update
            pos_x, pos_y = self.env.world.map.coord2ind(self.env.agents[agent_id].state.p_pos)
            self.heatmap_agents[agent_id, 2, pos_x, pos_y] += 1

        for landmark in self.env.world.landmarks:
            pos_x, pos_y = self.env.world.map.coord2ind(landmark.state.p_pos)
            # eventは黄色
            self.heatmap_events[..., pos_x, pos_y] += 1
            self.episode_events_left += 1

        new_states, rewards, dones = self.env.step(actions)

        exp = Experience(self.states, actions, rewards, dones, new_states)

        self.buffer.append(exp)

        self.states = new_states

        return actions, rewards, dones

    def log_heatmaps(self):
        for i in range(self.env.num_agents):
            self.heatmap_agents[i, 2, ...] = 0.5 * self.heatmap_agents[i, 2, ...] / torch.max(self.heatmap_agents[i, 2, ...])
            self.heatmap_agents[i, 2, ...] = torch.where(self.heatmap_agents[i, 2, ...]>0, self.heatmap_agents[i, 2, ...]+0.5, self.heatmap_agents[i, 2, ...])

        # 壁の情報を追加
        self.heatmap_agents[:, 2, ...] += torch.tensor(self.env.world.map.matrix[..., 0])
        # eventsの情報を追加
        self.heatmap_events = 0.8 * self.heatmap_events / torch.max(self.heatmap_events)
        self.heatmap_events = torch.where(self.heatmap_events>0, self.heatmap_events+0.2, self.heatmap_events)
        self.heatmap_agents[:, torch.tensor([0, 1]), ...] += self.heatmap_events
        heatmap_agents = F.interpolate(self.heatmap_agents, size=(self.env.world.map.SIZE_X*10, self.env.world.map.SIZE_Y*10))
        heatmap_agents = torch.transpose(heatmap_agents, 2, 3)
        heatmap_agents = make_grid(heatmap_agents, nrow=2)
        self.writer.add_image('episode/heatmap_agents', heatmap_agents, self.episode_count, dataformats='CHW')


@hydra.main(config_path='../../conf/exp4.yaml')
def main(config: DictConfig):
    torch.manual_seed(921)
    np.random.seed(921)

    model = Exp4(config=config)
    model.fit()

if __name__ == '__main__':
    main()
