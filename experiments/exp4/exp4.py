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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
        self.epsilon = config.epsilon_initial
        self.heatmap_agents = torch.zeros(self.env.num_agents, 3, self.env.world.map.SIZE_X, self.env.world.map.SIZE_Y)
        self.heatmap_agents[:, 0, ...] = torch.tensor(self.env.world.map.matrix[..., 0])# + (torch.tensor(self.env.world.map.matrix_probs[...] * 3))
        self.heatmap_agents[:, 1, ...] = torch.tensor(self.env.world.map.matrix[..., 0])
        # 3 (blue)はagentの軌跡

        self.states = self.env.reset()
        torch.backends.cudnn.benchmark = True
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
        self.heatmap_agents[:, 0, ...] = torch.tensor(self.env.world.map.matrix[..., 0])
        self.heatmap_agents[:, 1, ...] = torch.tensor(self.env.world.map.matrix[..., 0])
        self.heatmap_agents[:, 2, ...] = torch.zeros(self.env.num_agents, self.env.world.map.SIZE_X, self.env.world.map.SIZE_Y)

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
        loss = torch.from_numpy(np.array(loss, dtype=np.float))
        return loss

    def fit(self):
        self.writer = SummaryWriter('exp4')
        # set dataloader
        dataset = RLDataset(self.buffer, self.cfg.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.cfg.batch_size, pin_memory=True)

        # put models on GPU and change to training mode
        for agent in self.agents:
            agent.dqn.to(self.device)
            agent.dqn_target.to(self.device)
            agent.dqn.train()
            agent.dqn_target.eval()

        # populate buffer
        self.populate(self.cfg.populate_steps)
        self.reset()

        # training loop
        with tqdm(total=self.cfg.max_epochs) as pbar:
            for epoch in range(self.cfg.max_epochs):
                # training phase
                for step in range(self.cfg.max_episode_length):
                    self.global_step += 1
                    self.episode_step += 1
                    total_loss_sum = 0.0

                    # train based on experiments
                    for batch in dataloader:
                        loss_list = self.loss_and_update(batch)
                        total_loss_sum += torch.sum(loss_list)

                    # execute in environment
                    actions, rewards, dones = self.play_step(self.epsilon)
                    self.episode_reward += np.sum(rewards)

                    # log
                    self.writer.add_scalar('training/epsilon', torch.tensor(self.epsilon), self.global_step)
                    self.writer.add_scalar('training/reward', torch.tensor(rewards).mean(), self.global_step)
                    self.writer.add_scalar('training/total_loss', total_loss_sum, self.global_step)

                self.episode_count += 1
                self.epsilon *= self.cfg.epsilon_decay
                self.epsilon = max(self.cfg.epsilon_end, self.epsilon)

                # update target network
                for agent in self.agents:
                    hard_update(agent.dqn_target, agent.dqn)

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
                pbar.set_postfix({'loss': total_loss_sum.item()})
                pbar.update(1)

        self.writer.close()
        pbar.close()

        model_path = 'model.pth'
        torch.save(self.agents[0].dqn.to('cpu').state_dict(), model_path)

    def validate(self):
        # put models on GPU and change to eval mode
        for agent in self.agents:
            agent.dqn.to(self.device)
            agent.dqn_target.to(self.device)
            agent.dqn.eval()
            agent.dqn_target.eval()

        with tqdm(total=self.cfg.validate_epochs) as pbar:
            for epoch in range(self.cfg.validate_epochs):
                for step in range(self.cfg.max_episode_length):
                    actions, rewards, dones = self.play_step()
                self.log_validate(epoch)
                self.reset()
                # updates pbar
                pbar.set_description('Validation')
                pbar.update(1)

        pbar.close()

    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):

        actions = list()
        for agent_id, agent in enumerate(self.agents):
            # normalize states [0, map.SIZE] -> [0, 1.0]
            states = torch.tensor(self.states).float()

            action = agent.get_action(states[agent_id], epsilon)
            actions.append(action)

            # heatmap update
            pos_x, pos_y = self.env.world.map.coord2ind(self.env.agents[agent_id].state.p_pos)
            self.heatmap_agents[agent_id, 2, pos_x, pos_y] += 1

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
        heatmap_events = 0.8 * self.env.heatmap_events / torch.max(self.env.heatmap_events)
        heatmap_events = torch.where(heatmap_events>0, heatmap_events+0.2, heatmap_events)
        self.heatmap_agents[:, torch.tensor([0, 1]), ...] += heatmap_events
        heatmap_agents = F.interpolate(self.heatmap_agents, size=(self.env.world.map.SIZE_X*10, self.env.world.map.SIZE_Y*10))
        heatmap_agents = torch.transpose(heatmap_agents, 2, 3)
        heatmap_agents = make_grid(heatmap_agents, nrow=2)
        self.writer.add_image('episode/heatmap_agents', heatmap_agents, self.episode_count, dataformats='CHW')

    def log_validate(self, epoch):
        # make directory
        path = 'validate'
        if not os.path.isdir(path):
            os.mkdir(path)
        epoch_path = os.path.join(path, 'epoch_'+str(epoch))
        hm_agents_path = os.path.join(epoch_path, 'hm_agents')
        hm_complete_path = os.path.join(epoch_path, 'hm_complete')
        os.mkdir(epoch_path)
        os.mkdir(hm_agents_path)
        os.mkdir(hm_complete_path)

        size_x = self.env.world.map.SIZE_X // 2
        size_y = self.env.world.map.SIZE_Y // 2
        for agent_id, agent in enumerate(self.agents):
            fig = plt.figure()
            sns.heatmap(
                self.env.heatmap_agents[agent_id], vmin=0, cmap='Blues',
                xticklabels=list(str(x) for x in range(-size_x, size_x)),
                yticklabels=list(str(y) for y in range(-size_y, size_y))
            )
            plt.title(f'Agent {agent_id}')
            fig.savefig(os.path.join(hm_agents_path, f'agent_{agent_id}.png'))
            plt.close()

            fig = plt.figure()
            sns.heatmap(
                self.env.heatmap_complete[agent_id], vmin=0, cmap='Blues',
                xticklabels=list(str(x) for x in range(-size_x, size_x)),
                yticklabels=list(str(y) for y in range(-size_y, size_y))
            )
            plt.title(f'Agent {agent_id}')
            fig.savefig(os.path.join(hm_complete_path, f'agent_{agent_id}.png'))
            plt.close()

        fig = plt.figure()
        sns.heatmap(
            self.env.heatmap_events, vmin=0, cmap='Blues',
            xticklabels=list(str(x) for x in range(-size_x, size_x)),
            yticklabels=list(str(y) for y in range(-size_y, size_y))
        )
        plt.title(f'Agent {agent_id}')
        fig.savefig(os.path.join(epoch_path, 'hm_events.png'))
        plt.close()

        fig = plt.figure()
        sns.heatmap(
            self.env.heatmap_wall_collision, vmin=0, cmap='Blues',
            xticklabels=list(str(x) for x in range(-size_x, size_x)),
            yticklabels=list(str(y) for y in range(-size_y, size_y))
        )
        plt.title(f'Agent {agent_id}')
        fig.savefig(os.path.join(epoch_path, 'hm_wall_collision.png'))
        plt.close()

        fig = plt.figure()
        sns.heatmap(
            self.env.heatmap_agents_collision, vmin=0, cmap='Blues',
            xticklabels=list(str(x) for x in range(-size_x, size_x)),
            yticklabels=list(str(y) for y in range(-size_y, size_y))
        )
        plt.title(f'Agent {agent_id}')
        fig.savefig(os.path.join(epoch_path, 'hm_agents_collision.png'))
        plt.close()


@hydra.main(config_path='../../conf/exp4.yaml')
def main(config: DictConfig):
    torch.manual_seed(921)
    np.random.seed(921)

    model = Exp4(config=config)
    if config.phase == 'training':
        model.fit()
    elif config.phase == 'validate':
        model.validate()

if __name__ == '__main__':
    main()
