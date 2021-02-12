# -*- coding: utf-8 -*-

import os
import pathlib
import sys
import warnings
import random

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../../')
warnings.simplefilter('ignore')

import hydra
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns
import torch
from experiments.exp7.exp7_agent import DQNAgent
from experiments.exp7.exp7_env import Exp7_Env
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


class Exp7:
    def __init__(self, config):
        super(Exp7, self).__init__()
        self.cfg = config
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.env = Exp7_Env(config)
        obs_size = self.env.observation_space
        act_size = self.env.action_space
        # initialize for agents
        self.buffer = ReplayBuffer(config.capacity, state_conv=True)
        self.agents = [DQNAgent(obs_size[agent_id], act_size[agent_id], config) for agent_id in range(self.env.num_agents)]
        self.order = np.arange(len(self.agents))

        # load state_dict
        if config.load_weight:
            for agent_id, agent in enumerate(self.agents):
                agent.dqn.load_state_dict(torch.load(os.path.join(config.model_path, f'agent_{agent_id}.pth')))

        self.total_reward = 0
        self.global_step = 0
        self.episode_count = 0
        self.validation_count = 0
        self.epsilon = config.epsilon_initial
        self.visible_range = config.visible_range

        self.writer = SummaryWriter('exp7')
        print(self.env.world.map.matrix[...,0].T)

        self.reset()
        torch.backends.cudnn.benchmark = True
        # describe network
        print("""
================================================================
DQN Network Summary:""")
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #summary(self.agents[0].dqn, (3, obs_size[0], obs_size[0]), batch_size=config.batch_size, device=device)

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
                if epoch % (self.cfg.max_epochs//10) == 0:
                    for agent_id, agent in enumerate(self.agents):
                        model_path = f'epoch{epoch}_agent{agent_id}.pth'
                        torch.save(agent.dqn.to('cpu').state_dict(), model_path)
                        agent.dqn.to(self.device)
                # training phase
                for step in range(self.cfg.max_episode_length):
                    total_loss_sum = 0.0

                    # train based on experiments
                    for batch in dataloader:
                        loss_list = self.loss_and_update(batch)
                        total_loss_sum += torch.sum(loss_list)

                    # execute in environment
                    states, rewards, attention_maps = self.play_step(self.epsilon)
                    self.episode_reward += np.sum(rewards)

                    if epoch % 10 == 0 and step % (self.cfg.max_episode_length//5) == 0 and False:
                        # log attention_maps of agent0
                        for agent_id in range(len(self.agents)):
                            # 各headの出力を足し合わせる
                            attention_map = attention_maps[agent_id].mean(dim=0).sum(dim=1)[:, 0, 1:].view(-1, 10, 6)
                            am = F.interpolate(attention_map.unsqueeze(0), size=(10*20, 6*20))[0, 0]
                            self.writer.add_image(f'attention_{agent_id}/heatmap_mean', torch.t(am), self.global_step, dataformats='HW')
                            am /= torch.max(am)
                            self.writer.add_image(f'attention_{agent_id}/adjusted_heatmap_mean', torch.t(am), self.global_step, dataformats='HW')

                            #states[:, 0, self.visible_range//2, self.visible_range//2] = 1
                            state = F.interpolate(states, size=(10*20, 6*20))[agent_id]
                            image = torch.zeros((3, 10*20, 6*20), dtype=torch.float)

                            # agentの情報を追加(Blue)
                            image[2, ...] += state[0]
                            image[2, ...] += state[1]
                            # landmarkの情報を追加(Yellow)
                            image[0, ...] += state[2]
                            image[1, ...] += state[2]
                            # invisible areaの情報を追加(White)
                            image[:, ...] -= state[3]
                            self.writer.add_image(f'attention_{agent_id}/observation', image.permute(0, 2, 1), self.global_step, dataformats='CHW')

                    # log
                    self.writer.add_scalar('training/epsilon', torch.tensor(self.epsilon), self.global_step)
                    self.writer.add_scalar('training/reward', torch.tensor(rewards).mean(), self.global_step)
                    self.writer.add_scalar('training/total_loss', total_loss_sum, self.global_step)

                    self.global_step += 1
                    self.episode_step += 1

                self.episode_count += 1
                self.epsilon *= self.cfg.epsilon_decay
                self.epsilon = max(self.cfg.epsilon_end, self.epsilon)
                #self.epsilon = 0.05#max(0.05, 1.0 - (epoch+1)/7500)

                # update target network
                for agent in self.agents:
                    hard_update(agent.dqn_target, agent.dqn)

                self.log_scalars()
                self.log_heatmap()
                self.reset()

                # updates pbar
                pbar.set_description(f'[Step {self.global_step}]')
                pbar.set_postfix({'loss': total_loss_sum.item()})
                pbar.update(1)

        self.writer.close()
        pbar.close()

        for agent_id, agent in enumerate(self.agents):
            model_path = f'agent_{agent_id}.pth'
            torch.save(agent.dqn.to('cpu').state_dict(), model_path)

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
                    states, rewards, attention_maps = self.play_step()
                    self.episode_reward += np.sum(rewards)
                    self.log_attention(states, attention_maps)

                    self.global_step += 1
                    self.episode_step += 1

                self.log_scalars()
                self.log_heatmap()
                #if epoch % 100 == 0:
                #    self.log_validate()
                self.episode_count += 1
                self.reset()

                # updates pbar
                pbar.set_description('Validation')
                pbar.update(1)
            self.log_validate()

        self.writer.close()
        pbar.close()

    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):

        actions = [[] for i in range (self.env.num_agents)]
        attention_maps = [[] for i in range(self.env.num_agents)]
        random.shuffle(self.order)

        for agent_id in self.order:
            # normalize states [0, map.SIZE] -> [0, 1.0]
            states = torch.tensor(self.states).float()

            action, attns = self.agents[agent_id].get_action(states[agent_id], epsilon)
            actions[agent_id] = action
            attention_maps[agent_id] = attns

        new_states, rewards, dones = self.env.step(actions)

        exp = Experience(self.states, actions, rewards, dones, new_states)

        self.buffer.append(exp)

        self.states = new_states

        return states, rewards, attention_maps

    def log_attention(self, states, attention_maps):
        # make directory
        path = 'validate'
        if not os.path.isdir(path):
            os.mkdir(path)
        epoch_path = os.path.join(path, 'epoch_'+str(self.episode_count))
        if not os.path.isdir(epoch_path):
            os.mkdir(epoch_path)
        episodes_path = os.path.join(epoch_path, 'episodes')
        if not os.path.isdir(episodes_path):
            os.mkdir(episodes_path)
        episode_path = os.path.join(episodes_path, 'episode_'+str(self.episode_step))
        if not os.path.isdir(episode_path):
            os.mkdir(episode_path)

        states[:, 0, self.visible_range//2, self.visible_range//2] = 1
        states = F.interpolate(states, size=(10*20, 6*20))

        for agent_id, (state, attention_map) in enumerate(zip(states, attention_maps)):
            agent_path = os.path.join(episode_path, 'agent_'+str(agent_id))
            if not os.path.isdir(agent_path):
                os.mkdir(agent_path)

            image = np.zeros((10*20, 6*20, 3), dtype=np.float)
            obs = state.numpy() * 255.0

            # agentの情報を追加(Blue)
            image[..., 0] += obs[0]
            image[..., 0] += obs[1]
            # landmarkの情報を追加(Yellow)
            image[..., 1] += obs[2]
            image[..., 2] += obs[2]
            # invisible areaの情報を追加(White)
            image[..., 0] -= obs[3]
            image[..., 1] -= obs[3]
            image[..., 2] -= obs[3]

            cv2.imwrite(str(agent_path)+f'/observation.png', image.transpose(1,0,2))

            attention_map = attention_map.mean(dim=0)[0, :, 0, 1:].view(-1, 10, 6).cpu().detach()
            for head_id, am in enumerate(attention_map):
                fig = plt.figure()
                sns.heatmap(
                    torch.t(am), vmin=0, square=True, cbar_kws={"shrink": .65},
                )
                fig.savefig(os.path.join(agent_path, f'attention_head_{head_id}.png'))
                plt.close()
            fig = plt.figure()
            sns.heatmap(
                torch.t(attention_map.mean(dim=0)), vmin=0, square=True, cbar_kws={"shrink": .65},
            )
            fig.savefig(os.path.join(agent_path, 'attention_heads_mean.png'))
            plt.close()

    def log_scalars(self):
        self.writer.add_scalar('episode/episode_reward', torch.tensor(self.episode_reward), self.episode_count)
        self.writer.add_scalar('episode/episode_step', torch.tensor(self.episode_step), self.episode_count)
        self.writer.add_scalar('episode/global_step', torch.tensor(self.global_step), self.episode_count)
        self.writer.add_scalar('env/events_left', torch.tensor(self.env.events_generated-self.env.events_completed), self.episode_count)
        self.writer.add_scalar('env/events_completed', torch.tensor(self.env.events_completed), self.episode_count)
        self.writer.add_scalar('env/agents_collided', torch.tensor(self.env.agents_collided), self.episode_count)
        self.writer.add_scalar('env/walls_collided', torch.tensor(self.env.walls_collided), self.episode_count)

    def log_heatmap(self):
        heatmap = torch.zeros(self.env.num_agents, 3, self.env.world.map.SIZE_X, self.env.world.map.SIZE_Y)

        for i in range(self.env.num_agents):
            # pathの情報を追加
            heatmap_agents = 0.5 * self.env.heatmap_agents[i, ...] / torch.max(self.env.heatmap_agents[i, ...])
            heatmap_agents = torch.where(heatmap_agents>0, heatmap_agents+0.5, heatmap_agents)
            heatmap[i, 2, ...] += heatmap_agents

        # 壁の情報を追加
        heatmap[:, :, ...] += torch.tensor(self.env.world.map.matrix[..., 0])

        # eventsの情報を追加
        heatmap_events = 0.8 * self.env.heatmap_events / torch.max(self.env.heatmap_events)
        heatmap_events = torch.where(heatmap_events>0, heatmap_events+0.2, heatmap_events)
        heatmap[:, torch.tensor([0, 1]), ...] += heatmap_events

        heatmap = F.interpolate(heatmap, size=(self.env.world.map.SIZE_X*10, self.env.world.map.SIZE_Y*10))
        heatmap = torch.transpose(heatmap, 2, 3)
        heatmap = make_grid(heatmap, nrow=2)
        self.writer.add_image('episode/heatmap', heatmap, self.episode_count, dataformats='CHW')

    def log_validate(self):
        # make directory
        path = 'validate'
        if not os.path.isdir(path):
            os.mkdir(path)
        epoch_path = os.path.join(path, 'epoch_'+str(self.episode_count))
        if not os.path.isdir(epoch_path):
            os.mkdir(epoch_path)
        hm_agents_path = os.path.join(epoch_path, 'hm_agents')
        if not os.path.isdir(hm_agents_path):
            os.mkdir(hm_agents_path)
        hm_complete_path = os.path.join(epoch_path, 'hm_complete')
        if not os.path.isdir(hm_complete_path):
            os.mkdir(hm_complete_path)

        size_x = self.env.world.map.SIZE_X // 2
        size_y = self.env.world.map.SIZE_Y // 2
        for agent_id, agent in enumerate(self.agents):
            # log heatmap_agents
            fig = plt.figure()
            sns.heatmap(
                torch.t(self.env.heatmap_agents[agent_id]), vmin=0, cmap='Blues', square=True, cbar_kws={"shrink": .65},
                xticklabels=list(str(x) if x%2==0 else '' for x in range(-size_x, size_x)),
                yticklabels=list(str(y) if y%2==0 else '' for y in range(size_y, -size_y, -1))
            )
            plt.title(f'Agent {agent_id}')
            fig.savefig(os.path.join(hm_agents_path, f'agent_{agent_id}.png'))
            plt.close()

            # log heatmap_complete
            fig = plt.figure()
            sns.heatmap(
                torch.t(self.env.heatmap_complete[agent_id]), vmin=0, cmap='Blues', square=True, cbar_kws={"shrink": .65},
                xticklabels=list(str(x) if x%2==0 else '' for x in range(-size_x, size_x)),
                yticklabels=list(str(y) if y%2==0 else '' for y in range(size_y, -size_y, -1))
            )
            plt.title(f'Agent {agent_id}')
            fig.savefig(os.path.join(hm_complete_path, f'agent_{agent_id}.png'))
            plt.close()

        # log heatmap_events
        fig = plt.figure()
        sns.heatmap(
            torch.t(self.env.heatmap_events), vmin=0, cmap='Blues', square=True, cbar_kws={"shrink": .65},
            xticklabels=list(str(x) if x%2==0 else '' for x in range(-size_x, size_x)),
            yticklabels=list(str(y) if y%2==0 else '' for y in range(size_y, -size_y, -1))
        )
        fig.savefig(os.path.join(epoch_path, 'hm_events.png'))
        plt.close()

        # log heatmap_events_left
        fig = plt.figure()
        sns.heatmap(
            torch.t(self.env.heatmap_events_left), vmin=0, cmap='Blues', square=True, cbar_kws={"shrink": .65},
            xticklabels=list(str(x) if x%2==0 else '' for x in range(-size_x, size_x)),
            yticklabels=list(str(y) if y%2==0 else '' for y in range(size_y, -size_y, -1))
        )
        fig.savefig(os.path.join(epoch_path, 'hm_events_left.png'))
        plt.close()

        # log heatmap_wall_collision
        fig = plt.figure()
        sns.heatmap(
            torch.t(self.env.heatmap_wall_collision), vmin=0, cmap='Blues', square=True, cbar_kws={"shrink": .65},
            xticklabels=list(str(x) if x%2==0 else '' for x in range(-size_x, size_x)),
            yticklabels=list(str(y) if y%2==0 else '' for y in range(size_y, -size_y, -1))
        )
        fig.savefig(os.path.join(epoch_path, 'hm_wall_collision.png'))
        plt.close()

        # log heatmap_agents_collision
        fig = plt.figure()
        sns.heatmap(
            torch.t(self.env.heatmap_agents_collision), vmin=0, cmap='Blues', square=True, cbar_kws={"shrink": .65},
            xticklabels=list(str(x) if x%2==0 else '' for x in range(-size_x, size_x)),
            yticklabels=list(str(y) if y%2==0 else '' for y in range(size_y, -size_y, -1))
        )
        fig.savefig(os.path.join(epoch_path, 'hm_agents_collision.png'))
        plt.close()


@hydra.main(config_path='../../conf/exp7.yaml')
def main(config: DictConfig):
    seed = 1999 # 921, 1998, 1411, 331, 1999
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    exp = Exp7(config=config)
    if config.phase == 'training':
        exp.fit()
    elif config.phase == 'validate':
        exp.validate()

if __name__ == '__main__':
    main()
