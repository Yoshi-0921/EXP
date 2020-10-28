# -*- coding: utf-8 -*-

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
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from experiments.exp1.exp1_agent import DQNAgent
from experiments.exp1.exp1_env import Exp1_Env
from utils.buffer import Experience, ReplayBuffer
from utils.dataset import RLDataset
from utils.tools import hard_update


class Exp1:
    def __init__(self, config):
        super(Exp1, self).__init__()
        self.cfg = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.env = Exp1_Env(config)
        obs_size = self.env.observation_space
        act_size = self.env.action_space
        # initialize for agents
        self.buffer = ReplayBuffer(config.capacity)
        self.agents = [DQNAgent(obs_size[agent_id], act_size[agent_id], config) for agent_id in range(self.env.num_agents)]

        self.total_reward = 0
        self.global_step = 0
        self.episode_count = 0
        self.validation_count = 0

        self.states = self.env.reset()
        self.populate(config.populate_steps)
        self.reset()
        self.writer = SummaryWriter('exp1')

        # describe network
        print("""
================================================================
DQN Network Summary:""")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        summary(self.agents[0].dqn, (obs_size[0],), batch_size=self.cfg.batch_size, device=device)

    def populate(self, steps: int):
        for i in range(steps):
            _, _, _ = self.play_step(epsilon=1.0)

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

            # normalize states and rewards in range of [0, 1.0]
            state[:, 0::2] /= self.env.world.map.SIZE_X
            state[:, 1::2] /= self.env.world.map.SIZE_Y
            next_state[:, 0::2] /= self.env.world.map.SIZE_X
            next_state[:, 1::2] /= self.env.world.map.SIZE_Y

            loss.append(agent.update(state, action, reward, done, next_state))

        return loss

    def fit(self):
        # set dataloader
        dataset = RLDataset(self.buffer, self.cfg.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.cfg.batch_size)

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
                # validation phase
                if epoch % (self.cfg.max_epochs//100) == 0:
                    self.validation_count += 1
                    val_step = 0
                    episode_reward = 0.0
                    while True:
                        val_step += 1
                        epsilon = 0.0
                        actions, rewards, dones = self.play_step(epsilon)
                        episode_reward += np.sum(rewards)

                        if all(dones) or self.cfg.max_episode_length < val_step:
                            self.writer.add_scalar('validation/episode_reward', torch.tensor(episode_reward), self.validation_count)
                            self.writer.add_scalar('validation/episode_step', torch.tensor(val_step), self.validation_count)
                            self.reset()
                            break

                # training phase
                while True:
                    self.global_step += 1
                    self.episode_step += 1
                    total_loss_sum = 0.0

                    # train based on experiments
                    for batch in dataloader:
                        loss_list = self.loss_and_update(batch)

                        for loss in loss_list:
                            total_loss_sum += loss.item()

                    # update target network
                    if self.global_step % self.cfg.synch_epochs == 0:
                        for agent in self.agents:
                            hard_update(agent.dqn_target, agent.dqn)

                    # execute in environment
                    epsilon = max(0.1, 1.0 - (epoch+1)/self.cfg.decay_epochs)
                    actions, rewards, dones = self.play_step(epsilon)
                    self.episode_reward += np.sum(rewards)

                    # log
                    self.writer.add_scalar('training/epsilon', torch.tensor(epsilon), self.global_step)
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

                    if all(dones) or self.cfg.max_episode_length < self.episode_step:
                        self.episode_count += 1
                        self.writer.add_scalar('episode/episode_reward', torch.tensor(self.episode_reward), self.episode_count)
                        self.writer.add_scalar('episode/episode_step', torch.tensor(self.episode_step), self.episode_count)
                        self.writer.add_scalar('episode/global_step', torch.tensor(self.global_step), self.episode_count)
                        self.reset()
                        break

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
            states[:, 0::2] /= self.env.world.map.SIZE_Y
            states[:, 1::2] /= self.env.world.map.SIZE_Y

            action, self.q = agent.get_action(states[agent_id], epsilon)
            actions.append(action)

        new_states, rewards, dones = self.env.step(actions)

        exp = Experience(self.states, actions, rewards, dones, new_states)

        self.buffer.append(exp)

        self.states = new_states

        return actions, rewards, dones


@hydra.main(config_path='../../conf/exp1.yaml')
def main(config: DictConfig):
    torch.manual_seed(921)
    np.random.seed(921)

    exp = Exp1(config=config)
    exp.fit()

if __name__ == '__main__':
    main()
