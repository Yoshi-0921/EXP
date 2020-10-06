# -*- coding: utf-8 -*-

from random import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from experiments.exp2.exp2_agent import DDPGAgent
from experiments.exp2.exp2_env import Exp2_Env
from utils.buffer import Experience, ReplayBuffer
from utils.dataset import RLDataset


class Exp2:
    def __init__(self):
        super(Exp2, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.env = Exp2_Env()
        obs_size = self.env.observation_space
        act_size = self.env.action_space
        # initialize for agents
        self.buffer = ReplayBuffer(1000000, action_onehot=True)
        self.agents = [DDPGAgent(obs_size[agent_id], act_size[agent_id]) for agent_id in range(self.env.num_agents)]

        self.total_reward = 0
        self.global_step = 0
        self.episode_count = 0
        self.validation_count = 0

        self.states = self.env.reset()
        self.populate()
        self.reset()
        self.writer = SummaryWriter()

    def populate(self, steps=1000):
        for i in range(steps):
            _, _, _ = self.play_step(epsilon=1.0)

    def reset(self):
        self.states = self.env.reset()
        self.episode_reward = 0
        self.episode_step = 0

    def loss_and_update(self, batch):
        value_loss_list, policy_loss_list = list(), list()
        states, logits, rewards, dones, next_states = batch
        for agent_id, agent in enumerate(self.agents):
            state = states[agent_id].float().to(self.device)
            logit = logits[agent_id].to(self.device)
            reward = rewards[agent_id].to(self.device)
            done = dones[agent_id].to(self.device)
            next_state = next_states[agent_id].float().to(self.device)

            # normalize states in range of [0, 1.0]
            state[:, 0::2] /= self.env.world.map.SIZE_X
            state[:, 1::2] /= self.env.world.map.SIZE_Y
            next_state[:, 0::2] /= self.env.world.map.SIZE_X
            next_state[:, 1::2] /= self.env.world.map.SIZE_Y

            value_loss, policy_loss = agent.update(state, logit, reward, done, next_state)
            value_loss_list.append(value_loss)
            policy_loss_list.append(policy_loss)

        return value_loss_list, policy_loss_list

    def fit(self):
        # hard coding
        max_epochs = 10000

        # set dataloader
        dataset = RLDataset(self.buffer, 64)
        dataloader = DataLoader(dataset=dataset, batch_size=64)

        # put models on GPU and change to training mode
        for agent in self.agents:
            agent.actor.to(self.device)
            agent.target_actor.to(self.device)
            agent.critic.to(self.device)
            agent.target_critic.to(self.device)
            agent.actor.eval()
            agent.target_actor.eval()
            agent.critic.eval()
            agent.target_critic.eval()

        # training loop
        torch.backends.cudnn.benchmark = True
        with tqdm(total=max_epochs) as pbar:
            for epoch in range(max_epochs):
                # validation phase
                if epoch % (max_epochs//100) == 0:
                    self.validation_count += 1
                    val_step = 0
                    episode_reward = 0.0
                    while True:
                        val_step += 1
                        epsilon = 0.0
                        _, rewards, dones = self.play_step(epsilon)
                        episode_reward += np.sum(rewards)

                        if all(dones) or 20 < val_step:
                            self.writer.add_scalar('validation/episode_reward', torch.tensor(episode_reward), self.validation_count)
                            self.writer.add_scalar('validation/episode_step', torch.tensor(val_step), self.validation_count)
                            self.reset()
                            break

                # training phase
                while True:
                    self.global_step += 1
                    self.episode_step += 1
                    total_loss_sum = 0.0
                    value_loss_sum = 0.0
                    policy_loss_sum = 0.0

                    # train based on experiments
                    for batch in dataloader:
                        value_loss_list, policy_loss_list = self.loss_and_update(batch)

                        for value_loss, policy_loss in zip(value_loss_list, policy_loss_list):
                            total_loss_sum += (value_loss + policy_loss)
                            value_loss_sum += value_loss
                            policy_loss_sum += policy_loss

                    # execute in environment
                    epsilon = max(0.1, 1.0 - (epoch+1)/3000)
                    actions, rewards, dones = self.play_step(epsilon)
                    self.episode_reward += np.sum(rewards)

                    # log
                    self.writer.add_scalar('training/epsilon', torch.tensor(epsilon), self.global_step)
                    self.writer.add_scalar('training/reward', torch.tensor(rewards).mean(), self.global_step)
                    self.writer.add_scalar('training/total_loss', torch.tensor(total_loss_sum), self.global_step)
                    self.writer.add_scalar('training/value_loss', torch.tensor(value_loss_sum), self.global_step)
                    self.writer.add_scalar('training/policy_loss', torch.tensor(policy_loss_sum), self.global_step)

                    # print on terminal
                    if epoch % (max_epochs//10) == 0:
                        print(f"""
    logits: {self.logits}
    actions: {actions}
    rewards: {rewards}

    agent: {self.env.agents[0].state.p_pos}
    landmark: {self.env.world.landmarks[0].state.p_pos}""")

                    if all(dones) or 20 < self.episode_step:
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

        actions, logits = list(), list()
        for agent_id, agent in enumerate(self.agents):
            # normalize states [0, map.SIZE] -> [0, 1.0]
            states = torch.tensor(self.states).float()
            states[:, 0::2] /= self.env.world.map.SIZE_Y
            states[:, 1::2] /= self.env.world.map.SIZE_Y

            action, logit = agent.get_action(states[agent_id], epsilon)
            actions.append(action)
            logits.append(logit)
        self.logits = logits

        next_states, rewards, dones = self.env.step(actions)

        exp = Experience(self.states, logits, rewards, dones, next_states)

        self.buffer.append(exp)

        self.states = next_states

        return actions, rewards, dones


if __name__ == '__main__':
    torch.manual_seed(921)
    np.random.seed(921)

    model = Exp2()
    model.fit()