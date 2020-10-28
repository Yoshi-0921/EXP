# -*- coding: utf-8 -*-

from random import random

import numpy as np
import math
import torch
from utils.core import Agent, Env, Landmark, Map, World


class Exp5_Env(Env):
    def __init__(self, config):
        super(Exp5_Env, self).__init__()
        self.cfg = config
        self.world = self.make_world(config)
        self.agents = self.world.agents
        self.num_agents = len(self.world.agents)
        self.num_landmarks = config.num_landmarks
        self.visible_range = config.visible_range
        self.input_range = config.visible_range + 2 * (config.visible_range//2)
        self.action_space, self.observation_space = list(), list()
        self.reset()
        self.describe_env()
        for agent in self.agents:
            self.action_space.append(4)
            self.observation_space.append(self.input_range)

    def reset(self):
        self.events_generated = 0
        self.events_completed = 0
        self.agents_collided = 0
        self.walls_collided = 0
        self.world.map.reset()
        self.heatmap_agents = torch.zeros(self.num_agents, self.world.map.SIZE_X, self.world.map.SIZE_Y)
        self.heatmap_complete = torch.zeros(self.num_agents, self.world.map.SIZE_X, self.world.map.SIZE_Y)
        self.heatmap_events = torch.zeros(self.world.map.SIZE_X, self.world.map.SIZE_Y)
        self.heatmap_events_left = torch.zeros(self.world.map.SIZE_X, self.world.map.SIZE_Y)
        self.heatmap_wall_collision = torch.zeros(self.world.map.SIZE_X, self.world.map.SIZE_Y)
        self.heatmap_agents_collision = torch.zeros(self.world.map.SIZE_X, self.world.map.SIZE_Y)

        region = (self.world.map.SIZE_X//2) - 1
        # agentのposの初期化
        for i in range(self.num_agents):
            if i==0: self.world.agents[0].state.p_pos = np.array([-1, 1])
            elif i==1: self.world.agents[1].state.p_pos = np.array([1, 1])
            elif i==2: self.world.agents[2].state.p_pos = np.array([-1, 0])
            elif i==3: self.world.agents[3].state.p_pos = np.array([1, 0])
        for agent in self.world.agents:
            agent.collide_agents = False
            agent.collide_walls = False
        # landmarkのposの初期化
        self.world.landmarks = list()
        self.generate_events(self.num_landmarks)

        obs_n = list()
        for agent in self.agents:
            obs_n.append(self.__observation(agent))
        return obs_n

    def generate_events(self, num_events):
        #self.world.landmarks.appendしたい
        num_generated = 0
        while num_generated < num_events:
            x, y = 1 + int(random() * (self.world.map.SIZE_X-1)), 1 + int(random() * (self.world.map.SIZE_Y-1))
            if self.world.map.matrix[x, y, 0] == 0 and self.world.map.matrix[x, y, 2] == 0 and self.world.map.aisle[x, y] == 0:
                self.world.landmarks.append(Landmark())
                self.world.landmarks[-1].state.p_pos = self.world.map.ind2coord((x, y))
                self.world.map.matrix[x, y, 2] = 1
                self.heatmap_events[x, y] += 1
                num_generated += 1
                self.events_generated += 1

    def step(self, action_n):
        obs_n, reward_n, done_n = list(), list(), list()
        for agent_id, agent in enumerate(self.agents):
            self.__action(action_n[agent_id], agent)
        self.world.step()
        # record observation for each agent
        for agent_id, agent in enumerate(self.agents):
            obs_n.append(self.__observation(agent))
            reward_n.append(self.__reward(agent_id, agent))
            done_n.append(self.__done(agent))

        self.heatmap_events_left += self.world.map.matrix[..., 2]

        return obs_n, reward_n, done_n

    def __reward(self, agent_id, agent):
        def is_collision(agent1, agent2):
            delta_pos = agent1.state.p_pos - agent2.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            return True if dist == 0 else False

        # heatmap update
        a_pos_x, a_pos_y = self.world.map.coord2ind(agent.state.p_pos)
        self.heatmap_agents[agent_id, a_pos_x, a_pos_y] += 1

        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0.0
        for l_idx, l in enumerate(self.world.landmarks):
            if all(agent.state.p_pos == l.state.p_pos):
                rew = 1.0
                self.world.landmarks.pop(l_idx)
                e_pos_x, e_pos_y = self.world.map.coord2ind(l.state.p_pos)
                self.world.map.matrix[e_pos_x, e_pos_y, 2] = 0
                self.events_completed += 1
                self.heatmap_complete[agent_id, e_pos_x, e_pos_y] += 1
                self.generate_events(1)

        if agent.collide:
            # 他エージェントに衝突したら負の報酬
            if agent.collide_agents:
                rew = -1.0
                agent.collide_agents = False
                self.heatmap_agents_collision[a_pos_x, a_pos_y] += 1
                self.agents_collided += 1
            # 壁に衝突したら負の報酬
            if agent.collide_walls:
                rew = -1.0
                agent.collide_walls = False
                self.heatmap_wall_collision[a_pos_x, a_pos_y] += 1
                self.walls_collided += 1
        return rew

    def __observation(self, agent):
        # 3 x input_range x input_rangeの入力が欲しい
        # 0:agents, 1:landmarks, 2:invisible area
        obs = np.zeros((3, self.input_range, self.input_range), dtype=np.int8)
        offset = self.visible_range // 2

        # 壁と見えないセルの入力
        obs[2, :, :] -= 1
        obs = self.fill_obs(obs, agent, offset, offset, 'area')

        # イベントの入力
        obs = self.fill_obs(obs, agent, offset, offset, 'event')

        # エージェントの入力
        for a in self.world.agents:
            diff_x, diff_y = a.state.p_pos - agent.state.p_pos
            if abs(diff_x) > 3 or abs(diff_y) > 3 or (diff_x== 0 and diff_y == 0):
                continue
            pos_x, pos_y = self.world.map.coord2ind((a.state.p_pos[0]-agent.state.p_pos[0], a.state.p_pos[1]-agent.state.p_pos[1]),
                                                    size_x=self.visible_range, size_y=self.visible_range)
            # 見える範囲なら追加
            if obs[2, offset+pos_x, offset+pos_y] != -1:
                obs[0, offset+pos_x, offset+pos_y] = 1
                offset_x, offset_y = diff_x + offset, offset - diff_y
                # agent aが観測した情報の入力
                #obs = self.fill_obs(obs, a, offset_x, offset_y, 'area')
                #obs = self.fill_obs(obs, a, offset_x, offset_y, 'agent')
                #obs = self.fill_obs(obs, a, offset_x, offset_y, 'event')

        return obs

    def fill_obs(self, obs, agent, offset_x, offset_y, mode):
        if mode == 'area':
            # 自分の場所は0
            obs[2, offset_x+self.visible_range//2, offset_y+self.visible_range//2] = 0
            for x in range(-1, 2):
                for y in [-1, 1]:
                    for opr in range(-1, 2):
                        for j in range(3):
                            pos_x, pos_y = x + j * opr, y + j * y
                            local_pos_x, local_pos_y = self.world.map.coord2ind((pos_x, pos_y), self.visible_range, self.visible_range)
                            pos_x, pos_y = self.world.map.coord2ind((pos_x+agent.state.p_pos[0], pos_y+agent.state.p_pos[1]))
                            # 場外なら-1
                            if pos_x < 0 or self.world.map.SIZE_X <= pos_x or pos_y < 0 or self.world.map.SIZE_Y <= pos_y:
                                obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = -1
                                continue
                            # 壁なら-1
                            if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                                obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = -1
                                break
                            # 何もないなら0
                            else:
                                obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = 0

            for y in range(-1, 2):
                for x in [-1, 1]:
                    for opr in range(-1, 2):
                        for j in range(3):
                            pos_x, pos_y = x + j * x, y + j * opr
                            local_pos_x, local_pos_y = self.world.map.coord2ind((pos_x, pos_y), self.visible_range, self.visible_range)
                            pos_x, pos_y = self.world.map.coord2ind((pos_x+agent.state.p_pos[0], pos_y+agent.state.p_pos[1]))
                            # 場外なら-1
                            if pos_x < 0 or self.world.map.SIZE_X <= pos_x or pos_y < 0 or self.world.map.SIZE_Y <= pos_y:
                                obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = -1
                                continue
                            # 壁なら-1
                            if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                                obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = -1
                                break
                            # 何もないなら0
                            else:
                                obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = 0

        elif mode == 'agent':
            for a in self.world.agents:
                diff_x, diff_y = a.state.p_pos - agent.state.p_pos
                if abs(diff_x) > 3 or abs(diff_y) > 3 or (diff_x== 0 and diff_y == 0):
                    continue
                pos_x, pos_y = self.world.map.coord2ind((a.state.p_pos[0]-agent.state.p_pos[0], a.state.p_pos[1]-agent.state.p_pos[1]),
                                                        size_x=self.visible_range, size_y=self.visible_range)
                # 見える範囲なら追加
                if obs[2, offset_x + pos_x, offset_y + pos_y] != -1:
                    obs[0, offset_x + pos_x, offset_y + pos_y] = 1

        elif mode == 'event':
            for landmark in self.world.landmarks:
                if abs(landmark.state.p_pos[0]-agent.state.p_pos[0]) > 3 or abs(landmark.state.p_pos[1]-agent.state.p_pos[1]) > 3:
                    continue
                pos_x, pos_y = self.world.map.coord2ind((landmark.state.p_pos[0]-agent.state.p_pos[0], landmark.state.p_pos[1]-agent.state.p_pos[1]),
                                                        size_x=self.visible_range, size_y=self.visible_range)
                # 見える範囲なら追加
                if obs[2, offset_x + pos_x, offset_y + pos_y] != -1:
                    obs[1, offset_x + pos_x, offset_y + pos_y] = 1

        return obs

    def __done(self, agent):
        for landmark in self.world.landmarks:
            if all(agent.state.p_pos == landmark.state.p_pos):
                return 1

        return 0

    def __action(self, action, agent):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            agent.action.u = np.zeros(self.world.dim_p)
            if action == 0: agent.action.u[0] = 1.0
            elif action == 1: agent.action.u[1] = 1.0
            elif action == 2: agent.action.u[0] = -1.0
            elif action == 3: agent.action.u[1] = -1.0

    def make_world(self, config):
        world = World()
        num_agents = config.num_agents
        num_landmarks = config.num_landmarks
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent {i}'
            agent.collide = True
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'landmark {i}'
            landmark.collide = False
        world.map = Exp5_Map(config)

        return world

    def describe_env(self):
        print("""
    Experiment 5 Environment generated!

    ======================Action======================
    | 0: Right | 1: Up | 2: Left | 3: Down | ------- |
    ===================State(agent)===================
    | obs1 : 7x7 obs of agents                       |
    | obs2 : 7x7 obs of events                       |
    | obs3: 7x7 invisible area                       |
    | np.stack(ob1 + obs2 + obs3)                    |
    ======================Reward======================
    | rew1: +1 if success                            |
    | rew2 : -1 if is_collision                      |
    | rew1 + rew2                                    |
    ==================================================
    """)

class Exp5_Map(Map):
    def __init__(self, config):
        super(Exp5_Map, self).__init__()
        self.SIZE_X = 24 # 37
        self.SIZE_Y = 24
        # 0:walls, 1:agents, 2:landmarks
        self.matrix = np.zeros((self.SIZE_X, self.SIZE_Y, 3), dtype=np.int8)
        self.matrix_probs = np.zeros((self.SIZE_X, self.SIZE_Y), dtype=np.float)
        self.aisle = np.zeros((self.SIZE_X, self.SIZE_Y), dtype=np.float)
        self.locate_walls()
        self.set_aisle()
        self.set_probs()

    def reset(self):
        self.matrix[..., 2] = np.zeros((self.SIZE_X, self.SIZE_Y), dtype=np.int8)

    def coord2ind(self, p_pos, size_x=None, size_y=None):
        if size_x == None:
            size_x = self.SIZE_X
            size_y = self.SIZE_Y
        pos_x, pos_y = p_pos
        pos_x = (size_x // 2) + pos_x
        pos_y = (size_y // 2) - pos_y
        res = np.array([pos_x, pos_y])

        return res

    def ind2coord(self, p_pos, size_x=None, size_y=None):
        if size_x == None:
            size_x = self.SIZE_X
            size_y = self.SIZE_Y
        pos_x, pos_y = p_pos
        pos_x = pos_x - (size_x // 2)
        pos_y = (size_y // 2) - pos_y
        res = np.array([pos_x, pos_y])

        return res

    """def locate_walls(self):
        self.matrix[np.array([0, self.SIZE_X-1]), :, 0] = 1
        self.matrix[:, np.array([0, self.SIZE_Y-1]), 0] = 1
        self.matrix[:, np.array([10, 13]), 0] = 1
        self.matrix[18, :, 0] = 1
        self.matrix[18, np.array([11, 12]), 0] = 0
        self.matrix[np.array([9, 27]), 10, 0] = 0
        self.matrix[np.array([9, 27]), 13, 0] = 0

    def set_aisle(self):
        self.aisle[:, np.array([11, 12])] = 1"""

    def locate_walls(self):
        self.matrix[np.array([0, self.SIZE_X-1]), :, 0] = 1
        self.matrix[:, np.array([0, self.SIZE_Y-1]), 0] = 1
        self.matrix[np.array([1,2,3,7,8,9,10,13,14,15,16,20,21,22]), 10, 0] = 1
        self.matrix[np.array([1,2,3,7,8,9,10,13,14,15,16,20,21,22]), 13, 0] = 1
        self.matrix[10, np.array([1,2,3,7,8,9,10,13,14,15,16,20,21,22]), 0] = 1
        self.matrix[13, np.array([1,2,3,7,8,9,10,13,14,15,16,20,21,22]), 0] = 1

    def set_aisle(self):
        self.aisle[np.arange(10, 14), :] = 1
        self.aisle[:, np.arange(10, 14)] = 1

    def locate_agents(self):
        for agent in self.agents:
            pos_x, pos_y = self.coord2ind(agent.state.p_pos)
            self.matrix[pos_x, pos_y, 1] = 1

    def locate_landmarks(self):
        for landmark in self.landmarks:
            pos_x, pos_y = self.coord2ind(landmark.state.p_pos)
            self.matrix[pos_x, pos_y, 2] = 1

    def print_walls(self):
        print(self.matrix.transpose(2, 0, 1)[0])

    def print_probs(self):
        print(self.matrix_probs)

    def set_probs(self):
        """
        イベントの確率分布を設定
        """
        pass
        """# 右上
        for x in range(16, 21):
            for y in range(3, 8):
                self.matrix_probs[x, y] = 1e-3
        # 左上
        for x in range(3, 8):
            for y in range(3, 8):
                self.matrix_probs[x, y] = 1e-3
        # 左下
        for x in range(3, 8):
            for y in range(16, 21):
                self.matrix_probs[x, y] = 1e-3
        # 右下
        for x in range(16, 21):
            for y in range(16, 21):
                self.matrix_probs[x, y] = 1e-3"""
