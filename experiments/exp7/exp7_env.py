# -*- coding: utf-8 -*-

from random import random

import numpy as np
import math
import torch
from utils.core import Agent, Env, Landmark, Map, World


class Exp7_Env(Env):
    def __init__(self, config):
        super(Exp7_Env, self).__init__()
        self.cfg = config
        self.world = self.make_world(config)
        self.agents = self.world.agents
        self.num_agents = len(self.world.agents)
        self.num_landmarks = config.num_landmarks
        self.visible_range = config.visible_range
        self.action_space, self.observation_space = list(), list()

        """self.heatmap_agents = torch.zeros(self.num_agents, self.world.map.SIZE_X, self.world.map.SIZE_Y)
        self.heatmap_complete = torch.zeros(self.num_agents, self.world.map.SIZE_X, self.world.map.SIZE_Y)
        self.heatmap_events = torch.zeros(self.world.map.SIZE_X, self.world.map.SIZE_Y)
        self.heatmap_events_left = torch.zeros(self.world.map.SIZE_X, self.world.map.SIZE_Y)
        self.heatmap_wall_collision = torch.zeros(self.world.map.SIZE_X, self.world.map.SIZE_Y)
        self.heatmap_agents_collision = torch.zeros(self.world.map.SIZE_X, self.world.map.SIZE_Y)"""

        self.reset()
        self.describe_env()
        for agent in self.agents:
            self.action_space.append(4)
            self.observation_space.append(self.visible_range)

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
            # gap
            """if i==0: self.world.agents[0].state.p_pos = np.array([-1, 1])
            elif i==1: self.world.agents[1].state.p_pos = np.array([1, 1])
            elif i==2: self.world.agents[2].state.p_pos = np.array([-1, 0])
            elif i==3: self.world.agents[3].state.p_pos = np.array([1, 0])"""
            # after
            if self.num_agents == 4:
                if i==0: self.world.agents[0].state.p_pos = np.array([0, 1])
                elif i==1: self.world.agents[1].state.p_pos = np.array([-1, 1])
                elif i==2: self.world.agents[2].state.p_pos = np.array([-1, 0])
                elif i==3: self.world.agents[3].state.p_pos = np.array([0, 0])
            elif self.num_agents == 8 and self.world.map.SIZE_X == 37:
                if i==0: self.world.agents[0].state.p_pos = np.array([-2, 1])
                elif i==1: self.world.agents[1].state.p_pos = np.array([-1, 1])
                elif i==2: self.world.agents[2].state.p_pos = np.array([-2, 0])
                elif i==3: self.world.agents[3].state.p_pos = np.array([-1, 0])
                elif i==4: self.world.agents[4].state.p_pos = np.array([1, 1])
                elif i==5: self.world.agents[5].state.p_pos = np.array([2, 1])
                elif i==6: self.world.agents[6].state.p_pos = np.array([1, 0])
                elif i==7: self.world.agents[7].state.p_pos = np.array([2, 0])
            elif self.num_agents == 8 and self.world.map.SIZE_X == 40:
                if i==0: self.world.agents[0].state.p_pos = np.array([-3, 1])
                elif i==1: self.world.agents[1].state.p_pos = np.array([-2, 1])
                elif i==2: self.world.agents[2].state.p_pos = np.array([-3, 0])
                elif i==3: self.world.agents[3].state.p_pos = np.array([-2, 0])
                elif i==4: self.world.agents[4].state.p_pos = np.array([1, 1])
                elif i==5: self.world.agents[5].state.p_pos = np.array([2, 1])
                elif i==6: self.world.agents[6].state.p_pos = np.array([1, 0])
                elif i==7: self.world.agents[7].state.p_pos = np.array([2, 0])
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
        # 4 x SIZE_X x SIZE_Yの入力が欲しい
        # 0: me, 1:agents, 2:landmarks, 3:visible area
        obs = np.zeros((4, 40, 24), dtype=np.int8)
        offset = 0

        agent_pos = self.world.map.coord2ind(agent.state.p_pos)

        # 壁と見えないセルの入力
        obs[3, :, :] -= 1
        # 自分の場所は0
        obs[3, agent_pos[0], agent_pos[1]] = 0
        obs = self.fill_obs(obs, agent, offset, offset, 'area')

        # イベントの入力
        obs = self.fill_obs(obs, agent, offset, offset, 'event')

        # エージェントの入力
        obs[0, agent_pos[0], agent_pos[1]] = 1
        obs = self.fill_obs(obs, agent, offset, offset, 'agent')

        return obs

    def fill_obs(self, obs, agent, offset_x, offset_y, mode):
        if mode == 'area':

            for x in range(-1, 2):
                for y in [-1, 1]:
                    for opr in [-1, 1]:
                        for j in range(3):
                            pos_x, pos_y = x + j * opr, y + j * y
                            pos_x, pos_y = self.world.map.coord2ind((pos_x+agent.state.p_pos[0], pos_y+agent.state.p_pos[1]))
                            # 場外なら-1
                            if pos_x < 0 or self.world.map.SIZE_X <= pos_x or pos_y < 0 or self.world.map.SIZE_Y <= pos_y:
                                obs[3, pos_x, pos_y] = -1
                                break
                            # セルが真ん中で壁のない角の方向なら続ける
                            if j == 0 and x == 0 and self.world.map.matrix[pos_x+opr, pos_y, 0] == 0:
                                obs[3, pos_x, pos_y] = 0
                                continue
                            # 壁なら-1
                            if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                                obs[3, pos_x, pos_y] = -1
                                break
                            # セルが角で真ん中に壁があるならbreak
                            if j == 0 and x != 0 and self.world.map.matrix[pos_x-x, pos_y, 0] == 1 and opr != x:
                                break
                            # 何もないなら0
                            else:
                                obs[3, pos_x, pos_y] = 0

            for y in range(-1, 2):
                for x in [-1, 1]:
                    for opr in range(-1, 2):
                        for j in range(3):
                            pos_x, pos_y = x + j * x, y + j * opr
                            pos_x, pos_y = self.world.map.coord2ind((pos_x+agent.state.p_pos[0], pos_y+agent.state.p_pos[1]))
                            # 場外なら-1
                            if pos_x < 0 or self.world.map.SIZE_X <= pos_x or pos_y < 0 or self.world.map.SIZE_Y <= pos_y:
                                obs[3, pos_x, pos_y] = -1
                                break
                            # セルが真ん中で壁のない角の方向なら続ける
                            if j == 0 and y == 0 and self.world.map.matrix[pos_x, pos_y-opr, 0] == 0:
                                obs[3, pos_x, pos_y] = 0
                                continue
                            # 壁なら-1
                            if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                                obs[3, pos_x, pos_y] = -1
                                break
                            # セルが角で真ん中に壁があるならbreak
                            if j == 0 and y != 0 and self.world.map.matrix[pos_x, pos_y+y, 0] == 1 and opr != y:
                                break
                            # 何もないなら0
                            else:
                                obs[3, pos_x, pos_y] = 0

            for opr_x in [-1, 1]:
                for x in range(self.visible_range//2 + 1):
                    # 壁ならbreak
                    pos_x, pos_y = agent.state.p_pos[0] + (x * opr_x), agent.state.p_pos[1]
                    pos_x, pos_y = self.world.map.coord2ind((pos_x, pos_y))
                    if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                        obs[3, pos_x, pos_y] = -1
                        break
                    for opr_y in [-1, 1]:
                        for y in range(2):
                            pos_x, pos_y = agent.state.p_pos[0] + (x * opr_x), agent.state.p_pos[1] + (y * opr_y)
                            pos_x, pos_y = self.world.map.coord2ind((pos_x, pos_y))
                            # 場外なら-1
                            if pos_x < 0 or self.world.map.SIZE_X <= pos_x or pos_y < 0 or self.world.map.SIZE_Y <= pos_y:
                                obs[3, pos_x, pos_y] = -1
                                continue
                            # 壁なら-1
                            if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                                obs[3, pos_x, pos_y] = -1
                                break
                            # 何もないなら0
                            else:
                                obs[3, pos_x, pos_y] = 0

            for opr_y in [-1, 1]:
                for y in range(self.visible_range//2 + 1):
                    # 壁ならbreak
                    pos_x, pos_y = agent.state.p_pos[0], agent.state.p_pos[1] + (y * opr_y)
                    pos_x, pos_y = self.world.map.coord2ind((pos_x, pos_y))
                    if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                        obs[3, pos_x, pos_y] = -1
                        break
                    for opr_x in [-1, 1]:
                        for x in range(2):
                            pos_x, pos_y = agent.state.p_pos[0] + (x * opr_x), agent.state.p_pos[1] + (y * opr_y)
                            pos_x, pos_y = self.world.map.coord2ind((pos_x, pos_y))
                            local_pos_x, local_pos_y = self.world.map.coord2ind(((x * opr_x), (y * opr_y)), self.visible_range, self.visible_range)
                            # 場外なら-1
                            if pos_x < 0 or self.world.map.SIZE_X <= pos_x or pos_y < 0 or self.world.map.SIZE_Y <= pos_y:
                                obs[3, pos_x, pos_y] = -1
                                continue
                            # 壁なら-1
                            if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                                obs[3, pos_x, pos_y] = -1
                                break
                            # 何もないなら0
                            else:
                                obs[3, pos_x, pos_y] = 0

        elif mode == 'agent':
            for a in self.world.agents:
                diff_x, diff_y = a.state.p_pos - agent.state.p_pos
                if abs(diff_x) > 3 or abs(diff_y) > 3 or (diff_x== 0 and diff_y == 0):
                    continue
                pos_x, pos_y = self.world.map.coord2ind(a.state.p_pos)
                # 見える範囲なら追加
                if obs[3, pos_x, pos_y] != -1:
                    obs[1, pos_x, pos_y] = 1

        elif mode == 'event':
            for landmark in self.world.landmarks:
                if abs(landmark.state.p_pos[0]-agent.state.p_pos[0]) > 3 or abs(landmark.state.p_pos[1]-agent.state.p_pos[1]) > 3:
                    continue
                pos_x, pos_y = self.world.map.coord2ind(landmark.state.p_pos)
                # 見える範囲なら追加
                if obs[3, pos_x, pos_y] != -1:
                    obs[2, pos_x, pos_y] = 1

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
        world.map = Exp7_Map(config)

        return world

    def describe_env(self):
        print("""
    Experiment 7 Environment generated!

    ======================Action======================
    | 0: Right | 1: Up | 2: Left | 3: Down | ------- |
    ===================State(agent)===================
    | obs1 : 24x40 obs of agents                       |
    | obs2 : 24x40 obs of events                       |
    | obs3: 24x40 invisible area                       |
    | np.stack(ob1 + obs2 + obs3)                    |
    ======================Reward======================
    | rew1: +1 if success                            |
    | rew2 : -1 if is_collision                      |
    | rew1 + rew2                                    |
    ==================================================
    """)

class Exp7_Map(Map):
    def __init__(self, config):
        super(Exp7_Map, self).__init__()
        self.SIZE_X = 40 # 24, 40
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

    def locate_walls(self):
        self.matrix[np.array([0, self.SIZE_X-1]), :, 0] = 1
        self.matrix[:, np.array([0, self.SIZE_Y-1]), 0] = 1

        # exp (24, 24)
        if self.SIZE_X == 24:
            self.matrix[np.array([1,2,3,7,8,9,10,13,14,15,16,20,21,22]), 10, 0] = 1
            self.matrix[np.array([1,2,3,7,8,9,10,13,14,15,16,20,21,22]), 13, 0] = 1
            self.matrix[10, np.array([1,2,3,7,8,9,10,13,14,15,16,20,21,22]), 0] = 1
            self.matrix[13, np.array([1,2,3,7,8,9,10,13,14,15,16,20,21,22]), 0] = 1

        # exp (24, 37)
        elif self.SIZE_X == 37:
            self.matrix[18, :, 0] = 1
            self.matrix[18, np.array([11,12]), 0] = 0

            self.matrix[:, np.array([10,13]), 0] = 1
            self.matrix[np.arange(8, 11), 10, 0] = 0
            self.matrix[np.arange(26, 29), 10, 0] = 0
            self.matrix[np.arange(8, 11), 13, 0] = 0
            self.matrix[np.arange(26, 29), 13, 0] = 0

        # exp (24, 40)
        elif self.SIZE_X == 40:
            self.matrix[np.array([19,20]), :, 0] = 1
            self.matrix[19, np.array([11,12]), 0] = 0
            self.matrix[20, np.array([11,12]), 0] = 0

            self.matrix[:, np.array([10,13]), 0] = 1
            self.matrix[np.arange(8, 12), 10, 0] = 0
            self.matrix[np.arange(28, 32), 10, 0] = 0
            self.matrix[np.arange(8, 12), 13, 0] = 0
            self.matrix[np.arange(28, 32), 13, 0] = 0

    def set_aisle(self):
        # exp (24, 24)
        if self.SIZE_X == 24:
            self.aisle[np.arange(10, 14), :] = 1
            self.aisle[:, np.arange(10, 14)] = 1

        # exp (24, 37)
        elif self.SIZE_X == 37:
            self.aisle[:, np.arange(10, 14)] = 1

        # exp (24, 40)
        elif self.SIZE_X == 40:
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
