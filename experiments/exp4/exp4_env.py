# -*- coding: utf-8 -*-

import numpy as np
from random import random
from utils.core import Env, World, Agent, Landmark, Map

class Exp4_Env(Env):
    def __init__(self, config):
        super(Exp4_Env, self).__init__()
        self.cfg = config
        self.world = self.make_world(config)
        self.agents = self.world.agents
        self.num_agents = len(self.world.agents)
        self.action_space, self.observation_space = list(), list()
        self.reset()

        self.describe_env()

        for agent in self.agents:
            self.action_space.append(5)
            self.observation_space.append(self.__observation(agent).shape[0])

    def reset(self):
        self.events_generated = 0
        self.events_completed = 0
        self.agents_collided = 0
        self.walls_collided = 0

        region = (self.world.map.SIZE_X//2) - 1
        # agentのposの初期化
        #for agent in self.world.agents:
            #agent.state.p_pos = np.random.randint(-region, region, self.world.dim_p)
            #agent.state.p_pos = np.array([0, 0])
        for i in range(self.num_agents):
            if i==0: self.world.agents[0].state.p_pos = np.array([0, 0]) # (12, 12)
            elif i==1: self.world.agents[1].state.p_pos = np.array([-1, 0]) # (11, 12)
            elif i==2: self.world.agents[2].state.p_pos = np.array([-1, -1]) # (11, 11)
            elif i==3: self.world.agents[3].state.p_pos = np.array([0, -1]) # (12, 11)
        for agent in self.world.agents:
            agent.collide_walls = False
        # landmarkのposの初期化
        """self.world.landmarks = [Landmark() for i in range(self.cfg.num_landmarks)]
        for landmark in self.world.landmarks:
            landmark.state.p_pos = np.random.randint(-region, region, self.world.dim_p)"""
        self.world.landmarks = list()
        self.generate_events(20)

        self.world.map.reset(agents=self.world.agents, landmarks=self.world.landmarks)

        obs_n = list()
        for agent in self.agents:
            obs_n.append(self.__observation(agent))
        return obs_n

    def generate_events(self, num_events):
        #self.world.landmarks.appendしたい
        num_generated = 0
        while num_generated < num_events:
            x, y = int(random() * self.world.map.SIZE_X), int(random() * self.world.map.SIZE_Y)
            if self.world.map.matrix[x, y, 0] == 0 and self.world.map.matrix[x, y, 2] == 0:
                self.world.landmarks.append(Landmark())
                self.world.landmarks[-1].state.p_pos = self.world.map.ind2coord((x, y))
                self.world.map.matrix[x, y, 2] = 1
                num_generated += 1
                self.events_generated += 1

        """for x in range(self.world.map.SIZE_X):
            for y in range(self.world.map.SIZE_Y):
                if random() < self.world.map.matrix_probs[x, y]:
                    self.world.landmarks.append(Landmark())
                    self.world.landmarks[-1].state.p_pos = self.world.map.ind2coord((x, y))"""

    def step(self, action_n):
        obs_n, reward_n, done_n = list(), list(), list()
        for i, agent in enumerate(self.agents):
            self.__action(action_n[i], agent)
        self.world.step()
        # generate events
        #self.generate_events()
        # update map
        self.world.map.step(self.agents)
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self.__observation(agent))
            reward_n.append(self.__reward(agent))
            done_n.append(self.__done(agent))

        return obs_n, reward_n, done_n

    def __reward(self, agent):
        def is_collision(agent1, agent2):
            delta_pos = agent1.state.p_pos - agent2.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            return True if dist == 0 else False

        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0.0
        for l_idx, l in enumerate(self.world.landmarks):
            #dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in self.world.agents]
            #rew -= (min(dists) / (self.world.map.SIZE_X * self.num_agents))
            if all(agent.state.p_pos == l.state.p_pos):
                rew += 1.0
                self.world.landmarks.pop(l_idx)
                pos_x, pos_y = self.world.map.coord2ind(l.state.p_pos)
                self.world.map.matrix[pos_x, pos_y, 2] = 0
                self.events_completed += 1
                self.generate_events(1)

        if agent.collide:
            for a in self.world.agents:
                if agent == a: continue
                if is_collision(a, agent):
                    rew -= 1.0
                    self.agents_collided += 1
            # 壁に衝突したら負の報酬
            if agent.collide_walls:
                rew = 1.0
                agent.collide_walls = False
                self.walls_collided += 1
        return rew

    def __observation(self, agent):
        # 3 x 7 x 7の入力が欲しい
        #pos_x, pos_y = self.world.map.coord2ind(agent.state.p_pos)
        # 0:agents, 1:landmarks, 2:invisible area
        obs = np.zeros((3, 7, 7), dtype=np.int8)

        # エージェントの入力
        for a in self.world.agents:
            if abs(a.state.p_pos[0]-agent.state.p_pos[0]) > 3 or abs(a.state.p_pos[1]-agent.state.p_pos[1]) > 3:
                continue
            pos_x, pos_y = self.world.map.coord2ind((a.state.p_pos[0]-agent.state.p_pos[0], a.state.p_pos[1]-agent.state.p_pos[1]),
                                                    size_x=7, size_y=7)
            obs[1, pos_x, pos_y] = 1

        # イベントの入力
        for landmark in self.world.landmarks:
            if abs(landmark.state.p_pos[0]-agent.state.p_pos[0]) > 3 or abs(landmark.state.p_pos[1]-agent.state.p_pos[1]) > 3:
                continue
            pos_x, pos_y = self.world.map.coord2ind((landmark.state.p_pos[0]-agent.state.p_pos[0], landmark.state.p_pos[1]-agent.state.p_pos[1]),
                                                    size_x=7, size_y=7)
            obs[1, pos_x, pos_y] = 1

        # 壁と見えないセルの入力
        for x in range(7):
            for y in range(7):
                pos_x, pos_y = self.world.map.ind2coord((x, y), size_x=7, size_y=7)
                pos_x, pos_y = self.world.map.coord2ind((pos_x+agent.state.p_pos[0], pos_y+agent.state.p_pos[1]))
                if pos_x < 0 or self.world.map.SIZE_X <= pos_x or pos_y < 0 or self.world.map.SIZE_Y <= pos_y:
                    obs[2, x, y] = -1
                    continue

                if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                    obs[2, x, y] = 1

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
            if action == 1: agent.action.u[0] = 1.0
            elif action == 2: agent.action.u[1] = 1.0
            elif action == 3: agent.action.u[0] = -1.0
            elif action == 4: agent.action.u[1] = -1.0

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
        world.map = Exp4_Map(config)

        return world

    def describe_env(self):
        print("""
    Experiment 4 Environment generated!

    ======================Action======================
    | 0: Stay | 1: Right | 2: Up | 3: Left | 4: Down |
    ===================State(agent)===================
    | obs1 : [agent.state.p_pos]                     |
    | obs2 : entity_pos                              |
    | obs3: other_pos                                |
    | np.concatenate(ob1 + obs2 + obs3)              |
    ======================Reward======================
    | rew1 : -min(dist) / SIZE_X                     |
    | rew2 : +1 if success                           |
    | rew3 : -1 if is_collision                      |
    | rew1 + rew2 + rew3                             |
    ==================================================
    """)


class Exp4_Map(Map):
    def __init__(self, config):
        super(Exp4_Map, self).__init__()
        self.SIZE_X = 24
        self.SIZE_Y = 24
        # 0:walls, 1:agents, 2:landmarks
        self.matrix = np.zeros((self.SIZE_X, self.SIZE_Y, 3), dtype=np.int8)
        self.matrix_probs = np.zeros((self.SIZE_X, self.SIZE_Y), dtype=np.float)
        self.agents_pos = dict()
        self.landmarks_pos = dict()
        self.locate_walls()
        self.set_probs()

    def reset(self, agents, landmarks):
        for agent in agents:
            self.agents_pos[agent.name] = agent.state.p_pos
        for landmark in landmarks:
            self.landmarks_pos[landmark.name] = landmark.state.p_pos

    def step(self, agents):
        for agent in agents:
            self.agents_pos[agent.name] = agent.state.p_pos

    def coord2ind(self, p_pos, size_x=24, size_y=24):
        pos_x, pos_y = p_pos
        pos_x = (size_x // 2) + pos_x
        pos_y = (size_y // 2) - pos_y
        res = np.array([pos_x, pos_y])

        return res

    def ind2coord(self, p_pos, size_x=24, size_y=24):
        pos_x, pos_y = p_pos
        pos_x = pos_x - (size_x // 2)
        pos_y = (size_y // 2) - pos_y
        res = np.array([pos_x, pos_y])

        return res

    def locate_walls(self):
        self.matrix[:, np.array([0, self.SIZE_Y-1]), 0] = 1
        self.matrix[np.array([0, self.SIZE_X-1]), :, 0] = 1
        self.matrix[np.array([1,2,3,7,8,9,10,13,14,15,16,20,21,22]), 10, 0] = 1
        self.matrix[np.array([1,2,3,7,8,9,10,13,14,15,16,20,21,22]), 13, 0] = 1
        self.matrix[10, np.array([1,2,3,7,8,9,10,13,14,15,16,20,21,22]), 0] = 1
        self.matrix[13, np.array([1,2,3,7,8,9,10,13,14,15,16,20,21,22]), 0] = 1

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
        self.matrix_probs + 1e-4
        # 右上
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
                self.matrix_probs[x, y] = 1e-3
