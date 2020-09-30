# -*- coding: utf-8 -*-

import numpy as np
from utils.core import Env, World, Agent, Landmark, Map

class Exp1_Env(Env):
    def __init__(self):
        super(Exp1_Env, self).__init__()
        self.describe_env()
        self.world = self.make_world()
        self.agents = self.world.agents
        self.num_agents = len(self.world.agents)
        self.shared_reward = True
        self.action_space, self.observation_space = list(), list()
        self.reset()

        for agent in self.agents:
            self.action_space.append(5)
            self.observation_space.append(self.__observation(agent).shape[0])

    def reset(self):
        # agentのposとvelの初期化
        for agent in self.world.agents:
            agent.state.p_pos = np.random.randint(-5, 5, self.world.dim_p)
            agent.collide_walls = False
        # landmarkのposとvelの初期化
        for landmark in self.world.landmarks:
            landmark.state.p_pos = np.random.randint(-5, 5, self.world.dim_p)

        self.world.map.reset(agents=self.world.agents, landmarks=self.world.landmarks)

        obs_n = list()
        for agent in self.agents:
            obs_n.append(self.__observation(agent))
        return obs_n

    def step(self, action_n):
        obs_n, reward_n, done_n = list(), list(), list()
        for i, agent in enumerate(self.agents):
            self.__action(action_n[i], agent)
        self.world.step()
        self.world.map.step(self.agents)
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self.__observation(agent))
            reward_n.append(self.__reward(agent))
            done_n.append(self.__done(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.num_agents

        return obs_n, reward_n, done_n

    def __reward(self, agent):
        def is_collision(agent1, agent2):
            delta_pos = agent1.state.p_pos - agent2.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = agent1.size + agent2.size
            return True if dist < dist_min else False

        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in self.world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in self.world.agents]
            rew -=min(dists) / self.world.map.SIZE_X
        if agent.collide:
            for a in self.world.agents:
                if agent == a: continue
                if is_collision(a, agent):
                    rew -= 1
                if agent.collide_walls:
                    rew -= 1
        return rew

    def __observation(self, agent):
        landmark_pos, other_pos = list(), list()
        # landmarkと他agentとの相対的な距離
        for landmark in self.world.landmarks:
            landmark_pos.append(landmark.state.p_pos - agent.state.p_pos)
        for other in self.world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_pos] + landmark_pos + other_pos)

    def __done(self, agent):
        for landmark in self.world.landmarks:
            if all(agent.state.p_pos == landmark.state.p_pos):
                return True
        # 壁にぶつかったらリセット
        if agent.collide_walls:
            return True
        return False

    def __action(self, action, agent):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            agent.action.u = np.zeros(self.world.dim_p)
            if action == 1: agent.action.u[0] = 1.0
            elif action == 2: agent.action.u[1] = 1.0
            elif action == 3: agent.action.u[0] = -1.0
            elif action == 4: agent.action.u[1] = -1.0

    def make_world(self):
        world = World()
        num_agents = 1
        num_landmarks = 1
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent {i}'
            agent.collide = True
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'landmark {i}'
            landmark.collide = False
        world.map = Exp1_Map()

        return world

    def describe_env(self):
        print("""
    Experiment 1 Environment generated!

    ======================Action======================
    | 0: Stay | 1: Right | 2: Up | 3: Left | 4: Down |
    ===================State(agent)===================
    | obs1 : [agent.state.p_pos]                     |
    | obs2 : entity_pos                              |
    | obs3: other_pos                                |
    | np.concatenate(ob1 + obs2 + obs3)              |
    ======================Reward======================
    | rew1 : -min(dist)                              |
    | rew2 : -1 if is_collision                      |
    | rew1 + rew2                                    |
    ==================================================""")


class Exp1_Map(Map):
    def __init__(self):
        super(Exp1_Map, self).__init__()
        self.SIZE_X = 13
        self.SIZE_Y = 13
        # 0:walls, 1:agents, 2:landmarks
        self.matrix = np.zeros((self.SIZE_X, self.SIZE_Y, 3), dtype=np.int8)
        self.agents_pos = dict()
        self.landmarks_pos = dict()
        self.locate_walls()

    def reset(self, agents, landmarks):
        for agent in agents:
            self.agents_pos[agent.name] = agent.state.p_pos
        for landmark in landmarks:
            self.landmarks_pos[landmark.name] = landmark.state.p_pos

    def step(self, agents):
        for agent in agents:
            self.agents_pos[agent.name] = agent.state.p_pos

    def coord2ind(self, p_pos):
        pos_x, pos_y = p_pos
        pos_x = (self.SIZE_X // 2) + pos_x
        pos_y = (self.SIZE_Y // 2) - pos_y
        res = np.array([pos_x, pos_y])

        return res

    def locate_walls(self):
        #internal_x_walls = np.array([0, 11, 14, 25, 28, self.SIZE_Y-1])
        #internal_y_walls = np.array([0, 11, 14, 25, 28, self.SIZE_X-1])
        #self.matrix[:, internal_x_walls, 0] = 1
        #self.matrix[internal_y_walls, :, 0] = 1
        self.matrix[:, np.array([0, self.SIZE_Y-1]), 0] = 1
        self.matrix[np.array([0, self.SIZE_X-1]), :, 0] = 1

    def locate_agents(self):
        for agent in self.agents:
            pos_x, pos_y = self.coord2ind(agent.state.p_pos)
            self.matrix[pos_x, pos_y, 1] = 1

    def locate_landmarks(self):
        for landmark in self.landmarks:
            pos_x, pos_y = self.coord2ind(landmark.state.p_pos)
            self.matrix[pos_x, pos_y, 2] = 1
