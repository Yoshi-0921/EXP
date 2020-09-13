# -*- coding: utf-8 -*-

import numpy as np
from utils.core import Env, World, Agent, Landmark, Map

class Exp1_Env(Env):
    def __init__(self):
        self.world = self.make_world()
        self.agents = self.world.agents
        self.n = len(self.world.agents)
        self.shared_reward = True
        #self.action_space, observation_space = list(), list()
        self.describe_env()

    def reset(self):
        # agentのposとvelの初期化
        for agent in self.world.agents:
            agent.state.p_pos = np.random.uniform(-1, 1, self.world.dim_p)
            agent.state.p_vel = np.zeros(self.world.dim_p)
        # landmarkのposとvelの初期化
        for landmark in self.world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, 1, self.world.dim_p)
            landmark.state.p_vel = np.zeros(self.world.dim_p)

        obs_n = list()
        for agent in self.agents:
            obs_n.append(self.__observation(agent))
        return obs_n

    def step(self, action_n):
        obs_n, reward_n, done_n = list(), list(), list()
        for i, agent in enumerate(self.agents):
            self.__action(action_n[i], agent)
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self.__observation(agent))
            reward_n.append(self.__reward(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n

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
            rew -= min(dists)
        if agent.collide:
            for a in self.world.agents:
                if is_collision(a, agent):
                    rew -= 1
        return rew

    def __observation(self, agent):
        entity_pos, other_pos = list(), list()
        # landmarkと他agentとの相対的な距離
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        for other in self.world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_pos] + entity_pos + other_pos)

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
        num_agents = 3
        num_landmarks = 3
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent {i}'
            agent.collide = True
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'landmark {i}'
            landmark.collide = False
        return world

    def describe_env(self):
        print("""
    Experiment 1 Environment generated!

    =================Action=================
    | 1: Right | 2: Up | 3: Left | 4: Down |
    ========================================""")