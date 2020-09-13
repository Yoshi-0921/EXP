# -*- coding: utf-8 -*-

from experiments.exp1.exp1_env import Exp1_Env
from random import random
from pprint import pprint

if __name__ == '__main__':
    env = Exp1_Env()
    state = env.reset()
    for i in range(10):
        action = [int(random()*4)+1]
        state, reward = env.step(action)
        print(f"""
    pos: {[agent.state.p_pos for agent in env.agents]}
    state: {state}
    action: {action}
    reward: {reward}""")

    print('debug')