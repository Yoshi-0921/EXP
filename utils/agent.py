# -*- coding: utf-8 -*-

from random import random
from typing import Tuple

import numpy as np
import torch
from torch import nn

from utils.buffer import Experience


class Agent:
    """
    Base Agent class handeling the interaction with the environment
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        raise NotImplementedError

    def random_action(self) -> int:
        """
        Returns:
            random action
        """
        action = int(random()*4)

        return action

    def get_action(self, state, epsilon):
        raise NotImplementedError