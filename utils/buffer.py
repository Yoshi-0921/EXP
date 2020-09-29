# -*- coding: utf-8 -*-

import numpy as np
import collections
from typing import Tuple

# Named tuple for storing experience steps gathered in training
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])

class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        global_states, global_actions, global_rewards, global_dones, global_next_states = zip(*[self.buffer[idx] for idx in indices])

        global_states = np.array(global_states).transpose(1, 0, 2)
        global_actions = np.array(global_actions).transpose(1, 0)
        global_rewards = np.array(global_rewards, dtype=np.float32).transpose(1, 0)
        global_dones = np.array(global_dones, dtype=np.bool).transpose(1, 0)
        global_next_states = np.array(global_next_states).transpose(1, 0, 2)

        return global_states, global_actions, global_rewards, global_dones, global_next_states