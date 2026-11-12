"""
Step 2 of 4 - the experience replay buffer.

Consecutive transitions of one episode are strongly correlated, and
training a network on them in order is like SGD on a non-shuffled,
drifting dataset. The buffer stores the last `buffer_size` transitions
(s, a, r, s', terminated) and hands out uniformly random mini-batches.
This is allowed because Q-learning is off-policy: its target does not
depend on which policy collected the data.
"""
import random
from collections import deque, namedtuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.memory = deque(maxlen=buffer_size)  # oldest entries drop out
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward",
                                       "next_state", "terminated"])
        self.device = device

    def add(self, state, action, reward, next_state, terminated):
        self.memory.append(
            self.experience(state, action, reward, next_state, terminated))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        def stack(field, dtype):
            return torch.from_numpy(
                np.vstack([getattr(e, field) for e in experiences]).astype(dtype)
            ).to(self.device)

        states = stack('state', np.float32)
        actions = stack('action', np.int64)
        rewards = stack('reward', np.float32)
        next_states = stack('next_state', np.float32)
        terminated = stack('terminated', np.float32)
        return states, actions, rewards, next_states, terminated

    def __len__(self):
        return len(self.memory)
