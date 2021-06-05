import random
from collections import deque, namedtuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# Prioritized Experience Replay: https://github.com/rlcode/per/blob/master/prioritized_memory.py

e = 0.0001
a = 0.6
beta = 0.4
beta_increment = 0.0001


class PEReplayBuffer:  # stored as ( s, a, r, s_ ) in SumTree

    def __init__(self,
                 buffer_size,
                 batch_size,
                 device,
                 alpha=a,
                 beta=beta,
                 beta_increment=beta_increment,
                 epsilon=e):
        self.tree = SumTree(buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        # self.experience = namedtuple(
        #     'Transition',
        #     field_names=['state', 'action', 'reward', 'next_state', 'mask', 'error', 'idx']
        # )

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, error, sample):
        p = self._get_priority(error)
        if sample:
            self.tree.add(p, sample)

    def sample(self):
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        states = (
            torch.from_numpy(np.vstack([e[0] for e in batch if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e[1] for e in batch if e is not None])
            )
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e[2] for e in batch if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e[3] for e in batch if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e[4] for e in batch if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones), idxs, is_weight

    def update(self, idxs, errors):
        for i, e in zip(idxs, errors):
            p = self._get_priority(e)
            self.tree.update(i, p)


class OrnsteinUhlenbeckProcess:
    def __init__(self, size, seed, mu=0, std=0.2, theta=0.15, dt=1):
        self.theta = theta
        self.mu = mu * np.ones(size)
        self.std = std
        self.dt = dt
        self.size = size
        self.seed = random.seed(seed)
        self.reset()

    def sample(self):
        dx = self.theta * (self.mu - self.x_prev) * self.dt + self.std * np.sqrt(
            self.dt
        ) * np.random.standard_normal(self.size)
        self.x_prev = self.x_prev + dx
        return self.x_prev

    def reset(self):
        self.x_prev = self.mu * np.ones(self.size)


# SumTree: https://github.com/rlcode/per/blob/master/SumTree.py
# A binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
