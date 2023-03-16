import threading
import numpy as np
import torch
from gym.spaces import Box, Discrete


class ReplayBuffer:
    def __init__(self, args, env):
        self.size = args.buffer_size
        self.args = args
        self.env = env
        self.current_size = 0
        self.buffer = {}
        self.n_agents = len(self.env.agents)
        self.observation_shape = [obsp.shape[0] for obsp in env.observation_space]
        self.action_shape = [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space]
        for i in range(self.n_agents):
            self.buffer['o_%d' % i] = np.empty([self.size, self.observation_shape[i]])
            self.buffer['u_%d' % i] = np.empty([self.size, self.action_shape[i]])
            self.buffer['r_%d' % i] = np.empty([self.size, 1])
            self.buffer['o_next_%d' % i] = np.empty([self.size, self.observation_shape[i]])
        self.lock = threading.Lock()


    def store_transition(self, o, o_next, u, r):
        idxs = self._get_storage_idx(inc=1)
        for i in range(self.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]


    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        for key in temp_buffer.keys():
            temp_buffer[key] = torch.tensor(temp_buffer[key], dtype=torch.float32)
        return temp_buffer


    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
