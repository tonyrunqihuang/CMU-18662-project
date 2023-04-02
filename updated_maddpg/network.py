import torch
from torch.optim import Adam
from torch import nn, Tensor
import torch.nn.functional as F
from copy import deepcopy
from typing import List

class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
    

class QMixer(nn.Module):
    def __init__(self, state_dim, n_agents, output_dim=1, embed_dim=32, hypernet_embed=64):
        super(QMixer, self).__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.n_agents = n_agents

        self.hyper_w1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                      nn.ReLU(),
                                      nn.Linear(hypernet_embed, hypernet_embed),
                                      nn.ReLU(),
                                      nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        self.hyper_w2 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                      nn.ReLU(),
                                      nn.Linear(hypernet_embed, hypernet_embed),
                                      nn.ReLU(),
                                      nn.Linear(hypernet_embed, self.embed_dim))

        self.hyper_b1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                      nn.ReLU(),
                                      nn.Linear(hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                      nn.ReLU(),
                                      nn.Linear(hypernet_embed, output_dim))

    def forward(self, agent_qs, states):
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        w1 = self.hyper_w1(states).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)

        hidden_1 = F.relu(torch.bmm(agent_qs, w1) + b1)

        w2 = self.hyper_w2(states).view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        y = torch.bmm(hidden_1, w2) + b2
        q_tot = y.view(-1, 1)

        return q_tot