import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64,
                 constrain_out=False, discrete_action=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        if constrain_out and not discrete_action:
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh
        else:
            self.out_fn = lambda x: x

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.out_fn(self.fc3(x))
        return out


class QMixer(nn.Module):
    def __init__(self, state_dim, n_agents, output_dim=1,
                 embed_dim=32, hypernet_embed=64):
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
