import os
import torch
import numpy as np
from torch.autograd import Variable
from copy import deepcopy
from utils.networks import MLP
from utils.misc import gumbel_softmax, onehot_from_logits

class Agent:
    def __init__(self, actor_in_dim, actor_out_dim, critic_in_dim,
                 type, lr=0.0003, hidden_dim=64, discrete_action=True):

        self.actor = MLP(input_dim=actor_in_dim, output_dim=actor_out_dim,
                         constrain_out=True, discrete_action=discrete_action)
        self.critic = MLP(input_dim=critic_in_dim, output_dim=1,
                          constrain_out=False)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.type = type
        self.action_shape = actor_out_dim
        self.discrete_action = discrete_action


    def step(self, obs, epsilon, noise_rate):
        action = self.actor(obs)
        if self.discrete_action:
            if np.random.uniform() < epsilon: #explore
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:
            if np.random.uniform() < epsilon: #explore
                action = -2 * torch.rand((1, self.action_shape)) + 1
            else:
                noise = noise_rate * torch.rand((1, self.action_shape))
                action += noise
            action = action.clamp(-1, 1)
        return action
