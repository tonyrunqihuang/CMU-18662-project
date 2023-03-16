import os
import torch
import numpy as np
from typing import List
from torch import nn, Tensor
from torch.autograd import Variable
from copy import deepcopy
from networks import MLP


class Agent:
    def __init__(self, actor_in_dim, actor_out_dim, critic_in_dim, lr=0.0003, hidden_dim=64):

        self.actor = MLP(input_dim=actor_in_dim, output_dim=actor_out_dim)
        self.critic = MLP(input_dim=critic_in_dim, output_dim=1)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.action_shape = actor_out_dim

    def action(self, obs):
        return self.actor(obs)

    def target_action(self, obs):
        return self.target_actor(obs)

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()