import os
import torch
import numpy as np
from copy import deepcopy
from utils.networks import QMixer


class Mixer:
    def __init__(self, mixer_state_dim, n_agents, agents, type,
                 actor_lr=0.0001, critic_lr=0.01):
        self.type = type
        self.mixer = QMixer(state_dim=mixer_state_dim, n_agents=n_agents)
        self.target_mixer = deepcopy(self.mixer)

        self.critic_mixer_param = list(self.mixer.parameters())
        self.actor_param = list()
        for i, a in enumerate(agents):
            if a.type == self.type:
                self.critic_mixer_param += list(a.critic.parameters())
                self.actor_param += list(a.actor.parameters())
        self.actor_optim = torch.optim.Adam(self.actor_param, lr=0.0001)
        self.critic_mixer_optim = torch.optim.Adam(self.critic_mixer_param, lr=0.001)
