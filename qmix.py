import numpy as np
import torch
import torch.nn.functional as F
from agent import Agent
from buffer import Buffer
from networks import QMixer

class QMIX:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, dim_info, capacity, typ, actor_lr=0.0001, critic_lr=0.01):
                
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())
        self.agents = {}
        self.buffers = {}
        n_agents = 0
        
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            n_agents+=1
            if 'adversary' in agent_id:
                typ='adversary'
            else:
                typ='agent'
            self.agents[agent_id] = Agent(obs_dim, act_dim, global_obs_act_dim, device,typ=typ)
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, device)
        
        self.typ = typ
        self.mixer = QMixer(state_dim=global_obs_act_dim, n_agents=n_agents)
        self.target_mixer = deepcopy(self.mixer)

        self.critic_mixer_param = list(self.mixer.parameters())
        self.actor_param = list()
        for i, a in enumerate(agents):
            if a.typ == self.typ:
                self.critic_mixer_param += list(a.critic.parameters())
                self.actor_param += list(a.actor.parameters())
        self.actor_optim = torch.optim.Adam(self.actor_param, lr=0.0001)
        self.critic_mixer_optim = torch.optim.Adam(self.critic_mixer_param, lr=0.001)