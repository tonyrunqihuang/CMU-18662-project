import torch
from torch import nn, Tensor
from typing import List
from copy import deepcopy
from network import QMixer

class QMIX:
    def __init__(self, dim_info, agents, actor_lr=0.0001, critic_lr=0.01):
                
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())
        n_agents = 0
        for agent_id in dim_info:
            if 'adversary' in agent_id:
                n_agents += 1

        self.mixer = QMixer(state_dim=global_obs_act_dim, n_agents=n_agents)
        self.target_mixer = deepcopy(self.mixer)
        self.critic_mixer_param = list(self.mixer.parameters())
        self.actor_param = list()
        for agent_id in agents:
            if 'adversary' in agent_id:
                self.critic_mixer_param += list(agents[agent_id].critic.parameters())
                self.actor_param += list(agents[agent_id].actor.parameters())

        self.actor_optimizer = torch.optim.Adam(self.actor_param, lr=0.0001)
        self.critic_mixer_optimizer = torch.optim.Adam(self.critic_mixer_param, lr=0.001)


    def mixer_value(self, qs: List[Tensor], state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.mixer(qs, x).squeeze(1)  # tensor with a given length
    
    def target_mixer_value(self, qs: List[Tensor], state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_mixer(qs, x).squeeze(1)  # tensor with a given length
    
    def update_mixer_critic(self, loss):
        self.critic_mixer_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_mixer_param, 5)
        self.critic_mixer_optimizer.step()

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_param, 0.5)
        self.actor_optimizer.step()