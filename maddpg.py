import numpy as np
import torch
import torch.nn.functional as F
from agent import Agent
from buffer import Buffer
import os
import pickle

class MADDPG:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, dim_info, capacity, device):
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())
        self.agents = {}
        self.buffers = {}
        self.device=device
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            if 'adversary' in agent_id:
                typ='adversary'
            else:
                typ='agent'
            self.agents[agent_id] = Agent(obs_dim, act_dim, global_obs_act_dim,device,typ)
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, device)

        self.dim_info = dim_info

    def select_action(self, obs):
        actions = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float().to(self.device)
            a = self.agents[agent].action(o)
            actions[agent] = a.squeeze(0).detach().cpu().numpy()
        return actions
    
    def add(self, obs, action, reward, next_obs, done):
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].add(o, a, r, next_o, d)

    def sample(self, batch_size):
        total_num = len(self.buffers['agent_0'])
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, d = buffer.sample(indices)
            obs[agent_id] = o
            act[agent_id] = a
            reward[agent_id] = r
            next_obs[agent_id] = n_o
            done[agent_id] = d
            next_act[agent_id] = self.agents[agent_id].target_action(n_o)

        return obs, act, reward, next_obs, done, next_act
    
    def learn(self, batch_size, gamma):
        for agent_id, agent in self.agents.items():
            obs, act, reward, next_obs, done, next_act = self.sample(batch_size)

            # critic update
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))

            next_target_critic_value = agent.target_critic_value(list(next_obs.values()), list(next_act.values()))
            target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id])

            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            # actor update
            action = agent.action(obs[agent_id])
            actor_loss = -agent.critic_value(list(obs.values()), list(act.values())).mean()
            agent.update_actor(actor_loss)

    def soft_update(self, from_network, to_network, tau):
        for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
            to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

    def update_target(self, tau):
        for agent in self.agents.values():
            self.soft_update(agent.actor, agent.target_actor, tau)
            self.soft_update(agent.critic, agent.target_critic, tau)
    def save(self, reward,res_dir):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},  # actor parameter
            os.path.join(res_dir, 'model.pt')
        )
        with open(os.path.join(res_dir, 'rewards.pkl'), 'wb') as f:  # save training data
            pickle.dump({'rewards': reward}, f)
    @classmethod
    def load(cls, dim_info, file):
        """init maddpg using the model saved in `file`"""
        instance = cls(dim_info, 0, 0)
        data = torch.load(file)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance