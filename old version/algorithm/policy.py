import os
import torch
from copy import deepcopy
from gym.spaces import Box, Discrete
from utils.agent import Agent
from utils.mixer import Mixer
from utils.misc import soft_update, hard_update, onehot_from_logits, gumbel_softmax

MSELoss = torch.nn.MSELoss()

class Policy:
    def __init__(self, args, agent_algo, team_algo, team_types, agent_init_params, mixer_init_params, gamma=0.95, tau=0.01, discrete_action=False):
        self.agents = [Agent(discrete_action=discrete_action, **param) for param in agent_init_params]
        self.mixers = [Mixer(agents=self.agents, **param) for param in mixer_init_params]
        self.agent_algo = agent_algo
        self.team_algo = team_algo
        self.teams = team_types
        self.n_agents = len(self.agents)
        self.n_teams = len(self.teams)
        self.args = args
        self.gamma = gamma
        self.tau = tau
        self.discrete_action = discrete_action

        self.model_path = os.path.join(self.args.model_dir, self.args.scenario_name)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        for i, a in enumerate(self.agents):
            agent_path = os.path.join(self.model_path, a.type, self.agent_algo[i], 'agent_%d' % i)
            if not os.path.exists(agent_path):
                os.makedirs(agent_path)
        for i, m in enumerate(self.mixers):
            mixer_path = os.path.join(self.model_path, m.type, 'qmix', 'mixer')
            if not os.path.exists(mixer_path):
                os.makedirs(mixer_path)
        if self.args.load_model:
            self.load_model()

    @property
    def policies(self):
        return [a.actor for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_actor for a in self.agents]

    def step(self, observations, epsilon, noise_rate):
        return [a.step(obs, epsilon=epsilon, noise_rate=noise_rate) for a, obs in zip(self.agents, observations)]

    def qmix_update(self, batch, mixer_i):
        curr_team = self.mixers[mixer_i]
        curr_type = curr_team.type

        o, u, r, o_next = [], [], [], []
        for i, a in enumerate(self.agents):
            o.append(batch['o_%d' % i])
            o_next.append(batch['o_next_%d' % i])
            u.append(batch['u_%d' % i])
            if a.type == curr_type:
                r.append(batch['r_%d' % i])
        r_tot = torch.cat(r, dim=1).sum(dim=1, keepdim=True)

        # ----- critic + mixer update -----
        if self.discrete_action:
            u_next = [onehot_from_logits(pi_target(obs_next)) for pi_target, obs_next in zip(self.target_policies, o_next)]
        else:
            u_next = [pi_target(obs_next) for pi_target, obs_next in zip(self.target_policies, o_next)]
        next_state = torch.cat((*o_next, *u_next), dim=1)
        curr_state = torch.cat((*o, *u), dim=1)

        qs, qs_next = [], []
        for i, a in enumerate(self.agents):
            if a.type == curr_type:
                qs.append(a.critic(curr_state))
                qs_next.append(a.target_critic(next_state))
        qs = torch.cat(qs, dim=1)
        qs_next = torch.cat(qs_next, dim=1)

        curr_q_tot = curr_team.mixer(qs, curr_state)
        next_q_tot = curr_team.target_mixer(qs_next, next_state)
        target_q_tot = r_tot + self.gamma * next_q_tot

        curr_team.critic_mixer_optim.zero_grad()
        critic_loss = MSELoss(curr_q_tot, target_q_tot.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_team.critic_mixer_param, 5)
        curr_team.critic_mixer_optim.step()

        # ----- actor update -----
        all_actions, agent_qs = [], []
        for i, a in enumerate(self.agents):
            all_actions.append(a.actor(o[i]))
        curr_state = torch.cat((*o, *all_actions), dim=1)

        for i, a in enumerate(self.agents):
            if a.type == curr_type:
                agent_qs.append(a.critic(curr_state))
        agent_qs = torch.cat(agent_qs, dim=1)

        curr_team.actor_optim.zero_grad()
        q_tot = -curr_team.mixer(agent_qs, curr_state).mean()
        q_tot.backward()
        torch.nn.utils.clip_grad_norm_(curr_team.actor_param, 0.1)
        curr_team.actor_optim.step()


    def maddpg_update(self, batch, agent_i):
        curr_agent = self.agents[agent_i]
        r = batch['r_%d' % agent_i]
        o, u, o_next = [], [], []
        for i in range(self.n_agents):
            o.append(batch['o_%d' % i])
            u.append(batch['u_%d' % i])
            o_next.append(batch['o_next_%d' % i])

        # -----critic update------
        if self.discrete_action:
            u_next = [onehot_from_logits(pi(obs_next)) for pi, obs_next in zip(self.target_policies, o_next)]
        else:
            u_next = [pi(obs_next) for pi, obs_next in zip(self.target_policies, o_next)]
        next_state = torch.cat((*o_next, *u_next), dim=1)
        curr_state = torch.cat((*o, *u), dim=1)

        curr_q = curr_agent.critic(curr_state)
        next_q = curr_agent.target_critic(next_state)
        target_q = r + self.gamma * next_q
        critic_loss = MSELoss(curr_q, target_q.detach())

        curr_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        # -----policy update-----
        if self.discrete_action:
            curr_act = curr_agent.actor(o[agent_i])
            agent_action = gumbel_softmax(curr_act, hard=True)
        else:
            curr_act = curr_agent.actor(o[agent_i])
            agent_action = curr_act

        all_actions = []
        for i, pi, obs in zip(range(self.n_agents), self.policies, o):
            if i == agent_i:
                all_actions.append(agent_action)
            else:
                all_actions.append(u[i])
        critic_input = torch.cat((*o, *all_actions), dim=1)

        curr_agent.actor_optimizer.zero_grad()
        actor_loss = -curr_agent.critic(critic_input).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.actor.parameters(), 0.5)
        curr_agent.actor_optimizer.step()


    def soft_update_all_target_networks(self):
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_actor, a.actor, self.tau)
        for m in self.mixers:
            soft_update(m.target_mixer, m.mixer, self.tau)


    @classmethod
    def init_from_env(cls, args, env):
        if all([hasattr(agent, 'adversary') for agent in env.agents]):
            agent_types = ['adversary' if agent.adversary else 'agent' for agent in env.agents]
        else:
            agent_types = ['agent' for agent in env.agents]

        team_types = list(dict.fromkeys(agent_types).keys())
        team_algo = [args.adv_algo if atype == 'adversary' else args.agent_algo for atype in team_types]
        agent_algo = [args.adv_algo if atype == 'adversary' else args.agent_algo for atype in agent_types]

        agent_init_params = []
        for type, acsp, obsp in zip(agent_types, env.action_space, env.observation_space):
            actor_in_dim = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:
                discrete_action = True
                get_shape = lambda x: x.n
            actor_out_dim = get_shape(acsp)

            critic_in_dim = 0
            for o_dim, a_dim in zip(env.observation_space, env.action_space):
                critic_in_dim += o_dim.shape[0]
                critic_in_dim += get_shape(a_dim)

            agent_init_params.append({'type': type,
                                      'actor_in_dim': actor_in_dim,
                                      'actor_out_dim': actor_out_dim,
                                      'critic_in_dim': critic_in_dim})

        mixer_init_params = []
        for i in range(len(team_types)):
            if team_algo[i] == 'qmix':
                n_agents = 0
                state_dim = 0
                for type, obsp, acsp in zip(agent_types, env.observation_space, env.action_space):
                    if type == team_types[i]:
                        n_agents += 1
                    state_dim += obsp.shape[0]
                    state_dim += get_shape(acsp)
                mixer_init_params.append({'type': team_types[i],
                                          'n_agents': n_agents,
                                          'mixer_state_dim': state_dim})

        init_dict = {'args': args, 'agent_algo': agent_algo, 'team_algo': team_algo, 'team_types': team_types,
                     'agent_init_params': agent_init_params, 'mixer_init_params': mixer_init_params,
                     'discrete_action': discrete_action}
        print(init_dict)
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance


    def save_model(self):
        for i, a in enumerate(self.agents):
            agent_path = os.path.join(self.model_path, a.type, self.agent_algo[i], 'agent_%d' % i)
            if not os.path.exists(agent_path):
                os.makedirs(agent_path)
            torch.save(a.actor.state_dict(), agent_path + '/' + 'actor_params.pkl')
            torch.save(a.critic.state_dict(), agent_path + '/' + 'critic_params.pkl')
        for i, m in enumerate(self.mixers):
            mixer_path = os.path.join(self.model_path, m.type, 'qmix', 'mixer')
            if not os.path.exists(mixer_path):
                os.makedirs(mixer_path)
            torch.save(m.mixer.state_dict(), mixer_path + '/' + 'qmix_params.pkl')


    def load_model(self):
        for i, a in enumerate(self.agents):
            actor_path = os.path.join(self.model_path, a.type, self.agent_algo[i], 'agent_%d' % i, 'actor_params.pkl')
            if os.path.exists(actor_path):
                a.actor.load_state_dict(torch.load(actor_path))
                a.target_actor.load_state_dict(torch.load(actor_path))
                print('{} {} successfully loaded actor network: {}'.format(a.type, i, actor_path))
            critic_path = os.path.join(self.model_path, a.type, self.agent_algo[i], 'agent_%d' % i, 'critic_params.pkl')
            if os.path.exists(critic_path):
                a.critic.load_state_dict(torch.load(critic_path))
                a.target_critic.load_state_dict(torch.load(critic_path))
                print('{} {} successfully loaded critic network: {}'.format(a.type, i, critic_path))
        for i, m in enumerate(self.mixers):
            mixer_path = os.path.join(self.model_path, m.type, 'qmix', 'mixer', 'qmix_params.pkl')
            if os.path.exists(mixer_path):
                m.mixer.load_state_dict(torch.load(mixer_path))
                m.target_mixer.load_state_dict(torch.load(mixer_path))
                print('{} team successfully loaded mixer network: {}'.format(m.type, mixer_path))
