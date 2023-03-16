import torch
import random
import argparse
import numpy as np
from pettingzoo.mpe import simple_tag_v2


def get_args():

    parser = argparse.ArgumentParser("CMU 18662 Project - Multiagent Reinforcement Learning")

    # training
    parser.add_argument("--n_episodes", type=int, default=30000, help="number of training episodes")
    parser.add_argument("--episode_length", type=int, default=25, help="length of each episode")
    
    # buffer
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')

    
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')

    # misc
    parser.add_argument("--seed", type=int, default=int(10), help="random seed for experiment")
    parser.add_argument('--device', type=str, default='cpu', help='device to train on')

    args = parser.parse_args()

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(args):

    env = simple_tag_v2.parallel_env(max_cycles=args.episode_length, continuous_actions=True)
    env.reset()

    env_info = {}
    for agent_id in env.agents:
        env_info[agent_id] = []
        env_info[agent_id].append(env.observation_space(agent_id).shape[0])
        env_info[agent_id].append(env.action_space(agent_id).shape[0])

    return env, env_info