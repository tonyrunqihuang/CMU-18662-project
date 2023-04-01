import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable


def get_args():

    parser = argparse.ArgumentParser("Multiagent RL with OpenAI's MPE")

    parser.add_argument("--scenario_name", type=str, default="simple_tag", help="name of the scenario",
                        choices=['simple_adversary', 'simple_crypto', 'simple_push',
                        'simple_reference', 'simple_speaker_listener', 'simple_spread',
                        'simple_tag', 'simple_world_comm', 'simple'])
    parser.add_argument("--adv_algo", type=str, default='qmix', help="adversary algorithm",
                        choices=['qmix', 'maddpg'])
    parser.add_argument("--agent_algo", type=str, default='maddpg', help="agent algorithm",
                        choices=['qmix', 'maddpg'])
    parser.add_argument("--n_episodes", type=int, default=25000, help="number of training episodes")
    parser.add_argument("--episode_length", type=int, default=25, help="length of each episode")
    parser.add_argument("--noise_rate", type=float, default=0.5, help="noise for action")
    parser.add_argument("--min_noise_rate", type=float, default=0.05, help="minimum noise for action")
    parser.add_argument("--epsilon", type=float, default=0.5, help="epsilon for epsilon greddy")
    parser.add_argument("--min_epsilon", type=float, default=0.05, help="minimum epsilon for epsilon greddy")
    parser.add_argument("--anneal_episodes", type=int, default=10000, help="number of episodes to anneal")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="number of transitions stored in buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of transitions to optimize")
    parser.add_argument("--evaluate_rate", type=int, default=1000, help="how often to evaluate model")
    parser.add_argument("--save_rate", type=int, default=5000, help="how many episodes to save model")
    parser.add_argument("--evaluate_episodes", type=int, default=10000, help="number of episodes to evaluate")
    parser.add_argument("--model_dir", type=str, default="./model", help="directory to load model")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--load_model", type=bool, default=False, help="whether to load pretrained model")
    parser.add_argument("--seed", type=int, default=int(10), help="random seed for experiment")

    args = parser.parse_args()

    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(args, discrete_action=False, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    scenario = scenarios.load(args.scenario_name + '.py').Scenario()
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    return env


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def onehot_from_logits(logits, eps=0.0):

    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()

    if eps == 0.0:
        return argmax_acs

    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
                range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in enumerate(torch.rand(logits.shape[0]))])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):

    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)

    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):

    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))

    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0, hard=False):

    y = gumbel_softmax_sample(logits, temperature)

    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
        
    return y
