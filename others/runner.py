import numpy as np
from maddpg import MADDPG
import os

def train(args, env, dim_info):
    # create folder to save result
    env_dir = os.path.join('./results')
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    maddpg =  MADDPG(dim_info, args.buffer_capacity, args.device)
    obs = env.reset()
    step = 0
    episode_rewards = {agent_id: np.zeros(args.n_episodes) for agent_id in env.agents}

    for episode in range(args.n_episodes):
        obs = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}

        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
            next_obs, reward, done, _, _ = env.step(actions)

            break

        break

    #     while env.agents:
    #         step += 1
    #         if step < args.random_steps:
    #             action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
    #         else:
    #             action = maddpg.select_action(obs)

    #         next_obs, reward, done, info = env.step(action)
    #         maddpg.add(obs, action, reward, next_obs, done)

    #         for agent_id, r in reward.items():
    #             agent_reward[agent_id] += r

    #         if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
    #             maddpg.learn(args.batch_size, args.gamma)
    #             maddpg.update_target(args.tau)

    #         obs = next_obs

    #     for agent_id, r in agent_reward.items():  # record reward
    #         episode_rewards[agent_id][episode] = r
    #     if (episode + 1) % 100 == 0:  # print info every 100 episodes
    #         message = f'episode {episode + 1}, '
    #         sum_reward = 0
    #         for agent_id, r in agent_reward.items():  # record reward
    #             message += f'{agent_id}: {r:>4f}; '
    #             sum_reward += r
    #         message += f'sum reward: {sum_reward}'
    #         print(message)
    # maddpg.save(episode_rewards,result_dir)  # save model