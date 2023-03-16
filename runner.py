from maddpg import MADDPG


def train(args, env, dim_info):
    
    maddpg =  MADDPG(dim_info, args.buffer_capacity, args.device)
    obs = env.reset()
    step = 0

    for episode in range(args.episode_num):
        obs = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}

        while env.agents:
            step += 1
            if step < args.random_steps:
                action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
                action = maddpg.select_action(obs)

            next_obs, reward, done, truncations, info = env.step(action)
            maddpg.add(obs, action, reward, next_obs, done)

            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                maddpg.learn(args.batch_size, args.gamma)
                maddpg.update_target(args.tau)

            obs = next_obs