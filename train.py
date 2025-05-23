import torch
import numpy as np
from actor_critic.env_wrapper import CartPoleEnvWrapper
from actor_critic.models import ActorCritic
from actor_critic.utils import reward_to_go, compute_actor_loss, compute_critic_loss

env = CartPoleEnvWrapper()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
ac = ActorCritic(obs_dim, act_dim)

pi_optimizer = torch.optim.Adam(ac.actor.parameters(), lr=0.01)
v_optimizer = torch.optim.Adam(ac.critic.parameters(), lr=0.1)

n = 30
for i in range(n):
    env.render_mode = "human" if i == n - 1 else None
    obs = env.reset()[0]

    batch_obs, batch_acts, batch_rtg, batch_advs = [], [], [], []
    epoisode_rets, episode_vals, trajectory_reward = [], [], []
    total_reward = 0

    while True:
        if env.render_mode == "human":
            env.render()

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        a, v = ac.step(obs_tensor)

        batch_obs.append(obs.copy())
        batch_acts.append(a.item())
        episode_vals.append(v.item())
        obs, reward, done, _, _ = env.step(a)
        epoisode_rets.append(reward)
        total_reward += reward

        if done:
            trajectory_reward.append(total_reward)
            advs = reward_to_go(epoisode_rets) - np.array(episode_vals)
            advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
            batch_advs.extend(advs)
            batch_rtg.extend(reward_to_go(epoisode_rets))

            epoisode_rets, episode_vals = [], []
            total_reward = 0
            obs = env.reset()[0]
            if len(batch_obs) > 5000:
                print("Average reward: ", np.mean(trajectory_reward))
                trajectory_reward = []
                break

    batch_acts = torch.as_tensor(batch_acts, dtype=torch.float32)
    batch_obs = torch.as_tensor(batch_obs, dtype=torch.float32)
    batch_rtg = torch.as_tensor(batch_rtg, dtype=torch.float32)
    batch_advs = torch.as_tensor(batch_advs, dtype=torch.float32)

    for _ in range(30):
        critic_loss = compute_critic_loss(ac.critic, batch_obs, batch_rtg)
        v_optimizer.zero_grad()
        critic_loss.backward()
        v_optimizer.step()

    log_ps = ac.actor.distribution(batch_obs).log_prob(batch_acts)
    actor_loss = compute_actor_loss(log_ps, batch_advs)

    pi_optimizer.zero_grad()
    actor_loss.backward()
    pi_optimizer.step()

torch.save(ac.state_dict(), "actor_critic_cartpole.pth")
