import numpy as np

def reward_to_go(rewards):
    n = len(rewards)
    rtgs = np.zeros_like(rewards, dtype=np.float32)
    for i in reversed(range(n)):
        rtgs[i] = rewards[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def compute_actor_loss(batch_log_ps, batch_advs):
    return - (batch_log_ps * batch_advs).mean()

def compute_critic_loss(v_net, obs, batch_rews):
    return ((v_net(obs) - batch_rews) ** 2).mean()
