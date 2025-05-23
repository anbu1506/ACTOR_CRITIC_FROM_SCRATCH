import torch
from actor_critic.env_wrapper import CartPoleEnvWrapper
from actor_critic.models import ActorCritic

env = CartPoleEnvWrapper()
env.render_mode = "human"
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

ac = ActorCritic(obs_dim, act_dim)
ac.load_state_dict(torch.load("actor_critic_cartpole.pth"))
ac.eval()

obs = env.reset()[0]
while True:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    action_distrib = ac.actor.distribution(obs_tensor)
    action = torch.argmax(action_distrib.probs).item()
    obs, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        break

env.close()
