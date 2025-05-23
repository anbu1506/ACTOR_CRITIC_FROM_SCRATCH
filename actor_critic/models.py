import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )
    
    def distribution(self, obs):
        return torch.distributions.Categorical(logits=self.policy_net(obs))
    
    def forward(self, obs):
        pi = self.distribution(obs)
        a = pi.sample()
        return a

    def log_prob(self, pi, act):
        return pi.log_prob(act)

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_size=64):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, obs):
        return torch.squeeze(self.value_net(obs), -1)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)
    
    def step(self, obs):
        with torch.no_grad():
            a = self.actor(obs)
            v = self.critic(obs)
        return a.numpy(), v.numpy()
