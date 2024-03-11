import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, min_log_std=-6, max_log_std=0, min_action=-1, max_action=1):
        super(Actor, self).__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std
        self._min_action = min_action
        self._max_action = max_action

    def _get_policy(self, state):
        mean = self._mlp(state)
        log_std = torch.sigmoid(self._log_std)
        log_std = self._min_log_std + log_std * (self._max_log_std - self._min_log_std)
        # log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    def log_prob(self, state, action):
        policy = self._get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return log_prob

    def forward(self, state):
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(min=self._min_action, max=self._max_action)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state, device, train=True):
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if train:
            action_t = policy.sample()
        else:
            action_t = policy.mean
        action_t.clamp_(min=self._min_action, max=self._max_action)
        action = action_t[0].detach().cpu().numpy()
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        self._mlp1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self._mlp2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        q1_value = self._mlp1(torch.cat([state, action], dim=1))
        q2_value = self._mlp2(torch.cat([state, action], dim=1))
        return q1_value, q2_value


class AWAC(object):
    def __init__(self, state_dim, action_dim, max_action, hidden_size=256, learning_rate=3e-4, gamma=0.99, tau=5e-3,
                 awac_lambda=2., exp_adv_max=100., device='cpu'):
        self._actor = Actor(
            state_dim, action_dim, hidden_size, min_action=-max_action, max_action=max_action).to(device)
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=learning_rate)

        self._critic = Critic(state_dim, action_dim, hidden_size).to(device)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=learning_rate)
        self._critic_target = copy.deepcopy(self._critic)

        self._device = device
        self._gamma = gamma
        self._tau = tau
        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_max

    def _actor_loss(self, states, actions):
        pi_actions, _ = self._actor(states)
        v = torch.min(*self._critic(states, pi_actions)).detach()
        q = torch.min(*self._critic(states, actions)).detach()
        adv = q - v
        weights = F.softmax(adv / self._awac_lambda, dim=0)
        policy_logpp = self._actor.log_prob(states, actions)
        loss = (-policy_logpp * len(weights) * weights.detach()).mean()
        return loss

    def _critic_loss(self, states, actions, next_states, rewards, dones):
        next_actions, _ = self._actor(next_states)
        q_next = torch.min(*self._critic_target(next_states, next_actions)).detach()
        q_target = rewards + self._gamma * (1 - dones) * q_next
        q1, q2 = self._critic(states, actions)
        return F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

    def train(self, replay_buffer, batch_size):
        states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)

        critic_loss = self._critic_loss(states, actions, next_states, rewards, dones)
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        actor_loss = self._actor_loss(states, actions)
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()
        for param, target_param in zip(self._critic.parameters(), self._critic_target.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

    def predict(self, state, train=True):
        return self._actor.act(state, self._device, train)

    def save(self, path, episode):
        torch.save(self._critic.state_dict(), "{}/awac_critic_{}".format(path, episode))
        torch.save(self._critic_optimizer.state_dict(), "{}/awac_critic_optimizer_{}".format(path, episode))
        torch.save(self._critic_target.state_dict(), "{}/awac_critic_target_{}".format(path, episode))

        torch.save(self._actor.state_dict(), "{}/awac_actor_{}".format(path, episode))
        torch.save(self._actor_optimizer.state_dict(), "{}/awac_actor_optimizer_{}".format(path, episode))

    def load(self, path, episode):
        self._critic.load_state_dict(torch.load("{}/awac_critic_{}".format(path, episode)))
        self._critic_optimizer.load_state_dict(torch.load("{}/awac_critic_optimizer_{}".format(path, episode)))
        self._critic_target.load_state_dict(torch.load("{}/awac_critic_target_{}".format(path, episode)))

        self._actor.load_state_dict(torch.load("{}/awac_actor_{}".format(path, episode)))
        self._actor_optimizer.load_state_dict(torch.load("{}/awac_actor_optimizer_{}".format(path, episode)))