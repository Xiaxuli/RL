import copy
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action, hidden_size):
        super(Actor, self).__init__()
        self.max_action = max_action

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu((self.fc2(x)))
        mu = torch.tanh(self.mu(x)) * self.max_action
        log_std = self.log_std(x).clamp(-10, 2)
        return mu, log_std

    def evaluate(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        return action, dist

    def get_action(self, state):
        mu, log_std = self.forward(state)
        action = Normal(mu, log_std.exp()).rsample()
        return action.detach().cpu().numpy()

    def get_det_action(self, state):
        mu, _ = self.forward(state)
        return mu.detach().cpu().numpy()


# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action, net_width):
#         super(Actor, self).__init__()
#
#         self.l1 = nn.Linear(state_dim, net_width)
#         self.l2 = nn.Linear(net_width, net_width)
#         self.l3 = nn.Linear(net_width, action_dim)
#
#         self.max_action = max_action
#
#     def forward(self, state):
#         a = F.relu(self.l1(state))
#         a = F.relu(self.l2(a))
#         a = torch.tanh((self.l3(a)))
#         return self.max_action * a
#
#     def get_action(self, state):
#         return self.forward(state).detach().cpu().numpy()


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_size + action_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        self.l4 = nn.Linear(state_size + action_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


class Value(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class IQL(object):
    def __init__(self, state_size, action_size, max_action, learning_rate=3e-4, hidden_size=256, device='cpu',
                 gamma=0.99, tau=5e-3, beta=3.0, quantile=0.7):

        self.device = device
        self.gamma = torch.FloatTensor([gamma]).to(device)
        self.tau = tau
        self.clip_grad_param = 1
        self.beta = torch.FloatTensor([beta]).to(device)
        self.quantile = torch.FloatTensor([quantile]).to(device)
        self.clip_score = torch.FloatTensor([100.0]).to(device)

        self.actor = Actor(state_size, action_size, max_action, hidden_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Critic(state_size, action_size, hidden_size).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.critic_target = copy.deepcopy(self.critic)

        self.value = Value(state_size, hidden_size).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=learning_rate)

    def value_loss(self, states, actions):
        min_q = torch.min(*self.critic_target(states, actions)).detach()
        val_pred = self.value(states)
        val_error = min_q - val_pred
        val_weight = torch.where(val_error > 0, self.quantile, (1 - self.quantile))
        val_loss = (val_weight * (val_error ** 2)).mean()
        return val_loss

    def actor_loss(self, states, actions):
        min_q = torch.min(*self.critic_target(states, actions)).detach()
        val_pred = self.value(states).detach()
        exp_a = torch.exp((min_q - val_pred) * self.beta).clamp(max=self.clip_score)
        _, dist = self.actor.evaluate(states)
        log_probs = dist.log_prob(actions)
        actor_loss = -(exp_a * log_probs).mean()
        return actor_loss

    # def actor_loss(self, states, actions):
    #     min_q = torch.min(*self.critic_target(states, actions)).detach()
    #     val_pred = self.value(states).detach()
    #     exp_a = torch.exp((min_q - val_pred) * self.beta).clamp(max=self.clip_score)
    #     actor_loss = (exp_a * torch.sum((self.actor(states) - actions) ** 2, dim=1)).mean()
    #     return actor_loss

    def critic_loss(self, states, actions, next_states, rewards, dones):
        val_next = self.value(next_states).detach()
        q_target = rewards + self.gamma * (1 - dones) * val_next
        q1, q2 = self.critic(states, actions)
        q1_loss, q2_loss = F.mse_loss(q1, q_target), F.mse_loss(q2, q_target)
        return q1_loss + q2_loss

    def train(self, replay_buffer, batch_size):
        states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)

        value_loss = self.value_loss(states, actions)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        actor_loss = self.actor_loss(states, actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic_loss = self.critic_loss(states, actions, next_states, rewards, dones)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.clip_grad_param)
        self.critic_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def predict(self, state, noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            if noise:
                action = self.actor.get_action(state)
            else:
                action = self.actor.get_det_action(state)
        return action[0]

    def save(self, path, episode):
        torch.save(self.critic.state_dict(), "{}/iql_critic_{}".format(path, episode))
        torch.save(self.critic_optimizer.state_dict(), "{}/iql_critic_optimizer_{}".format(path, episode))
        torch.save(self.critic_target.state_dict(), "{}/iql_critic_target_{}".format(path, episode))

        torch.save(self.actor.state_dict(), "{}/iql_actor_{}".format(path, episode))
        torch.save(self.actor_optimizer.state_dict(), "{}/iql_actor_optimizer_{}".format(path, episode))

        torch.save(self.value.state_dict(), "{}/iql_value_{}".format(path, episode))
        torch.save(self.value_optimizer.state_dict(), "{}/iql_value_optimizer_{}".format(path, episode))

    def load(self, path, episode):
        self.critic.load_state_dict(torch.load("{}/iql_critic_{}".format(path, episode)))
        self.critic_optimizer.load_state_dict(torch.load("{}/iql_critic_optimizer_{}".format(path, episode)))
        self.critic.load_state_dict(torch.load("{}/iql_critic_target_{}".format(path, episode)))

        self.actor.load_state_dict(torch.load("{}/iql_actor_{}".format(path, episode)))
        self.actor_optimizer.load_state_dict(torch.load("{}/iql_actor_optimizer_{}".format(path, episode)))

        self.value.load_state_dict(torch.load("{}/iql_value_{}".format(path, episode)))
        self.value_optimizer.load_state_dict(torch.load("{}/iql_value_optimizer_{}".format(path, episode)))
