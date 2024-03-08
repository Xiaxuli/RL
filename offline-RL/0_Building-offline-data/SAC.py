import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


# Actor网络定义
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, layer_size):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, layer_size)
        self.l2 = nn.Linear(layer_size, layer_size)
        self.mean = nn.Linear(layer_size, action_dim)
        self.log_std = nn.Linear(layer_size, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        a_mean = self.mean(x)
        a_log_std = self.log_std(x).clamp(-20, 2)
        return a_mean, a_log_std


# Critic网络定义
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, layer_size):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, layer_size)
        self.l2 = nn.Linear(layer_size, layer_size)
        self.l3 = nn.Linear(layer_size, 1)

        self.l4 = nn.Linear(state_dim + action_dim, layer_size)
        self.l5 = nn.Linear(layer_size, layer_size)
        self.l6 = nn.Linear(layer_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


# SAC算法类
class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, layer_size=256, device='cpu', gamma=0.99, tau=0.005,
                 alpha=0.2, actor_lr=3e-4, critic_lr=3e-4, adaptive_alpha=True):
        """SAC算法

        Args:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            max_action (float): 动作值的最大值
            layer_size (int): 神经网络的隐藏层大小
            device (str): 使用的设备 ('cpu' 或 'cuda')
            gamma (float): 奖励的折扣因子
            tau (float): 在更新 self.target_model 的权重时使用的衰减系数
            alpha (float): 温度参数，决定熵相对于奖励的相对重要性
            actor_lr (float): actor 模型的学习率
            critic_lr (float): critic 模型的学习率
            adaptive_alpha (bool): 是否在Actor中使用自动熵调整训练
        """

        self.actor = Actor(state_dim, action_dim, max_action, layer_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, layer_size).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha

        if self.adaptive_alpha:
            self.target_entropy = torch.tensor(-action_dim, dtype=torch.float, requires_grad=True, device=self.device)
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=actor_lr)

    def sample(self, state):
        a_mean, a_log_std = self.actor(state)
        normal = td.Normal(a_mean, a_log_std.exp())
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = (normal.log_prob(x_t) - torch.log((1. - action.pow(2)) + 1e-6)).sum(1, keepdim=True)

        return action, log_prob

    def predict(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            act_mean, _ = self.actor(state)
            action = torch.tanh(act_mean)
        return action.cpu().numpy()[0]

    def train_predict(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, _ = self.sample(state)
        return action.cpu().detach().numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_pro = self.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            target_Q = torch.min(q1_next, q2_next) - self.alpha * next_log_pro
            target_Q = reward + (1 - done) * self.gamma * target_Q
        cur_q1, cur_q2 = self.critic(state, action)

        critic_loss = F.mse_loss(cur_q1, target_Q) + F.mse_loss(cur_q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        act, log_pi = self.sample(state)
        q1_pi, q2_pi = self.critic(state, act)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path, episode):
        torch.save(self.critic.state_dict(), "{}/sac_critic_{}".format(path, episode))
        torch.save(self.critic_optimizer.state_dict(), "{}/sac_critic_optimizer_{}".format(path, episode))

        torch.save(self.actor.state_dict(), "{}/sac_actor_{}".format(path, episode))
        torch.save(self.actor_optimizer.state_dict(), "{}/sac_actor_optimizer_{}".format(path, episode))

    def load(self, path, episode):
        self.critic.load_state_dict(torch.load("{}/sac_critic_{}".format(path, episode)))
        self.critic_optimizer.load_state_dict(torch.load("{}/sac_critic_optimizer_{}".format(path, episode)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load("{}/sac_actor_{}".format(path, episode)))
        self.actor_optimizer.load_state_dict(torch.load("{}/sac_actor_optimizer_{}".format(path, episode)))
