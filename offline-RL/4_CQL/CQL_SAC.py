import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, max_action):
        super(Actor, self).__init__()

        # 定义神经网络层
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        self.max_action = max_action

    def forward(self, state):
        # Actor网络的前向传播
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        mu = self.mu(x)
        log_std = self.log_std(x).clamp(-20, 2)  # 限制log标准差的范围，避免太小或太大
        return mu * self.max_action, log_std

    def evaluate(self, state, epsilon=1e-6):
        # 评估动作，用于计算概率密度和采样
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
        return action, log_prob

    def get_action(self, state):
        # 获取随机采样的动作
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()

    def get_det_action(self, state):
        # 获取确定性的动作（无噪音）
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()

        # 定义两个Q网络
        self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        self.l4 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        # Critic网络的前向传播，计算两个Q值
        x = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


class CQL_SAC(object):
    def __init__(self, state_size, action_size, max_action, hidden_size=256, gamma=0.99, tau=5e-3, device='cpu',
                 actor_lr=3e-4, critic_lr=3e-4, temp=1.0, with_lagrange=False, cql_weight=5.0, target_action_gap=10):
        # 初始化函数，定义了一些超参数和网络结构

        self.state_size = state_size  # 状态空间的维度
        self.action_size = action_size  # 动作空间的维度
        self.max_action = max_action
        self.policy_eval_start = 5000
        self._current_steps = 0
        self.device = device  # 设备（CPU或GPU）
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 软更新的参数
        self.clip_grad_param = 1  # 梯度裁剪的参数
        self.target_entropy = -action_size  # 目标熵，用于自动调整温度参数
        self.log_alpha = torch.tensor(0.0, requires_grad=True)  # log_alpha是用于自动调整温度参数的参数
        self.alpha = self.log_alpha.exp().detach()  # 温度参数
        self.alpha_optimizer = torch.optim.Adam(params=[self.log_alpha], lr=critic_lr)  # 温度参数的优化器

        # CQL参数
        self.num_repeat = 10
        self.with_lagrange = with_lagrange  # 是否使用Lagrange乘子
        self.temp = temp  # 温度参数
        self.cql_weight = cql_weight  # CQL（Conservative Q-Learning）的权重
        self.target_action_gap = target_action_gap  # CQL中的目标动作差异
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)  # CQL温度参数
        self.cql_alpha_optimizer = torch.optim.Adam([self.cql_log_alpha], lr=critic_lr)  # CQL温度参数的优化器

        # Actor网络
        self.actor = Actor(state_size, action_size, hidden_size, max_action).to(device)  # Actor网络
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)  # Actor网络的优化器

        # Critic网络
        self.critic = Critic(state_size, action_size, hidden_size)  # Critic网络
        self.critic_target = copy.deepcopy(self.critic)  # Critic的目标网络，用于软更新
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)  # Critic网络的优化器

    def predict(self, state, noise=True):
        # 预测动作
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            if noise:
                action = self.actor.get_action(state)
            else:
                action = self.actor.get_det_action(state)
        return action.numpy()[0]

    def calc_policy_loss(self, states, alpha):
        # 计算策略损失
        actions_new, log_pis = self.actor.evaluate(states)
        q1, q2 = self.critic_target(states, actions_new)
        min_q = torch.min(q1, q2)
        actor_loss = (alpha * log_pis - min_q).mean()
        return actor_loss, log_pis

    def _compute_policy_value(self, diff_state, now_state):
        # 计算当前策略值
        actions_new, log_pis = self.actor.evaluate(diff_state)
        q1, q2 = self.critic(now_state, actions_new)
        pi_values1 = (q1 - log_pis.detach()).view(self.batch_size, self.num_repeat, 1)
        pi_values2 = (q2 - log_pis.detach()).view(self.batch_size, self.num_repeat, 1)
        return pi_values1, pi_values2

    def _compute_random_value(self, now_state, random_actions):
        # 计算随机策略值
        q1, q2 = self.critic(now_state, random_actions)
        random_log_probs = np.log(0.5 ** self.action_size)
        pi_values1 = (q1 - random_log_probs).view(self.batch_size, self.num_repeat, 1)
        pi_values2 = (q2 - random_log_probs).view(self.batch_size, self.num_repeat, 1)
        return pi_values1, pi_values2

    @staticmethod
    def atanh(x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5 * torch.log(one_plus_x / one_minus_x)

    def _sample_log_prob(self, state, action):
        raw_action = self.atanh(action)
        act_mean, act_log_std = self.actor(state)

        normal = Normal(act_mean, act_log_std.exp())
        log_prob = normal.log_prob(raw_action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        return log_prob.sum(-1)

    def train(self, replay_buffer, random_num=True, batch_size=256):
        self.batch_size = batch_size
        self._current_steps += 1

        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        curr_alpha = copy.deepcopy(self.alpha)
        # ---------------------------------- 更新 Actor --------------------------------- #
        actor_loss, log_pis = self.calc_policy_loss(state, curr_alpha)

        # BC
        if self._current_steps < self.policy_eval_start:
            policy_log_prob = self._sample_log_prob(state, action)
            actor_loss = (self.alpha * log_pis - policy_log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新 alpha 参数
        alpha_loss = -(self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------------- 更新 Critic --------------------------------- #
        with torch.no_grad():
            next_action, next_log_pi = self.actor.evaluate(next_state)
            q1_target_next, q2_target_next = self.critic_target(next_state, next_action)
            q_target_next = torch.min(q1_target_next, q2_target_next) - self.alpha.to(self.device) * next_log_pi
            q_target = reward + (self.gamma * (1 - done) * q_target_next)

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        if random_num:
            # 添加 CQL
            random_action = torch.FloatTensor(batch_size * self.num_repeat, action.shape[1]).uniform_(
                -self.max_action, self.max_action).to(self.device)
            temp_state = state.unsqueeze(1).repeat(1, self.num_repeat, 1).view(
                random_action.shape[0], state.shape[1])
            temp_next_state = next_state.unsqueeze(1).repeat(1, self.num_repeat, 1).view(
                random_action.shape[0], state.shape[1])

            curr_pi_values1, curr_pi_values2 = self._compute_policy_value(temp_state, temp_state)
            next_pi_values1, next_pi_values2 = self._compute_policy_value(temp_next_state, temp_state)
            rand_pi_values1, rand_pi_values2 = self._compute_random_value(temp_state, random_action)

            cat_q1 = torch.cat([rand_pi_values1, curr_pi_values1, next_pi_values1], 1)
            cat_q2 = torch.cat([rand_pi_values2, curr_pi_values2, next_pi_values2], 1)

            cql1_scaled_loss = ((torch.logsumexp(cat_q1 / self.temp,
                                                 dim=1).mean() * self.cql_weight * self.temp) - q1.mean()) * self.cql_weight
            cql2_scaled_loss = ((torch.logsumexp(cat_q2 / self.temp,
                                                 dim=1).mean() * self.cql_weight * self.temp) - q2.mean()) * self.cql_weight

            if self.with_lagrange:
                cql_alpha = self.cql_log_alpha.exp().clamp(0.0, 1e6).to(self.device)[0]
                cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
                cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

                self.cql_alpha_optimizer.zero_grad()
                cql_alpha_loss = (-cql1_scaled_loss - cql2_scaled_loss) * 0.5
                cql_alpha_loss.backward(retain_graph=True)
                self.cql_alpha_optimizer.step()

            critic_loss += cql1_scaled_loss + cql2_scaled_loss
        # CQL 完毕

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.clip_grad_param)
        self.critic_optimizer.step()

        # 更新 Critic 目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path, episode):
        torch.save(self.critic.state_dict(), "{}/CqlSac_critic_{}".format(path, episode))
        torch.save(self.critic_optimizer.state_dict(), "{}/CqlSac_critic_optimizer_{}".format(path, episode))

        torch.save(self.actor.state_dict(), "{}/CqlSac_actor_{}".format(path, episode))
        torch.save(self.actor_optimizer.state_dict(), "{}/CqlSac_actor_optimizer_{}".format(path, episode))

    def load(self, path, episode):
        self.critic.load_state_dict(torch.load("{}/CqlSac_critic_{}".format(path, episode)))
        self.critic_optimizer.load_state_dict(torch.load("{}/CqlSac_critic_optimizer_{}".format(path, episode)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load("{}/CqlSac_actor_{}".format(path, episode)))
        self.actor_optimizer.load_state_dict(torch.load("{}/CqlSac_actor_optimizer_{}".format(path, episode)))
