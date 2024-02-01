import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Actor网络定义
class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action, hidden_size=64):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = nn.ReLU()(self.fc1(state))
        x = nn.ReLU()(self.fc2(x))
        x = nn.Tanh()(self.fc3(x))
        return x * self.max_action


# Critic网络定义，包含两个Q网络
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Critic, self).__init__()
        # Q网络1
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        # Q网络2
        self.fc4 = nn.Linear(state_size + action_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        q1 = torch.cat((state, action), 1)
        q1 = nn.ReLU()(self.fc1(q1))
        q1 = nn.ReLU()(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = torch.cat((state, action), 1)
        q2 = nn.ReLU()(self.fc4(q2))
        q2 = nn.ReLU()(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2


# TD3算法类
class TD3:
    def __init__(self, state_size, action_size, max_action, hidden_size=128, lr=3e-4, gamma=0.99, tau=5e-3):
        self.tau = tau
        self.gamma = gamma
        self.max_action = max_action
        self.policy_noise = 0.2
        self.noise_clip = 0.5

        self.actor = Actor(state_size, action_size, max_action, hidden_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.actor_target = copy.deepcopy(self.actor)

        # 双Q网络
        self.critic = Critic(state_size, action_size, hidden_size)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_target = copy.deepcopy(self.critic)
        self.update_cnt = 0

    # 预测动作
    def predict(self, state, noise=None):
        state = torch.tensor(np.array([state]), dtype=torch.float32)
        action = self.actor(state).detach().cpu().numpy()
        if noise:
            action = np.clip(np.random.normal(action, noise), -self.max_action, self.max_action)
        return action[0]

    # Critic的损失函数
    def _critic_loss(self, states, actions, next_states, rewards, dones):
        # 添加噪音
        noise = (torch.rand_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        action_next = self.actor_target(next_states).detach()
        action_next = (noise + action_next).clamp(-self.max_action, self.max_action)
        q_next = torch.min(*self.critic_target(next_states, action_next))
        q_target = rewards + (1 - dones) * self.gamma * q_next.detach()
        q1, q2 = self.critic(states, actions)
        loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        return loss

    # Actor的损失函数
    def _actor_loss(self, states):
        actions = self.actor(states)
        q, _ = self.critic(states, actions)
        loss = -q.mean()
        return loss

    # 训练TD3模型
    def train(self, replay_buffer, batch_size):
        states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)

        critic_loss = self._critic_loss(states, actions, next_states, rewards, dones)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟更新Actor
        if self.update_cnt % 2 == 0:
            actor_loss = self._actor_loss(states)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 更新目标网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # 保存模型
    def save(self, path, episode):
        torch.save(self.critic.state_dict(), "{}/td3_critic_{}".format(path, episode))
        torch.save(self.critic_optimizer.state_dict(), "{}/td3_critic_optimizer_{}".format(path, episode))
        torch.save(self.critic_target.state_dict(), "{}/td3_critic_target_{}".format(path, episode))

        torch.save(self.actor.state_dict(), "{}/td3_actor_{}".format(path, episode))
        torch.save(self.actor_optimizer.state_dict(), "{}/td3_actor_optimizer_{}".format(path, episode))
        torch.save(self.actor_target.state_dict(), "{}/td3_actor_target_{}".format(path, episode))

    # 加载模型
    def load(self, path, episode):
        self.critic.load_state_dict(torch.load("{}/td3_critic_{}".format(path, episode)))
        self.critic_optimizer.load_state_dict(torch.load("{}/td3_critic_optimizer_{}".format(path, episode)))
        self.critic_target.load_state_dict(torch.load("{}/td3_critic_target_{}".format(path, episode)))

        self.actor.load_state_dict(torch.load("{}/td3_actor_{}".format(path, episode)))
        self.actor_optimizer.load_state_dict(torch.load("{}/td3_actor_optimizer_{}".format(path, episode)))
        self.actor_target.load_state_dict(torch.load("{}/td3_actor_target_{}".format(path, episode)))
