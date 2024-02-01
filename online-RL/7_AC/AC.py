import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        v = F.relu(self.fc1(state))
        v = self.fc2(v)
        return v

# 定义Actor-Critic类
class ActorCritic(object):
    def __init__(self, state_size, action_size, hidden_size=16, lr=2e-3, gamma=0.96):
        self.gamma = gamma
        self.action_size = action_size
        # 初始化Actor网络
        self.actor = Actor(state_size, action_size, hidden_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        # 初始化Critic网络
        self.critic = Critic(state_size, hidden_size)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    # 选择动作
    def predict(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float32)
        probs = torch.softmax(self.actor(state).detach(), dim=1)
        a = torch.multinomial(probs, 1).item()
        return a

    # 训练Actor-Critic网络
    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)

        # 计算当前状态的值函数估计
        v = self.critic(state)
        # 计算目标值函数估计
        target_v = reward + (1 - int(done)) * self.gamma * self.critic(next_state).detach()
        # Critic网络的损失函数为均方误差
        critic_loss = F.mse_loss(v, target_v)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算Actor的损失函数
        _logit = self.actor(state)
        log_probs = torch.log(F.softmax(_logit, dim=1))
        actor_loss = (-log_probs.gather(1, action.unsqueeze(1)) * (target_v.detach() - v.detach())).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    # 保存模型
    def save(self, path, episode):
        torch.save(self.actor.state_dict(), "{}/ac_actor_{}".format(path, episode))
        torch.save(self.actor_optimizer.state_dict(), "{}/ac_actor_optimizer_{}".format(path, episode))
        torch.save(self.critic.state_dict(), "{}/ac_critic_{}".format(path, episode))
        torch.save(self.critic_optimizer.state_dict(), "{}/ac_critic_optimizer_{}".format(path, episode))

    # 加载模型
    def load(self, path, episode):
        self.actor.load_state_dict(torch.load("{}/ac_actor_{}".format(path, episode)))
        self.actor_optimizer.load_state_dict(torch.load("{}/ac_actor_optimizer_{}".format(path, episode)))
        self.critic.load_state_dict(torch.load("{}/ac_critic_{}".format(path, episode)))
        self.critic_optimizer.load_state_dict(torch.load("{}/ac_critic_optimizer_{}".format(path, episode)))
