import numpy as np
import torch
import torch.nn as nn


# 定义多层感知机（MLP）模型
class MLP(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义Policy Gradient类
class PG:
    def __init__(self, state_size, action_size, hidden_size, lr=3e-4, gamma=0.99):
        # 初始化MLP模型作为策略
        self.policy = MLP(state_size, hidden_size, action_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    # 预测动作
    def predict(self, state):
        states = torch.tensor(np.array([state]), dtype=torch.float32)
        probs = torch.softmax(self.policy(states), dim=1)
        a = torch.distributions.Categorical(probs).sample()
        return a.detach().cpu().numpy()

    # 训练Policy Gradient模型
    def train(self, states, actions, rewards):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)

        # 计算折扣累积奖励G
        g_lis = np.zeros_like(rewards)
        g = 0
        # 从ep_rs的最后往前，逐个计算G
        for t in range(len(rewards), 0, -1):
            g = g * self.gamma + rewards[t - 1]
            g_lis[t - 1] = g

        # 归一化G值
        g_lis -= np.mean(g_lis)
        g_lis /= np.std(g_lis)
        g_lis = torch.tensor(g_lis, dtype=torch.float32).unsqueeze(1)

        # 计算Policy Gradient的损失函数
        pred_g = self.policy(states)
        log_probs = torch.log(torch.softmax(pred_g, dim=1))
        loss = torch.mean(-log_probs.gather(1, actions) * g_lis)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 保存模型
    def save(self, path, episode):
        torch.save(self.policy.state_dict(), "{}/pg_{}".format(path, episode))
        torch.save(self.optimizer.state_dict(), "{}/pg_optimizer_{}".format(path, episode))

    # 加载模型
    def load(self, path, episode):
        self.policy.load_state_dict(torch.load("{}/pg_{}".format(path, episode)))
        self.optimizer.load_state_dict(torch.load("{}/pg_optimizer_{}".format(path, episode)))
