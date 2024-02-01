import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(MLP, self).__init__()
        # 定义多层感知机模型的结构
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        # 定义前向传播过程
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQN(object):
    def __init__(self, state_size, action_size, hidden_size=32, lr=3e-4, gamma=0.99, device='cpu'):
        # 初始化DQN智能体的参数和模型
        self.model = MLP(state_size, hidden_size, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.target_model = copy.deepcopy(self.model)
        self.gamma = gamma
        self.device = device
        self.action_size = action_size
        self.epsilon = 0

    def train(self, replay_buffer, batch_size=128):
        # 训练DQN智能体
        self.epsilon += 1
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        next_q = self.target_model(next_state).detach().max(1)[0].unsqueeze(1)
        target_q = reward + (1 - done) * self.gamma * next_q

        q_pred = self.model(state).gather(1, action.long())
        loss = F.mse_loss(q_pred, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon % 2 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def predict(self, state, epsilon):
        # 使用epsilon-greedy策略选择动作
        if random.random() >= epsilon:
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            q = self.model(state).detach()
            return np.argmax(q.cpu().data.numpy(), axis=1)
        else:
            return random.choices(np.arange(self.action_size), k=1)

    def save(self, path, episode):
        # 保存模型和优化器的状态字典
        torch.save(self.model.state_dict(), "{}/dqn_{}".format(path, episode))
        torch.save(self.optimizer.state_dict(), "{}/dqn_optimizer_{}".format(path, episode))

    def load(self, path, episode):
        # 加载模型和优化器的状态字典
        self.model.load_state_dict(torch.load("{}/dqn_{}".format(path, episode)))
        self.optimizer.load_state_dict(torch.load("{}/dqn_optimizer_{}".format(path, episode)))
        self.target_model.load_state_dict(self.model.state_dict())
