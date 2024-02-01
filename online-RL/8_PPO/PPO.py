import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor网络定义
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.sigma = nn.Linear(hidden_size, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = F.tanh(self.mean(x)) * self.max_action
        sigma = nn.Softplus()(self.sigma(x))
        return mean, sigma

# Critic网络定义
class Critic(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# PPO算法类
class PPO:
    def __init__(self, state_size, action_dim, max_action, hidden_size=64, lr=3e-4, gamma=0.95):
        # 初始化Actor和Critic网络
        self.actor = Actor(state_size, action_dim, max_action, hidden_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_size, hidden_size)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.max_action = max_action
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.cumulative_reward_buffer = []

    # 预测动作
    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mu, sigma = self.actor(state)
        pi = torch.distributions.Normal(mu, sigma)
        a = pi.sample().clamp(-self.max_action, self.max_action)
        return a.detach().cpu().numpy()

    # 将状态、动作、奖励添加到缓冲区
    def buffer_add(self, state, action, reward):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    # 计算折扣累积奖励
    def get_v(self, next_state, done):
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        v_s_ = (1 - int(done)) * self.critic(next_state).item()
        for r in self.reward_buffer[::-1]:
            v_s_ = r + self.gamma * v_s_
            self.cumulative_reward_buffer.append(v_s_)
        self.cumulative_reward_buffer.reverse()
        self.reward_buffer.clear()

    # 计算Actor的损失
    def actor_loss(self):
        states = torch.tensor(self.state_buffer, dtype=torch.float32)
        actions = torch.tensor(self.action_buffer, dtype=torch.float32)
        v = torch.tensor(self.cumulative_reward_buffer, dtype=torch.float32).unsqueeze(1)
        adv = v - self.critic(states)

        pi = torch.distributions.Normal(*self.actor(states))
        old_pi = torch.distributions.Normal(*self.actor_target(states))
        ratio = torch.exp(pi.log_prob(actions) - old_pi.log_prob(actions) + 1e-8)
        actor_loss = -torch.mean(torch.minimum(ratio * adv, torch.clamp(ratio, 1. - 0.2, 1. + 0.2) * adv))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    # 计算Critic的损失
    def critic_loss(self):
        states = torch.tensor(self.state_buffer, dtype=torch.float32)
        v = torch.tensor(self.cumulative_reward_buffer, dtype=torch.float32).unsqueeze(1)
        critic_loss = F.mse_loss(self.critic(states), v)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    # 训练PPO模型
    def train(self, next_state, done, train_size):
        self.get_v(next_state, done)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # 训练Actor
        for _ in range(train_size):
            self.actor_loss()

        # 训练Critic
        for _ in range(train_size):
            self.critic_loss()

        self.state_buffer = []
        self.action_buffer = []
        self.cumulative_reward_buffer = []

    # 保存模型
    def save(self, path, episode):
        torch.save(self.critic.state_dict(), "{}/ppo_critic_{}".format(path, episode))
        torch.save(self.critic_optimizer.state_dict(), "{}/ppo_critic_optimizer_{}".format(path, episode))

        torch.save(self.actor.state_dict(), "{}/ppo_actor_{}".format(path, episode))
        torch.save(self.actor_optimizer.state_dict(), "{}/ppo_actor_optimizer_{}".format(path, episode))

    # 加载模型
    def load(self, path, episode):
        self.critic.load_state_dict(torch.load("{}/ppo_critic_{}".format(path, episode)))
        self.critic_optimizer.load_state_dict(torch.load("{}/ppo_critic_optimizer_{}".format(path, episode)))

        self.actor.load_state_dict(torch.load("{}/ppo_actor_{}".format(path, episode)))
        self.actor_optimizer.load_state_dict(torch.load("{}/ppo_actor_optimizer_{}".format(path, episode)))
