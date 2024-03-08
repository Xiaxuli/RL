import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# 扰动模型 Actor 类
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, max_action, phi=0.05):
        super(Actor, self).__init__()

        # 定义神经网络的层
        self.l1 = nn.Linear(state_dim + action_dim, net_width)  # 第一层全连接层
        self.l2 = nn.Linear(net_width, net_width)  # 第二层全连接层
        self.l3 = nn.Linear(net_width, action_dim)  # 输出层

        # 最大动作范围
        self.max_action = max_action

        # 策略扰动参数
        self.phi = phi

    def forward(self, state, action):
        # 前向传播函数
        a = F.relu(self.l1(torch.cat([state, action], 1)))  # 使用 ReLU 激活函数的第一层
        a = F.relu(self.l2(a))  # 使用 ReLU 激活函数的第二层
        a = self.phi * self.max_action * torch.tanh(self.l3(a))  # 输出层经过 tanh 激活函数，并乘以扰动参数
        return (a + action).clamp(-self.max_action, self.max_action)  # 返回最终动作值，并确保在合法动作范围内


# 双 Q 网络 Critic 类
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Critic, self).__init__()

        # 第一个 Q 网络（Q1）
        self.l1 = nn.Linear(state_dim + action_dim, net_width)  # 第一层全连接层
        self.l2 = nn.Linear(net_width, net_width)  # 第二层全连接层
        self.l3 = nn.Linear(net_width, 1)  # 输出层

        # 第二个 Q 网络（Q2）
        self.l4 = nn.Linear(state_dim + action_dim, net_width)  # 第一层全连接层
        self.l5 = nn.Linear(net_width, net_width)  # 第二层全连接层
        self.l6 = nn.Linear(net_width, 1)  # 输出层

    def forward(self, state, action):
        # 前向传播函数，计算两个 Q 值
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))  # 使用 ReLU 激活函数的第一层
        q1 = F.relu(self.l2(q1))  # 使用 ReLU 激活函数的第二层
        q1 = self.l3(q1)  # 输出层

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))  # 使用 ReLU 激活函数的第一层
        q2 = F.relu(self.l5(q2))  # 使用 ReLU 激活函数的第二层
        q2 = self.l6(q2)  # 输出层

        return q1, q2  # 返回两个 Q 值

    def q1(self, state, action):
        # 计算第一个 Q 值
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))  # 使用 ReLU 激活函数的第一层
        q1 = F.relu(self.l2(q1))  # 使用 ReLU 激活函数的第二层
        q1 = self.l3(q1)  # 输出层
        return q1  # 返回第一个 Q 值


# 变分自编码器 VAE 类
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, net_width, max_action, device):
        super(VAE, self).__init__()

        # 编码器部分
        self.e1 = nn.Linear(state_dim + action_dim, net_width)  # 编码器的第一层全连接层
        self.e2 = nn.Linear(net_width, net_width)  # 编码器的第二层全连接层

        self.mean = nn.Linear(net_width, latent_dim)  # 编码器的均值输出层
        self.log_std = nn.Linear(net_width, latent_dim)  # 编码器的标准差输出层

        # 解码器部分
        self.d1 = nn.Linear(state_dim + latent_dim, net_width)  # 解码器的第一层全连接层
        self.d2 = nn.Linear(net_width, net_width)  # 解码器的第二层全连接层
        self.d3 = nn.Linear(net_width, action_dim)  # 解码器的输出层

        self.latent_dim = latent_dim  # 潜在变量的维度
        self.max_action = max_action  # 动作的最大值
        self.device = device  # 设备（通常是 GPU 或 CPU）

    def forward(self, state, action):
        # 编码器部分
        z = F.relu(self.e1(torch.cat([state, action], 1)))  # 使用 ReLU 激活函数的第一层
        z = F.relu(self.e2(z))  # 使用 ReLU 激活函数的第二层

        mean = self.mean(z)  # 编码器输出的均值
        log_std = self.log_std(z).clamp(-4, 15)  # 编码器输出的对数标准差，进行截断以限制范围
        std = torch.exp(log_std)  # 根据对数标准差计算标准差
        z = mean + std * torch.rand_like(std)  # 从均值和标准差生成潜在变量

        u = self.decode(state, z)  # 解码器生成动作

        return u, mean, std  # 返回生成的动作、均值和标准差

    def decode(self, state, z=None):
        # 解码器部分
        if z is None:
            # 如果没有给定潜在变量 z，生成一个随机的 z
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)
        a = F.relu(self.d1(torch.cat([state, z], 1)))  # 使用 ReLU 激活函数的第一层
        a = F.relu(self.d2(a))  # 使用 ReLU 激活函数的第二层
        return self.max_action * torch.tanh(self.d3(a))  # 输出层经过 tanh 激活函数，并缩放到动作的最大值范围内


class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, device='cpu', lr=1e-3,
                 a_net_width=128, c_net_width=128, vae_net_width=256, gamma=0.99, tau=0.005, lambda_=0.75, phi=0.05):
        latent_dim = action_dim * 2

        # 初始化 Actor 模型
        self.actor = Actor(state_dim, action_dim, a_net_width, max_action, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # 初始化 Critic 模型
        self.critic = Critic(state_dim, action_dim, c_net_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # 初始化 VAE 模型
        self.vae = VAE(state_dim, action_dim, latent_dim, vae_net_width, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

        # 其他参数和超参数
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.lambda_ = lambda_

    def select_action(self, state):
        with torch.no_grad():
            # 通过 Actor 模型选择动作
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
            action = self.actor(state, self.vae.decode(state))
            q1 = self.critic.q1(state, action)
            ind = q1.argmax(0)
        return action[ind].cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100):

        for it in range(iterations):
            state, action, next_state, reward, done = replay_buffer.sample(batch_size)
            # 训练 VAE 模型
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * kl_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            with torch.no_grad():
                # 扩展下一个状态的样本
                next_state = torch.repeat_interleave(next_state, 10, 0)
                # 计算目标 Q 值
                target_q1, target_q2 = self.critic_target(next_state,
                                                          self.actor_target(next_state, self.vae.decode(next_state)))
                target_q = self.lambda_ * torch.min(target_q1, target_q2) + (1 - self.lambda_) * torch.max(target_q1,
                                                                                                           target_q2)
                target_q = target_q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

                # 计算 TD 目标
                target_q = reward + (1 - done) * self.gamma * target_q

            # 训练 Critic 模型
            current_q1, current_q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 通过 VAE 模型生成样本动作
            sample_actions = self.vae.decode(state)
            # 通过 Actor 模型选择扰动动作
            perturbed_actions = self.actor(state, sample_actions)

            # 计算 Actor 损失
            action_loss = -self.critic.q1(state, perturbed_actions).mean()

            self.actor_optimizer.zero_grad()
            action_loss.backward()
            self.actor_optimizer.step()

            # 更新 Critic 目标网络和 Actor 目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path, episode):
        # 保存模型和优化器的状态字典
        torch.save(self.actor.state_dict(), "{}/bcq_actor_{}".format(path, episode))
        torch.save(self.actor_optimizer.state_dict(), "{}/bcq_actor_optimizer_{}".format(path, episode))
        torch.save(self.vae.state_dict(), "{}/bcq_vae_{}".format(path, episode))
        torch.save(self.vae_optimizer.state_dict(), "{}/bcq_vae_optimizer_{}".format(path, episode))
        torch.save(self.critic.state_dict(), "{}/bcq_critic_{}".format(path, episode))
        torch.save(self.critic_optimizer.state_dict(), "{}/bcq_critic_optimizer_{}".format(path, episode))
        torch.save(self.actor_target.state_dict(), "{}/bcq_actor_target_{}".format(path, episode))
        torch.save(self.critic_target.state_dict(), "{}/bcq_critic_target_{}".format(path, episode))

    def load(self, path, episode):
        # 加载模型和优化器的状态字典
        self.actor.load_state_dict(torch.load("{}/bcq_actor_{}".format(path, episode)))
        self.actor_optimizer.load_state_dict(torch.load("{}/bcq_actor_optimizer_{}".format(path, episode)))
        self.vae.load_state_dict(torch.load("{}/bcq_vae_{}".format(path, episode)))
        self.vae_optimizer.load_state_dict(torch.load("{}/bcq_vae_optimizer_{}".format(path, episode)))
        self.critic.load_state_dict(torch.load("{}/bcq_critic_{}".format(path, episode)))
        self.critic_optimizer.load_state_dict(torch.load("{}/bcq_critic_optimizer_{}".format(path, episode)))
        self.actor_target.load_state_dict(torch.load("{}/bcq_actor_target_{}".format(path, episode)))
        self.critic_target.load_state_dict(torch.load("{}/bcq_critic_target_{}".format(path, episode)))
