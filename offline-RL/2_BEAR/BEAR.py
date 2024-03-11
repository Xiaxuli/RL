import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

torch.autograd.set_detect_anomaly(True)


def atanh(x):
    # 计算反双曲正切函数的实现
    one_plus_x = (1 + x).clamp(min=1e-7)
    one_minus_x = (1 - x).clamp(min=1e-7)
    return 0.5 * torch.log(one_plus_x / one_minus_x)


class RegularActor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, max_action, device):
        super(RegularActor, self).__init__()
        # 定义神经网络层
        self.l1 = nn.Linear(state_dim, net_width)  # 输入层到第一隐藏层
        self.l2 = nn.Linear(net_width, net_width)  # 第一隐藏层到第二隐藏层
        self.mean = nn.Linear(net_width, action_dim)  # 均值输出层
        self.log_std = nn.Linear(net_width, action_dim)  # 对数标准差输出层
        self.max_action = max_action  # 动作的最大值
        self.device = device

    def forward(self, state):
        # 神经网络前向传播
        a = F.relu(self.l1(state))  # 第一隐藏层，使用ReLU激活函数
        a = F.relu(self.l2(a))  # 第二隐藏层，使用ReLU激活函数
        mean_a = self.mean(a)  # 计算动作的均值
        log_std_a = self.log_std(a)  # 计算动作的对数标准差

        std_a = torch.exp(log_std_a)  # 计算标准差，使用指数函数
        z = mean_a + std_a * torch.FloatTensor(np.random.normal(0, 1, size=(std_a.size()))).to(
            self.device)  # 通过均值和标准差生成动作 z
        return self.max_action * torch.tanh(z)  # 返回经过 Tanh 函数缩放的动作

    def sample_multiple(self, state, num_sample=10):
        # 在多次采样的情况下生成动作
        # a = f.relu(self.l1(state))  # 第一隐藏层，使用ReLU激活函数
        # a = f.relu(self.l2(a))  # 第二隐藏层，使用ReLU激活函数
        # mean_a = self.mean(a)  # 计算动作的均值
        # log_std_a = self.log_std(a)  # 计算动作的对数标准差
        a = F.linear(state, self.l1.weight.clone(), self.l1.bias)
        a = F.relu(a)  # 使用ReLU激活函数
        a = F.linear(a, self.l2.weight.clone(), self.l2.bias)
        a = F.relu(a)  # 使用ReLU激活函数
        mean_a = F.linear(a, self.mean.weight.clone(), self.mean.bias)
        log_std_a = F.linear(a, self.log_std.weight.clone(), self.log_std.bias)

        std_a = torch.exp(log_std_a)  # 计算标准差，使用指数函数
        z = mean_a.unsqueeze(1) + std_a.unsqueeze(1) * torch.FloatTensor(
            np.random.normal(0, 1, size=(
                std_a.size(0), num_sample, std_a.size(1)))).to(self.device).clamp(-0.5, 0.5)  # 生成多个动作 z
        return self.max_action * torch.tanh(z), z  # 返回多个动作，经过 Tanh 函数缩放的动作以及生成的 z

    def log_pis(self, state, action=None, raw_action=None):
        # 计算策略的对数概率
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        std_a = torch.exp(log_std_a)
        normal_dist = td.Normal(loc=mean_a, scale=std_a, validate_args=True)
        if raw_action is None:
            raw_action = atanh(action)  # 如果原始动作未提供，则通过反双曲正切函数计算原始动作
        else:
            action = torch.tanh(raw_action)  # 如果原始动作已提供，则通过双曲正切函数计算动作
        log_normal = normal_dist.log_prob(raw_action)  # 计算原始动作在正态分布下的对数概率
        log_pis = log_normal.sum(-1)  # 对数概率求和，通常用于策略梯度方法
        log_pis = log_pis - (1.0 - action ** 2).clamp(min=1e-6).log().sum(-1)  # 应用校正项，确保对数概率在动作变换后保持正确
        return log_pis  # 返回策略的对数概率


class EnsembleCritic(nn.Module):
    def __init__(self, num_qs, state_dim, action_dim, net_width):
        super(EnsembleCritic, self).__init__()

        self.num_qs = num_qs  # Critic 集成的数量
        self.q_modules = nn.ModuleList()  # 存储每个 Critic 的模型

        for _ in range(num_qs):
            # 定义神经网络层
            q_module = nn.Sequential(
                nn.Linear(state_dim + action_dim, net_width),
                nn.ReLU(),
                nn.Linear(net_width, net_width),
                nn.ReLU(),
                nn.Linear(net_width, 1)
            )
            self.q_modules.append(q_module)

    def forward(self, state, action):
        all_qs = []

        # 计算每个 Critic 的 Q 值
        for q_module in self.q_modules:
            q = q_module(torch.cat([state, action], 1))
            all_qs.append(q)

        all_qs = torch.cat([q.unsqueeze(0) for q in all_qs], 0)  # 将每个 Critic 的 Q 值堆叠起来
        return all_qs  # 返回所有 Critic 的 Q 值

    def q1(self, state, action):
        return self.q_modules[0](torch.cat([state, action], 1))


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, net_width, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, net_width)  # 编码器的第一层全连接层
        self.e2 = nn.Linear(net_width, net_width)  # 编码器的第二层全连接层

        self.mean = nn.Linear(net_width, latent_dim)  # 隐变量均值的全连接层
        self.log_std = nn.Linear(net_width, latent_dim)  # 隐变量对数标准差的全连接层

        self.d1 = nn.Linear(state_dim + latent_dim, net_width)  # 解码器的第一层全连接层
        self.d2 = nn.Linear(net_width, net_width)  # 解码器的第二层全连接层
        self.d3 = nn.Linear(net_width, action_dim)  # 解码器的输出全连接层

        self.max_action = max_action  # 动作的最大值
        self.latent_dim = latent_dim  # 隐变量维度
        self.device = device

    def forward(self, state, action):
        # 编码器部分
        z = F.relu(self.e1(torch.cat([state, action], 1)))  # 将状态和动作拼接并经过第一层全连接层
        z = F.relu(self.e2(z))  # 经过第二层全连接层

        mean = self.mean(z)  # 计算隐变量均值
        log_std = self.log_std(z).clamp(-4, 15)  # 计算隐变量对数标准差并进行截断
        std = torch.exp(log_std)  # 计算标准差
        z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(self.device)  # 从正态分布中采样隐变量

        u = self.decode(state, z)  # 解码器

        return u, mean, std

    def decode(self, state, z=None):
        # 如果隐变量 z 未提供，则从正态分布中随机采样
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(self.device).clamp(
                -0.5, 0.5)

        # 解码器部分
        a = F.relu(self.d1(torch.cat([state, z], 1)))  # 将状态和隐变量拼接并经过第一层全连接层
        a = F.relu(self.d2(a))  # 经过第二层全连接层
        return self.max_action * torch.tanh(self.d3(a))  # 经过输出层并应用 tanh 激活，然后缩放到 [-max_action, max_action] 范围内的动作

    def decode_multiple(self, state, z=None, num_decode=10):
        # 如果隐变量 z 未提供，则从正态分布中随机采样
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), num_decode, self.latent_dim))).to(
                self.device).clamp(-0.5, 0.5)

        # 解码器部分
        # 将状态和隐变量拼接并经过第一层全连接层
        a = F.relu(self.d1(torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))  # 经过第二层全连接层
        a = self.d3(a)
        return self.max_action * torch.tanh(a), a  # 经过输出层并应用 tanh 激活，然后缩放到 [-max_action, max_action] 范围内的动作


class BEAR(object):
    def __init__(self, num_qs, state_dim, action_dim, max_action, lr=1e-3, actor_net_width=256, critic_net_width=256,
                 vae_net_width=512, device='cpu', delta_conf=0.1, use_bootstrap=True, version=0, lambda_=0.4,
                 threshold=0.05, mode='auto', num_samples_match=10, mmd_sigma=10.0, lagrange_thresh=10.0, use_kl=False,
                 use_ensemble=True, kernel_type='laplacian'):
        """
        初始化 BEAR (Batch Ensemble Actor-Critic with MMD Support) 算法。

        参数:
            num_qs (int): 评估集合中 Q 值的数量。
            state_dim (int): 状态空间的维度。
            action_dim (int): 动作空间的维度。
            max_action (float): 动作空间的最大值。
            lr(float): 学习率
            actor_net_width (int): 演员网络的宽度（神经元数目）。
            critic_net_width (int): 评论家网络的宽度。
            vae_net_width (int): VAE 网络的宽度。
            device (str): 运行设备（'cpu' 或其他，例如 'cuda'）。
            delta_conf (float): 用于支持匹配的 MMD 约束的置信度。
            use_bootstrap (bool): 是否使用 bootstrap 来训练。
            version (int): 选择策略更新的版本（0, 1, 2 中的一个）。
            lambda_ (float): 用于计算策略损失的权重。
            threshold (float): 策略损失的阈值。
            mode (str): 模式，可以是 'auto' 或其他。
            num_samples_match (int): 支持匹配中采样的样本数量。
            mmd_sigma (float): MMD 损失函数中的 sigma 参数。
            lagrange_thresh (float): Lagrange 乘子的阈值。
            use_kl (bool): 是否使用 KL 散度损失。
            use_ensemble (bool): 是否使用集成网络。
            kernel_type (str): 用于 MMD 的核函数类型（'laplacian' 或 'gaussian'）。
        """
        # 计算潜变量维度
        latent_dim = action_dim * 2

        # 初始化演员（Actor）网络
        self.actor = RegularActor(state_dim, action_dim, actor_net_width, max_action, device).to(device)
        self.actor_target = copy.deepcopy(self.actor)  # 创建演员网络的目标网络副本
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)  # 初始化演员网络的优化器

        # 初始化评论家（Critic）网络
        self.critic = EnsembleCritic(num_qs, state_dim, action_dim, critic_net_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)  # 创建评论家网络的目标网络副本
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)  # 初始化评论家网络的优化器

        # 初始化变分自编码器（VAE）网络
        self.vae = VAE(state_dim, action_dim, latent_dim, vae_net_width, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)  # 初始化VAE网络的优化器

        # 初始化其他超参数和模式参数
        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.delta_conf = delta_conf
        self.device = device
        self.use_bootstrap = use_bootstrap
        self.version = version
        self._lambda = lambda_
        self.threshold = threshold
        self.mode = mode
        self.num_qs = num_qs
        self.num_samples_match = num_samples_match
        self.mmd_sigma = mmd_sigma
        self.lagrange_thresh = lagrange_thresh
        self.use_kl = use_kl
        self.use_ensemble = use_ensemble
        self.kernel_type = kernel_type

        if self.mode == 'auto':
            # 如果模式为 'auto'，则初始化 Lagrange 乘子
            self.log_lagrange2 = torch.randn((), requires_grad=True, device=device)
            self.lagrange2_opt = torch.optim.Adam([self.log_lagrange2, ], lr=1e-3)

        self.epoch = 0  # 初始化训练轮数

    def kl_loss(self, samples1, state):
        """
        计算策略分布之间的 KL 散度损失。

        参数:
            samples1 (Tensor): 从演员网络采样的动作。
            state (Tensor): 当前状态。

        返回:
            kl_loss (Tensor): KL 散度损失。
        """
        state_rep = state.unsqueeze(1).repeat(1, samples1.size(1), 1).view(-1, state.size(-1))
        samples1_reshape = samples1.view(-1, samples1.size(-1))
        samples1_log_pis = self.actor.log_pis(state=state_rep, raw_action=samples1_reshape)
        samples1_log_prob = samples1_log_pis.view(state.size(0), samples1.size(1))
        return (-samples1_log_prob).mean(1)

    @staticmethod
    def mmd_loss_laplacian(samples1, samples2, sigma=0.2):
        """
        计算基于拉普拉斯核的最大均值差异（MMD）损失。

        参数:
            samples1 (Tensor): 一组样本。
            samples2 (Tensor): 另一组样本。
            sigma (float): 高斯核的带宽。

        返回:
            mmd_loss (Tensor): MMD 损失。
        """
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    @staticmethod
    def mmd_loss_gaussian(samples1, samples2, sigma=0.2):
        """
        计算基于高斯核的最大均值差异（MMD）损失。

        参数:
            samples1 (Tensor): 一组样本。
            samples2 (Tensor): 另一组样本。
            sigma (float): 拉普拉斯核的带宽。

        返回:
            mmd_loss (Tensor): MMD 损失。
        """
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)
        diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)
        diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def select_action(self, state):
        """
        通过当前状态选择下一步的动作。

        参数:
            state (array): 当前状态。

        返回:
            action (Tensor): 选择的动作。
        """
        # 禁用梯度计算，因为我们只是在选择动作，而不是训练模型
        with torch.no_grad():
            # 将状态转换为PyTorch张量，并在设备上进行处理
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(self.device)
            # 使用演员网络（Actor）来计算动作
            action = self.actor(state)
            # 使用评论家网络（Critic）计算Q值（Q1值）
            q1 = self.critic.q1(state, action)
            # 从Q1值中选择具有最高值的动作索引
            ind = q1.max(0)[1]

        # 返回选择的动作，将其从PyTorch张量转换为NumPy数组，并扁平化为一维数组
        return action[ind].cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=128, discount=0.99, tau=0.005):
        """
        使用 BEAR 算法进行训练。

        参数:
            replay_buffer (ReplayBuffer): 经验回放缓冲区。
            iterations (int): 训练迭代次数。
            batch_size (int): 每次训练使用的批次大小。
            discount (float): 折扣因子。
            tau (float): 用于更新目标网络参数的软更新参数。
        """
        for it in range(iterations):
            # 步骤 1: 从经验回放缓冲区中采样批次数据
            state, action, next_state, reward, done = replay_buffer.sample(batch_size)

            # 步骤 2: 计算 VAE 损失
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * kl_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            with torch.no_grad():
                # 步骤 3: 使用目标网络计算 Q 值
                state_rep = torch.FloatTensor(np.repeat(next_state, 10, axis=0)).to(self.device)

                # 计算目标 Q 值
                target_qs = self.critic_target(state_rep, self.actor_target(state_rep))
                target_q = 0.75 * target_qs.min(0)[0] + 0.25 * target_qs.max(0)[0]
                target_q = target_q.view(batch_size, -1).max(1)[0].view(-1, 1)
                target_q = reward + (1 - done) * discount * target_q

            target_qs = self.critic(state, action)

            # 步骤 4: 计算评论家（Critic）网络的损失
            critic_loss = torch.tensor(0.)
            for i in range(self.num_qs):
                if self.use_bootstrap:
                    mask = np.squeeze(np.random.binomial(n=1, size=(1, batch_size, self.num_qs,), p=0.8), axis=0)
                    mask = torch.FloatTensor(mask).to(self.device)
                    critic_loss += (F.mse_loss(target_qs[i], target_q, reduction='none') * mask[:, i:i + 1]).mean()
                else:
                    critic_loss += F.mse_loss(target_qs[i], target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 步骤 5: 从 VAE 中采样动作和从演员（Actor）网络中采样动作
            sampled_actions, raw_sampled_actions = self.vae.decode_multiple(state, num_decode=self.num_samples_match)
            actor_actions, raw_actor_actions = self.actor.sample_multiple(state, self.num_samples_match)

            # 步骤 6: 计算 MMD 损失
            if self.use_kl:
                mmd_loss = self.kl_loss(raw_actor_actions, state)
            else:
                if self.kernel_type == 'gaussian':
                    mmd_loss = self.mmd_loss_gaussian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)
                else:
                    mmd_loss = self.mmd_loss_laplacian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)

            # 步骤 7: 计算集成 Q 值
            critic_qs = self.critic(state.unsqueeze(0).repeat(self.num_samples_match, 1, 1).view(
                self.num_samples_match * state.size(0), state.size(1)),
                actor_actions.permute(1, 0, 2).contiguous().view(
                    self.num_samples_match * actor_actions.size(0), actor_actions.size(2)))
            critic_qs = critic_qs.view(self.num_qs, self.num_samples_match, actor_actions.size(0), 1)
            critic_qs = critic_qs.mean(1)
            std_q = torch.std(critic_qs, dim=0, unbiased=False)

            if not self.use_ensemble:
                std_q = torch.zeros_like(std_q).to(self.device)

            # 步骤 8: 计算演员网络的损失（actor_loss 或 action_loss）
            if self.version == '0':
                critic_qs = critic_qs.min(0)[0]
            elif self.version == '1':
                critic_qs = critic_qs.max(0)[0]
            else:
                critic_qs = critic_qs.mean(0)

            if self.epoch >= 20:
                if self.mode == 'auto':
                    actor_loss = (-critic_qs + self._lambda * (
                        np.sqrt((1 - self.delta_conf) / self.delta_conf)) * std_q +
                                  self.log_lagrange2.exp() * mmd_loss).mean()
                else:
                    actor_loss = (-critic_qs + self._lambda * (
                        np.sqrt((1 - self.delta_conf) / self.delta_conf)) * std_q + 100.0 * mmd_loss).mean()
            else:
                if self.mode == 'auto':
                    actor_loss = (self.log_lagrange2.exp() * mmd_loss).mean()
                else:
                    actor_loss = 100.0 * mmd_loss.mean()

            # 步骤 9: 更新演员网络的参数
            self.actor_optimizer.zero_grad()
            if self.mode == 'auto':
                actor_loss.backward(retain_graph=True)
            else:
                actor_loss.backward()
            self.actor_optimizer.step()

            thresh = -2.0 if self.use_kl else self.threshold
            if self.mode == 'auto':
                # 步骤 10: 计算 Lagrange 乘子的损失（lagrange_loss）
                lagrange_loss = (-critic_qs + self._lambda * (
                    np.sqrt((1 - self.delta_conf) / self.delta_conf)) * std_q + self.log_lagrange2.exp() * (
                                         mmd_loss - thresh)).mean()
                self.lagrange2_opt.zero_grad()
                (-lagrange_loss).backward()
                self.lagrange2_opt.step()
                self.log_lagrange2.data.clamp_(min=-5.0, max=self.lagrange_thresh)

            # 步骤 11: 更新目标网络的参数
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # 步骤 12: 增加训练轮数 self.epoch
        self.epoch = self.epoch + 1

    def save(self, path, episode):
        # 保存模型和优化器的状态字典
        torch.save(self.actor.state_dict(), "{}/bear_actor_{}".format(path, episode))
        torch.save(self.actor_optimizer.state_dict(), "{}/bear_actor_optimizer_{}".format(path, episode))
        torch.save(self.vae.state_dict(), "{}/bear_vae_{}".format(path, episode))
        torch.save(self.vae_optimizer.state_dict(), "{}/bear_vae_optimizer_{}".format(path, episode))
        torch.save(self.critic.state_dict(), "{}/bear_critic_{}".format(path, episode))
        torch.save(self.critic_optimizer.state_dict(), "{}/bear_critic_optimizer_{}".format(path, episode))
        torch.save(self.actor_target.state_dict(), "{}/bear_actor_target_{}".format(path, episode))
        torch.save(self.critic_target.state_dict(), "{}/bear_critic_target_{}".format(path, episode))

    def load(self, path, episode):
        self.actor.load_state_dict(torch.load("{}/bear_actor_{}".format(path, episode)))
        self.actor_optimizer.load_state_dict(torch.load("{}/bear_actor_optimizer_{}".format(path, episode)))
        self.vae.load_state_dict(torch.load("{}/bear_vae_{}".format(path, episode)))
        self.vae_optimizer.load_state_dict(torch.load("{}/bear_vae_optimizer_{}".format(path, episode)))
        self.critic.load_state_dict(torch.load("{}/bear_critic_{}".format(path, episode)))
        self.critic_optimizer.load_state_dict(torch.load("{}/bear_critic_optimizer_{}".format(path, episode)))
        self.actor_target.load_state_dict(torch.load("{}/bear_actor_target_{}".format(path, episode)))
        self.critic_target.load_state_dict(torch.load("{}/bear_critic_target_{}".format(path, episode)))
