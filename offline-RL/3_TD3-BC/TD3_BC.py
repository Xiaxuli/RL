import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

        self.max_action = max_action

    def forward(self, state):
        # Actor的前向传播
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, 1)

        self.l4 = nn.Linear(state_dim + action_dim, net_width)
        self.l5 = nn.Linear(net_width, net_width)
        self.l6 = nn.Linear(net_width, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_BC(object):
    def __init__(self, state_dim, action_dim, max_action, device='cpu', lr=3e-4, net_width=256, discount=0.99,
                 tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, alpha=2.5):
        # 初始化TD3_BC对象
        self.actor = Actor(state_dim, action_dim, net_width, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim, net_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.device = device
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0

    def select_action(self, state):
        # 选择动作
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        # 训练
        self.total_it += 1

        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.rand_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.discount * target_q

        current_q1, current_q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:

            pi = self.actor(state)
            q = self.critic.q1(state, pi)
            lambda_ = self.alpha / q.abs().mean().detach()

            actor_loss = -lambda_ * q.mean() + F.mse_loss(pi, action)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            return critic_loss, actor_loss
        return critic_loss, None

    def save(self, path, episode):
        # 保存模型参数
        torch.save(self.critic.state_dict(), "{}/td3_bc_critic_{}".format(path, episode))
        torch.save(self.critic_optimizer.state_dict(), "{}/td3_bc_critic_optimizer_{}".format(path, episode))

        torch.save(self.actor.state_dict(), "{}/td3_bc_actor_{}".format(path, episode))
        torch.save(self.actor_optimizer.state_dict(), "{}/td3_bc_actor_optimizer_{}".format(path, episode))

    def load(self, path, episode=None):
        # 加载模型参数
        self.critic.load_state_dict(torch.load("{}/td3_bc_critic_{}".format(path, episode)))
        self.critic_optimizer.load_state_dict(torch.load("{}/td3_bc_critic_optimizer_{}".format(path, episode)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load("{}/td3_bc_actor_{}".format(path, episode)))
        self.actor_optimizer.load_state_dict(torch.load("{}/td3_bc_actor_optimizer_{}".format(path, episode)))
        self.actor_target = copy.deepcopy(self.actor)
