import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, max_action):
        super(MLP, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh((self.l3(a)))
        return self.max_action * a


class BC(object):
    def __init__(self, state_dim, action_dim, max_action, device='cpu', lr=3e-4, net_width=256):
        self.bc = MLP(state_dim, action_dim, net_width, max_action).to(device)
        self.bc_optimizer = torch.optim.Adam(self.bc.parameters(), lr=lr)

        self.device = device
        self.max_action = action_dim

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.bc(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        pi = self.bc(state)
        action_loss = F.mse_loss(pi, action)
        self.bc_optimizer.zero_grad()
        action_loss.backward()
        self.bc_optimizer.step()

    def save(self, path, episode):
        torch.save(self.bc.state_dict(), "{}/bc_critic_{}".format(path, episode))
        torch.save(self.bc_optimizer.state_dict(), "{}/bc_critic_optimizer_{}".format(path, episode))

    def load(self, path, episode=None):
        self.bc.load_state_dict(torch.load("{}/bc_critic_{}".format(path, episode)))
        self.bc_optimizer.load_state_dict(torch.load("{}/bc_critic_optimizer_{}".format(path, episode)))
