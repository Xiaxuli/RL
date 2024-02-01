import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device='cpu', max_size=int(1e6)):
        self.max_size = max_size  # 缓冲区的最大容量
        self.ptr = 0  # 缓冲区指针，指向下一个要存储的位置
        self.size = 0  # 缓冲区当前存储的经验数量

        # 定义存储经验数据的数组
        self.state = np.zeros((max_size, state_dim))  # 状态数据
        self.action = np.zeros((max_size, action_dim))  # 动作数据
        self.next_state = np.zeros((max_size, state_dim))  # 下一个状态数据
        self.reward = np.zeros((max_size, 1))  # 奖励数据
        self.done = np.zeros((max_size, 1))  # 终止标志数据

        self.device = device  # 存储设备（通常是 GPU 或 CPU）

    def add(self, state, action, next_state, reward, done):
        # 向缓冲区添加一条经验
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        # 更新指针和经验数量
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # 随机采样一批经验数据
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

    def save(self, save_folder):
        # 保存缓冲区数据到文件
        np.save(f"{save_folder}_state.npy", self.state[:self.size])
        np.save(f"{save_folder}_action.npy", self.action[:self.size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}_done.npy", self.done[:self.size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        # 从文件加载缓冲区数据
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
        self.reward[:self.size] = reward_buffer[:self.size]
        self.done[:self.size] = np.load(f"{save_folder}_done.npy")[:self.size]
