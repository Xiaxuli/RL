import os
import gym
import torch
import numpy as np
from AWAC import AWAC
from utils import ReplayBuffer


# 定义训练 AWAC 代理的函数
def train_awac(params):
    device = params.get("device")
    if params.get("seed"):
        torch.manual_seed(params.get("seed"))
        np.random.seed(params.get("seed"))

    # 获取环境和状态/动作空间的维度信息
    env = gym.make(params.get("env"))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    save_path = params.get("save_path")

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 初始化 AWAC 代理
    policy = AWAC(state_dim, action_dim, max_action, hidden_size=32)

    # 初始化经验缓冲区并从文件加载数据
    replay_buffer = ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(params.get("buffer_name"))

    # 主训练循环
    for episode in range(int(params.get("max_timestep"))):
        # 使用 IQL 代理训练并获得策略值
        for _ in range(int(params.get("eval_freq"))):
            policy.train(replay_buffer, batch_size=params.get("batch_size"))
        # 模型评估
        eval_policy(policy, params.get("env"), episode + 1)
        # 每50次迭代保存一次模型
        if (episode + 1) % 50 == 0:
            policy.save(save_path, episode + 1)
            print(f"training_iters: {episode + 1} Save to {save_path}")


# 定义评估 AWAC 代理性能的函数
def eval_policy(policy, env_name, training_iters, eval_episodes=3):
    eval_env = gym.make(env_name)
    avg_reward = 0.

    # 执行多次评估
    for _ in range(eval_episodes):
        (state, _), done = eval_env.reset(), False
        step = 0
        while not done:
            # 根据 AWAC 代理选择动作并与环境交互
            action = policy.predict(np.array(state), train=False)
            state, reward, done, _, _ = eval_env.step(action)
            avg_reward += reward
            step += 1
            # 最大步长500
            if step > 500:
                break
    avg_reward /= eval_episodes

    # 输出评估结果
    print(f"Training Iterations: {training_iters}  Average Reward: {avg_reward:.2f}")
    return avg_reward


param = {
    "env": "Pendulum-v1",  # OpenAI gym环境名称
    "seed": 42,  # 设置 Gym、PyTorch 和 Numpy 的随机种子
    "buffer_name": "./Pendulum_buffers/sac",  # 离线数据集的位置
    "max_timestep": 200,  # 运行环境或训练所需的最大时间步长(定义缓冲区大小)
    "eval_freq": 200,  # 评估频率（时间步数）
    "batch_size": 256,  # 网络训练的批量大小
    "save_path": "AWAC_Pendulum",  # 模型保存路径
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# 启动 AWAC 训练
train_awac(param)
