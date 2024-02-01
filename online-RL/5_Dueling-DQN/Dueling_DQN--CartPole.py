import os
import gym
import torch
from utils import ReplayBuffer
from Dueling_DQN import DuelingDQN


def train_dueling_dqn(params):
    # 获取环境和状态/动作空间的维度信息
    env = gym.make(params.get('env'))
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    save_path = params.get("save_path")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 初始化 DuelingDQN 代理
    policy = DuelingDQN(state_dim, action_dim)

    # 初始化经验缓冲区并从文件加载数据
    replay_buffer = ReplayBuffer(state_dim, 2, params.get("device"), int(1e4))
    epsilon = 1
    # 主训练循环
    for i in range(1, int(params.get("max_time_steps")) + 1):
        # 使用 DuelingDQN 代理训练并获得策略值
        state, _ = env.reset()
        epsilon = max(epsilon * 0.99, 0.001)
        for j in range(int(params.get("max_episodes"))):
            action = policy.predict(state, epsilon)
            next_state, reward, done, _, _ = env.step(action[0])
            replay_buffer.add(state, action, next_state, reward, int(done))
            state = next_state
            if done:
                break
        else:
            print("train_final")
            policy.save(save_path, f'best')
            print("Save the model with the best results")
            break
        if replay_buffer.size > 256:
            for _ in range(32):
                policy.train(replay_buffer, batch_size=params.get("batch_size"))
        print(f"Training Iterations: {i}  Average Reward: {j}")


if __name__ == '__main__':
    # 创建命令行参数解析器
    param = {
        "env": "CartPole-v1",  # OpenAI gym环境名称
        "max_time_steps": 1000,  # 训练次数
        "max_episodes": 10000,  # 最大运行步长
        "batch_size": 256,  # 网络训练的批量大小
        "save_path": "Dueling_DQN_CartPole",  # 模型保存路径
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # 启动 DuelingDQN 训练
    train_dueling_dqn(param)
