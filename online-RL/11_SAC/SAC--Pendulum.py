import os
import gym
import numpy as np
import torch
from SAC import SAC
from utils import ReplayBuffer


def train_sac(params):
    env = gym.make(params.get('env'))
    state_dim = env.observation_space.shape[0]  # 状态空间维度
    action_dim = env.action_space.shape[0]  # 动作空间维度
    max_action = env.action_space.high[0]  # 动作的最大值

    if params.get("seed") is not None:
        torch.manual_seed(params.get("seed"))

    save_path = params.get("save_path")
    if not os.path.exists(save_path):
        os.mkdir(save_path)  # 如果保存路径不存在，创建文件夹

    model = SAC(state_dim, action_dim, max_action)  # 创建TD3模型
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e4))  # 创建经验回放缓冲
    steps = 0
    for episode in range(params.get("max_time_steps")):
        (s, _), done = env.reset(), False  # 重置环境并获取初始状态
        ep_r = 0  # 本回合的总奖励
        pred_action = 0
        for _ in range(params.get("max_episodes")):
            if replay_buffer.size > 1000:
                a = model.train_predict(np.array(s))
            else:
                a = env.action_space.sample()
            s_, r, done, _, _ = env.step(a)  # 执行动作并获取奖励等信息
            replay_buffer.add(s, a, s_, r, False)  # 添加经验到回放缓冲
            steps += 1
            if replay_buffer.size > 1000 and steps % 2 == 0:
                model.train(replay_buffer, 256)  # 当回放缓冲中的经验足够多时，开始训练模型
            s = s_  # 更新状态
            ep_r += r  # 累积奖励
            if (np.array(pred_action) == np.array(a)).any():
                break
            pred_action = a

        if not os.path.exists(save_path):
            os.mkdir(save_path)  # 如果保存路径不存在，创建文件夹

        if (episode + 1) % params.get("save_interval") == 0:
            model.save(save_path, episode + 1)  # 每隔一定回合保存模型

        print('episode:', episode, ' score:', ep_r)  # 打印回合信息

    env.close()  # 关闭环境


if __name__ == '__main__':
    # 创建命令行参数解析器
    param = {
        "env": "Pendulum-v1",  # OpenAI gym环境名称
        "seed": 42,  # 随机种子数
        "max_time_steps": 60,  # 训练次数
        "max_episodes": 500,  # 最大运行步长
        "save_interval": 30,  # 模型保存间隔
        "batch_size": 256,  # 网络训练的批量大小
        "save_path": "SAC_Pendulum",  # 模型保存路径
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # 启动 SAC 训练
    train_sac(param)
