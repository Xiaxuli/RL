import os
import gym
import torch
from PPO import PPO


def train_ppo(params):
    env = gym.make(params.get('env'))
    state_dim = env.observation_space.shape[0]  # 状态空间维度
    action_dim = env.action_space.shape[0]  # 动作空间维度
    max_action = env.action_space.high[0]  # 动作的最大值

    if params.get("seed") is not None:
        torch.manual_seed(params.get("seed"))

    save_path = params.get("save_path")

    # 初始化 PPO 代理
    model = PPO(state_dim, action_dim, max_action)

    for episode in range(1, int(params.get("max_time_steps")) + 1):
        step = 0
        (s, _), done = env.reset(), False  # 重置环境并获取初始状态
        ep_r = 0  # 本回合的总奖励
        for _ in range(params.get("max_episodes")):
            a = model.predict(s)[0]
            s_, r, done, _, _ = env.step(a)  # 执行动作并获取奖励等信息
            model.buffer_add(s, a, (r + 6) / 10)
            s = s_  # 更新状态
            ep_r += r  # 累积奖励
            if (step + 1) % 32 == 0 or step == params.get("max_episodes") - 1:
                model.train(s, done, 10)
            step += 1
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
        "max_time_steps": 200,  # 训练次数
        "max_episodes": 500,  # 最大运行步长
        "save_interval": 100,  # 模型保存间隔
        "save_path": "PPO_Pendulum",  # 模型保存路径
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # 启动 PPO 训练
    train_ppo(param)
