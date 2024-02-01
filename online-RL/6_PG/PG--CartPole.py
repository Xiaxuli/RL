import os
import gym
import torch
from PG import PG


def train_pg(params):
    # 获取环境和状态/动作空间的维度信息
    env = gym.make(params.get('env'))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    save_path = params.get("save_path")

    # 初始化 PG 代理
    policy = PG(state_dim, action_dim, 16)
    for i in range(1, int(params.get("max_time_steps")) + 1):
        state_lis, action_lis, reward_lis = [], [], []
        (state, _), done = env.reset(), False
        steps = 0
        while not done and steps < params.get("max_episodes"):
            steps += 1
            action = policy.predict(state)[0]
            # 互动反馈
            next_state, reward, done, _, _ = env.step(action)
            # 添加s,a,r到列表中
            state_lis.append(state)
            action_lis.append(action)
            reward_lis.append(reward)
            state = next_state
            if done:
                for _ in range(10):
                    policy.train(state_lis, action_lis, reward_lis)
        print(f"第{i}次训练，持续了{steps}次")

        if steps >= 5000:
            print("train_final")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            policy.save(save_path, f'best')
            print("Save the model with the best results")
            break


if __name__ == '__main__':
    # 创建命令行参数解析器
    param = {
        "env": "CartPole-v1",  # OpenAI gym环境名称
        "max_time_steps": 1000,  # 训练次数
        "max_episodes": 5000,  # 最大运行步长
        "save_path": "PG_CartPole",  # 模型保存路径
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # 启动 PG 训练
    train_pg(param)
