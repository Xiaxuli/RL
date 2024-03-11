import os
import gym
from SAC import SAC
from utils import ReplayBuffer


def start_game(envs, model, replay_buffers):
    # 重置环境
    state, _ = envs.reset()
    for i in range(200):
        action = model.predict(state)

        # 执行动作并获取下一个状态、奖励和结束标志
        next_state, reward, done, _, _ = envs.step(action)
        replay_buffers.add(state, action, next_state, reward, int(done))
        state = next_state


# 创建游戏环境
env = gym.make('Pendulum-v1')

# 获取环境和状态/动作空间的维度信息
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

# 创建经验回放缓冲区
replay_buffer = ReplayBuffer(state_dim, action_dim)
load_path = "SAC_Pendulum"
save_path = "Pendulum_buffers2/"

# 如果保存路径不存在，创建文件夹
if not os.path.exists(save_path):
    os.mkdir(save_path)

# 加载策略
policy = SAC(state_dim, action_dim, max_action)
policy.load(load_path, 60)

# 交互构建连线数据集
while replay_buffer.size < 1e5:
    start_game(env, policy, replay_buffer)

# 保存数据集
replay_buffer.save(save_path + 'sac')
