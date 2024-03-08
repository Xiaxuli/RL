import gym
from SAC import SAC

env = gym.make('Pendulum-v1', render_mode="human")
state_dim = env.observation_space.shape[0]  # 状态空间维度
action_dim = env.action_space.shape[0]  # 动作空间维度
max_action = env.action_space.high[0]  # 动作的最大值

policy = SAC(state_dim, action_dim, max_action)
policy.load('SAC_Pendulum', '60')

for i in range(3):
    # 使用 SAC 代理训练并获得策略值
    state, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        steps += 1
        action = policy.predict(state)
        state, reward, done, _, _ = env.step(action)
