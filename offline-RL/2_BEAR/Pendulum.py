import gym
from BEAR import BEAR

env = gym.make('Pendulum-v1', render_mode="human")
state_dim = env.observation_space.shape[0]  # 状态空间维度
action_dim = env.action_space.shape[0]  # 动作空间维度
max_action = env.action_space.high[0]  # 动作的最大值

policy = BEAR(2, state_dim, action_dim, max_action, 3e-4, 32, 32, 32, 'cpu')

policy.load('BEAR_Pendulum', '150')

for i in range(3):
    # 使用 BEAR 代理训练并获得策略值
    state, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        steps += 1
        action = policy.select_action(state)
        state, reward, done, _, _ = env.step(action)
