import gym
from TD3_BC import TD3_BC

env = gym.make('Pendulum-v1', render_mode="human")
state_dim = env.observation_space.shape[0]  # 状态空间维度
action_dim = env.action_space.shape[0]  # 动作空间维度
max_action = env.action_space.high[0]  # 动作的最大值

policy = TD3_BC(state_dim, action_dim, max_action, net_width=32, alpha=1)

policy.load('TD3_BC_Pendulum', '200')

for i in range(3):
    # 使用 TD3_BC 代理训练并获得策略值
    state, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        steps += 1
        action = policy.select_action(state)
        state, reward, done, _, _ = env.step(action)
