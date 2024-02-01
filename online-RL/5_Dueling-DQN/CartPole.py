import gym
from Dueling_DQN import DuelingDQN

env = gym.make('CartPole-v1', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = 2
policy = DuelingDQN(state_dim, action_dim)
policy.load('Dueling_DQN_CartPole', 'best')

for i in range(3):
    # 使用 DuelingDQN 代理训练并获得策略值
    state, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 1000:
        steps += 1
        action = policy.predict(state, 0)
        state, reward, done, _, _ = env.step(action[0])
