import gym
from PG import PG

env = gym.make('CartPole-v1', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = 2
policy = PG(state_dim, action_dim, 16)
policy.load('PG_CartPole', 'best')

for i in range(3):
    # 使用 PG 代理训练并获得策略值
    state, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 1000:
        steps += 1
        action = policy.predict(state)[0]
        state, reward, done, _, _ = env.step(action)
