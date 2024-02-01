import gym
import numpy as np


class SARSA:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.9, e_greedy=0.1):
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.e_greedy = e_greedy
        self.Q = np.zeros([state_size, action_size])

    def sample(self, state):
        # epsilon-greedy策略选择动作
        if np.random.uniform() < self.e_greedy:
            action = np.random.choice(self.action_size)
        else:
            action = self.predict(state)
        return action

    def predict(self, state):
        # 根据当前状态选择动作，如果有多个最大值，随机选择其中一个
        all_actions = self.Q[state, :]
        max_action = np.max(all_actions)
        max_action_list = np.where(all_actions == max_action)[0]
        action = np.random.choice(max_action_list)
        return action

    def learn(self, state, action, reward, next_state, next_action, done):
        # SARSA更新规则
        target = reward + int(1 - done) * self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def run_episode(self, env):
        # 运行一个episode，更新Q值
        (state, _), done = env.reset(), False
        reward_count = 0
        action = self.sample(state)
        while not done:
            next_state, reward, done, _, _ = env.step(action)
            reward = -1 if done and reward != 1 else reward
            next_action = self.sample(next_state)
            self.learn(state, action, reward, next_state, next_action, done)
            reward_count += reward
            state = next_state
            action = next_action
        return reward_count


def play_game(env_name):
    # 初始化环境和SARSA智能体
    env1 = gym.make(env_name)
    policy = SARSA(env1.observation_space.n, env1.action_space.n)
    rewards = 0

    # 训练智能体
    for i in range(2000):
        reward_now = policy.run_episode(env1)
        rewards = rewards * 0.99 + max(0, reward_now) * 0.01
        print(f"第{i + 1}次，奖励为:", reward_now, "通关率为:", rewards)

    # 在图形界面中展示智能体的表现
    env2 = gym.make(env_name, render_mode="human")
    (states, _), dones = env2.reset(), False
    while not dones:
        actions = policy.predict(states)
        states, _, dones, _, _ = env2.step(actions)
    env2.close()


if __name__ == '__main__':
    play_game('FrozenLake-v1')
