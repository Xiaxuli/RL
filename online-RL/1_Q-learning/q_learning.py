import gym
import numpy as np


class QLearning:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.9, e_greedy=0.1):
        # 初始化 1_Q-learning 智能体的参数和 Q 表
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.e_greedy = e_greedy
        self.Q = np.zeros([state_size, action_size])

    def sample(self, state):
        # epsilon-greedy 策略选择动作
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

    def learn(self, state, action, reward, next_state, done):
        # 1_Q-learning 更新规则
        target = reward + int(1 - done) * self.gamma * np.max(self.Q[next_state, :])
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def run_episode(self, env):
        # 运行一个episode，更新Q值
        (state, _), done = env.reset(), False
        reward_count = 0
        while not done:
            action = self.sample(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = -1 if done and reward != 1 else reward
            self.learn(state, action, reward, next_state, done)
            reward_count += reward
            state = next_state
        return reward_count


def play_game(env_name):
    # 初始化训练环境和 1_Q-learning 智能体
    env_train = gym.make(env_name)
    policy = QLearning(env_train.observation_space.n, env_train.action_space.n)
    rewards = 0

    # 训练智能体
    for i in range(2000):
        reward_now = policy.run_episode(env_train)
        rewards = rewards * 0.99 + max(0, reward_now) * 0.01
        print(f"第{i + 1}次，奖励为:", reward_now, "通关率为:", rewards)

    # 在图形界面中展示智能体的表现
    env_test = gym.make(env_name, render_mode="human")
    (states, _), dones = env_test.reset(), False
    while not dones:
        actions = policy.predict(states)
        states, _, dones, _, _ = env_test.step(actions)
    env_test.close()


if __name__ == '__main__':
    play_game('FrozenLake-v1')
