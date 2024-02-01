## 介绍
该项目每个算法文件夹都可以单独运行并测试相应算法，下面是在线强化学习算法的简单介绍。
### 1、Q-learning
Q-learning 是一种基于值迭代的强化学习算法，用于学习在不同状态下采取各种动作的值函数。

### 2、SARSA
SARSA（State-Action-Reward-State-Action）是一个在线强化学习算法，与 Q-learning 类似，但它在学习过程中使用实际采取的动作。

### 3、DQN (Deep Q-Network)
DQN 是一种基于深度学习的 Q-learning 算法，使用深度神经网络来近似值函数，提高对复杂环境的适应性。

### 4、Double-DQN
Double-DQN 是对 DQN 的改进，通过解决 DQN 中过高估计 Q 值的问题，提高了算法的性能。

### 5、Dueling-DQN
Dueling-DQN 是一种改进的 DQN 变体，将值函数分解为状态值和动作优势两个部分，提高学习的效率。

### 6、PG (Policy Gradient)
Policy Gradient 是一类基于策略优化的强化学习算法，直接优化策略参数，适用于连续动作空间。

### 7、AC (Actor-Critic)
Actor-Critic 是一种结合了策略优化和值迭代的算法，通过一个策略网络（Actor）和一个值函数网络（Critic）实现学习。

### 8、PPO (Proximal Policy Optimization)
PPO 是一种策略优化算法，通过在优化过程中引入一定的约束，确保策略更新的稳定性。

### 9、DDPG (Deep Deterministic Policy Gradient)
DDPG 是一种适用于连续动作空间的深度强化学习算法，使用深度神经网络学习确定性策略。

### 10、TD3 (Twin Delayed DDPG)
TD3 是对 DDPG 的改进，通过使用双 Q 网络和延迟更新等技术提高算法的稳定性。

### 11、SAC (Soft Actor-Critic)
SAC 是一种基于最大熵理论的策略优化算法，通过最大化环境的熵来平衡探索和利用。

### 使用说明

- **python版本：**` 3.10.13`

- **依赖库：**[requirements.txt](./requirements.txt)

- **安装依赖库：**`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/`

### 参考资料

1.  [白话强化学习](https://github.com/louisnino/RLcode)