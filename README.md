## 强化学习算法

### online-RL

该部分包含了各种在线强化学习算法的实现，每个算法文件夹都可以独立运行并测试相应算法在不同环境中的性能。以下是各个算法的简单介绍：

1. **Q-learning**: 基于值迭代的算法，用于学习在不同状态下采取各种动作的值函数。
2. **SARSA**: 在线强化学习算法，与 Q-learning 类似，但在学习过程中使用实际采取的动作。
3. **DQN (Deep Q-Network)**: 使用深度学习的 Q-learning 变体，通过深度神经网络来近似值函数。
4. **Double-DQN**: 对 DQN 的改进，解决了 Q 值过高估计的问题，提高了算法性能。
5. **Dueling-DQN**: DQN 的改进，将值函数分解为状态值和动作优势两个部分，提高学习效率。
6. **PG (Policy Gradient)**: 基于策略优化的强化学习算法，适用于连续动作空间。
7. **AC (Actor-Critic)**: 结合了策略优化和值迭代，通过 Actor 和 Critic 实现学习。
8. **PPO (Proximal Policy Optimization)**: 策略优化算法，通过引入约束确保策略更新的稳定性。
9. **DDPG (Deep Deterministic Policy Gradient)**: 适用于连续动作空间的深度强化学习算法。
10. **TD3 (Twin Delayed DDPG)**: DDPG 的改进，使用双 Q 网络和延迟更新提高算法稳定性。
11. **SAC (Soft Actor-Critic)**: 基于最大熵理论的策略优化算法，平衡探索和利用。

### offline-RL

该部分包含了各种离线强化学习算法的实现，每个算法文件夹都可以独立运行并测试相应算法在不同环境中的性能。以下是各个算法的简单介绍：

1. **BCQ（Behavioral Cloning from Offline Data with Q-Value Correction）**: 一种离线行为克隆算法，从离线数据中学习策略，并通过 Q 值校正来改进性能。
2. **BEAR（Bootstrapping Error Accumulation Reduction）**: 一种离线强化学习算法，通过利用离线数据进行训练，并使用自举方法来减少误差积累。
3. **TD3-BC（Twin Delayed Deep Deterministic Policy Gradient with Behavior Cloning）**: 一种基于行为克隆的双延迟深度确定性策略梯度算法，通过结合行为克隆和双延迟 DDPG 来提高算法性能。
4. **CQL（Conservative Q-Learning）**: 一种离线 Q 学习算法，通过引入保守性目标来提高离线学习的稳定性和性能。
5. **IQL（Implicit Quantile Networks for Distributional Reinforcement Learning）**: 一种基于分位函数的离线强化学习算法，通过学习动作价值的分布来提高性能。
6. **AWAC（Actor-Critic with Adversarial Weight Perturbations）**: 一种离线强化学习算法，结合了确定性策略梯度和最大熵强化学习的思想，并使用生成对抗网络来提高策略学习的效果。
7. **BC（Behavioral Cloning）**: 一种简单的离线行为克隆算法，通过直接复制专家策略来学习行为。
