## 介绍
该项目每个算法文件夹都可以单独运行并测试相应算法，Building-offline-data中使用SAC在线训练交互体，之后采用最终模型模拟交互构建离线数据集以供离线算法学习。下面是离线强化学习算法的简单介绍。

### 1、BCQ (Behavioral Cloning from Offline Data with Q-Value Correction)
BCQ是一种离线行为克隆算法，从离线数据中学习策略，并通过Q值校正来改进性能。

### 2、BEAR (Bootstrapping Error Accumulation Reduction)
BEAR是一种离线强化学习算法，通过利用离线数据进行训练，并使用自举方法来减少误差积累。

### 3、TD3-BC (Twin Delayed Deep Deterministic Policy Gradient with Behavior Cloning)
TD3-BC是一种基于行为克隆的双延迟深度确定性策略梯度算法，通过结合行为克隆和双延迟DDPG来提高算法性能。

### 4、CQL (Conservative Q-Learning)
CQL是一种离线Q学习算法，通过引入保守性目标来提高离线学习的稳定性和性能。

### 5、IQL (Implicit Quantile Networks for Distributional Reinforcement Learning)
IQL是一种基于分位函数的离线强化学习算法，通过学习动作价值的分布来提高性能。

### 6、AWAC (Actor-Critic with Adversarial Weight Perturbations)
AWAC是一种离线强化学习算法，结合了确定性策略梯度和最大熵强化学习的思想，并使用生成对抗网络来提高策略学习的效果。

### 7、BC (Behavioral Cloning)
BC是一种简单的离线行为克隆算法，通过直接复制专家策略来学习行为。

### 使用说明

- **python版本：**` 3.10.13`

- **依赖库：**[requirements.txt](./requirements.txt)

- **安装依赖库：**`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/`

### 参考资料

1.  [离线强化学习系列](https://www.zhihu.com/column/c_1487193754071617536)