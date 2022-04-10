# 神经网络与深度学习第一次作业

本次实验所采用的激活函数为Sigmoid函数和Softmax函数，在隐藏层使用Sigmoid函数，输出层使用Softmax函数。

采用的Loss计算方式为交叉熵。

采用的学习率下降策略为指数下降策略

优化器采用的小批量随机梯度下降策略，每批的数量设为64.

训练的总轮次(epoch)设为200.

| 学习率初值 | 隐藏层神经单元数量 | 正则化强度 |
| ---------- | ------------------ | ---------- |
| 0.1        | 50                 | 0.1        |
| 0.01       | 100                | 0.01       |
| 0.001      | 200                | 0.001      |
