import numpy as np
import os
from pathlib import Path

from model.NN import NeuralNetwork
from common.dataprocessing import load_data, get_batch, accuracy
from common.plot import plot_loss, plot_acc
from common.utils import save_metrics


# 配置参数
class Config:
    def __init__(self):
        self.seed = 1 # numpy的随机数种子
        self.epoches = 200 # 训练轮次
        self.batch_size = 64 # 批量梯度下降中每批数量

        self.lr_start = 0.1 # 学习率的初始值
        self.hidden_dim = 256 # 隐藏层层数
        self.regularization_intensity = 0.005 # 正则化强度


# 训练函数
def train(cfg):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    model_path = f'{curr_path}/output/lr{cfg.lr_start}_hd{cfg.hidden_dim}_ri{cfg.regularization_intensity}/'

    np.random.seed(cfg.seed)
    train_X, train_y, test_X, test_y = load_data('./data/mnist.pkl.gz')
    n, input_dim = train_X.shape
    model = NeuralNetwork(input_dim, cfg.hidden_dim, 10, cfg.regularization_intensity)
    loss_train_total = []
    loss_test_total = []
    acc_total = []
    print(f'学习率起始值:{cfg.lr_start}, 隐藏层层数:{cfg.hidden_dim}, 正则化强度:{cfg.regularization_intensity}')
    for epoch in range(cfg.epoches):
        batch_indices = get_batch(len(test_X), cfg.batch_size)
        batch_num = 0
        loss_epoch = 0

        for batch in batch_indices:
            batch_num += 1
            train_X_batch = train_X[batch]
            model(train_X_batch)
            y_true = train_y[batch]
            loss_batch = float(model.loss(y_true))
            loss_epoch += 1 / len(batch_indices) * (loss_batch - loss_epoch)
            model.backward(cfg.lr_start, epoch)

        loss_train_total.append(loss_epoch)

        test_y_predict = model(test_X)
        test_loss = model.loss(test_y)
        loss_test_total.append(test_loss)

        acc = accuracy(test_y, test_y_predict)
        acc_total.append(acc)

        print(f'epoch:{epoch+1}/{cfg.epoches}\t   train_loss:{round(loss_epoch, 2)}\t  test_loss:{round(test_loss, 2)}\t acc:{np.round(acc*100, 2)}%.')

    Path(model_path).mkdir(parents=True, exist_ok=True) # 创建文件夹
    plot_loss(model_path, loss_train_total, loss_test_total) # 绘制loss曲线
    plot_acc(model_path, acc_total) # 绘制accuracy曲线
    save_metrics(model_path, loss_train_total, loss_test_total, acc_total) # 保存loss, accuracy 在每个epoch的值
    model.save(model_path+'parameters') # 保存模型参数

    print('='*25,'模型保存完成', '='*25)


if __name__ == '__main__':
    hyperparameters = ['lr', 'hidden_dim', 'regularization_intensity'] # 三个需要进行网格搜索的超参数

    grid = {
        'lr': [0.1, 0.01, 0.001], # 学习率
        'hidden_dim': [50, 100, 200], # 隐藏层
        'regularization_intensity': [0.1, 0.01, 0.001] # 正则化强度
    }

    # 进行 grid search
    for lr_start in grid['lr']:
        for hidden_dim in grid['hidden_dim']:
            for regularization_intensity in grid['regularization_intensity']:
                cfg = Config()
                cfg.lr_start = lr_start
                cfg.hidden_dim = hidden_dim
                cfg.regularization_intensity = regularization_intensity
                train(cfg)
