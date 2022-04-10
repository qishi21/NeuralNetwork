import numpy as np
import gzip
import pickle
from pathlib import Path


# 载入数据
def load_data(path):
    with gzip.open(Path(path).as_posix(), 'rb') as f:
        ((train_X, train_y), (test_X, test_y), _) = pickle.load(f, encoding='latin-1')
    return train_X, train_y, test_X, test_y


# 随机选择进行批量梯度下降的索引
def get_batch(n, batch_size):
    batch_step = np.arange(0, n, batch_size)
    indices = np.arange(n, dtype=np.int64)
    np.random.shuffle(indices)
    batches = [indices[i: i + batch_size] for i in batch_step]
    return batches


# 准确率计算
def accuracy(y_true, y_pred):
    return len(np.where(y_true==y_pred)[0]) / len(y_true)