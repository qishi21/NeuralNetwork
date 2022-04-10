# 构建两层神经网络分类器

将需要测试的数据分为 “features.npy" 和 "labels.npy" 放入 /predict_data 文件夹中，

- features.npy 存储特征，行为每个样本，列为784个特征；
- labels.npy 存储真实标签。

将所需文件放入相应的文件夹后，打开终端cd到项目根目录，之后运行：

```bash
python predict.py
```

即可输出预测的准确率。