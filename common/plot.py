import matplotlib.pyplot as plt


# 绘制手写数字的灰度图
def show_image(data, index):
    plt.imshow(data.reshape((28, 28)), cmap='gray')
    plt.savefig(f'./images/{index}.jpg')


# 绘制 Loss 曲线
def plot_loss(path, loss_train, loss_test):
    plt.figure(dpi=150)
    plt.title('Loss Curve')
    plt.plot(loss_train)
    plt.plot(loss_test)
    plt.legend(['train', 'test'])
    plt.savefig(path+'LossCurve.jpg')


# 绘制 Accuracy 曲线
def plot_acc(path, acc):
    plt.figure(dpi=150)
    plt.title('Accuracy Curve')
    plt.plot(acc)
    plt.savefig(path+'AccCurve.jpg')
