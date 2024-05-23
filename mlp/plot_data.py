import matplotlib.pyplot as plt
from dataset import dataset, label
import numpy as np


def random_plot_25():
    # 从dataset里随机选择25个数据并且展示
    idx = np.random.choice(len(dataset), 25)
    data_to_plot = []
    for _ in dataset:
        # 将400维数据转换为20*20的矩阵
        pic = np.reshape(_, (20, 20))
        # 将矩阵转置
        pic = pic.T
        data_to_plot.append(pic)
    fig, ax = plt.subplots(5, 5, sharex=True, sharey=True)
    for i in range(5):
        for j in range(5):
            ax[i, j].matshow(data_to_plot[idx[i * 5 + j]], cmap='gray')
    plt.show()


def plot_label_data():
    fig, ax = plt.subplots(4, 3, sharex=True, sharey=True)
    data_to_plot = []
    # 从各个标签里取一张图片绘制
    for i in range(10):
        # 查找index为i的位置
        for ind in range(len(label)):
            if np.argmax(label[ind]) == i:
                pic = dataset[ind]
                pic = np.reshape(pic, (20, 20))
                # 将矩阵转置
                pic = pic.T
                data_to_plot.append(pic)
                ax[i // 3, i % 3].matshow(data_to_plot[i], cmap='gray')
                break
    plt.show()


# random_plot_25()
plot_label_data()
