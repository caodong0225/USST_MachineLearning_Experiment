import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from dataset import A
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def initial_value(dimension):
    # 该函数的功能是初始化参数
    # 生成随机数
    # dimension: 参数维度
    return [random.random() for _ in range(dimension)]


def calculate_single_predict(x, theta):
    # 该函数的功能是计算单个预测值
    # x: 模型的训练集
    # theta: 模型参数
    return theta[0] + sum([x[i] * theta[i + 1] for i in range(len(x))])


def calculate_loss(x, y, theta):
    # 该函数的功能是计算损失函数
    # x: 模型的训练集
    # y: 实际值
    # theta: 模型参数
    return sum([(calculate_single_predict(x[i], theta) - y[i]) ** 2 for i in range(len(x))]) / len(x)


def calculate_gradient(x, y, theta):
    # 该函数的功能可以计算梯度
    # x: 模型的训练集
    # y: 实际值
    # theta: 模型参数
    res = [0 for _ in range(len(theta))]
    for j in range(len(x)):
        # 计算预测值
        res[0] += 2 * (calculate_single_predict(x[j], theta) - y[j])
    for i in range(len(theta) - 1):
        for j in range(len(x)):
            # 计算预测值
            res[i + 1] += 2 * x[j][i] * (calculate_single_predict(x[j], theta) - y[j])
    for i in range(len(theta)):
        res[i] /= len(x)
    # 返回最后的梯度值
    return res


def plot_loss(loss):
    # 该函数的功能是绘制损失函数
    # loss: 损失值
    # 横坐标表示迭代次数，纵坐标表示损失值
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.show()


def train(x, y, theta, alpha, epochs):
    # 该函数的功能是训练模型
    # x: 模型预测值
    # y: 实际值
    # theta: 模型参数
    # alpha: 学习率
    # epochs: 迭代次数
    loss = []
    for _ in range(epochs):
        # 计算梯度
        grad = calculate_gradient(x, y, theta)
        # 更新参数
        for i in range(len(theta)):
            theta[i] -= alpha * grad[i]
        loss.append(calculate_loss(x, y, theta))
    plot_loss(loss)
    return theta


def plot_model(theta, A):
    points = np.array(A)
    # 创建数据
    x = np.linspace(1000, 5000, 100)
    y = np.linspace(0, 5, 100)
    x, y = np.meshgrid(x, y)
    z = (theta[1] * x + theta[2] * y + theta[0])  # 将平面方程改写为z的形式

    # 创建一个三维图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维平面
    ax.plot_surface(x, y, z, color='b', alpha=0.6)

    # 绘制三维空间中的点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', s=50)

    # 设置轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 创建动画
    FuncAnimation(fig, lambda i: ax.view_init(elev=20, azim=i), frames=np.arange(0, 360, 1), interval=50)

    # 显示图形
    plt.show()


if __name__ == "__main__":
    alpha = 0.00000001  # 学习率
    epochs = 100  # 迭代次数
    dimension = 3  # 参数维度
    theta = initial_value(dimension)  # 初始化参数
    x = []
    y = []
    # 从数据集中提取数据
    for i in A:
        x.append(i[:dimension - 1])
        y.append(i[-1])
    # 训练模型
    theta = train(x, y, theta, alpha, epochs)
    plot_model(theta, A)
