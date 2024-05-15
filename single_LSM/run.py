import random
from dataset import A
import matplotlib.pyplot as plt
import numpy as np


def calculate_loss(x, y, theta):
    # 该函数的功能可以计算损失函数
    # x: 模型预测值
    # y: 实际值
    res = 0
    for i in range(len(x)):
        res += (theta[1] * x[i] + theta[0] - y[i]) ** 2
    res /= 2 * len(x)
    # 返回最后的损失值
    return res


def plot_data(x, y, theta):
    # 该函数的功能是绘制数据
    # x: 模型预测值
    # y: 实际值
    # theta: 模型参数
    # 横坐标表示城市人口
    # 纵坐标表示利润
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.plot(x, y, 'ro')
    plt.plot(x, [theta[1] * i + theta[0] for i in x])
    plt.show()


def plot_loss(loss):
    # 该函数的功能是绘制损失函数
    # loss: 损失值
    # 横坐标表示迭代次数，纵坐标表示损失值
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.show()


def plot_theta(train_data, valid_data, theta):
    # 生成参数范围
    theta0_range = np.linspace(-11, 11, 100)
    theta1_range = np.linspace(-1.5, 4.5, 100)

    # 生成参数网格
    theta0, theta1 = np.meshgrid(theta0_range, theta1_range)

    # 计算损失值
    loss = np.zeros_like(theta0)
    for i in range(len(theta0_range)):
        for j in range(len(theta1_range)):
            loss[i, j] = calculate_loss(train_data, valid_data, [theta0_range[i], theta1_range[j]])

    # 绘制3D散点图
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(theta0, theta1, loss)
    ax1.set_xlabel('Theta 0')
    ax1.set_ylabel('Theta 1')
    ax1.set_zlabel('Loss')
    ax1.set_title('3D Scatter Plot')

    # 绘制2D等高线图
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(theta0, theta1, loss, 20, cmap='RdGy')
    plt.colorbar(contour, ax=ax2)
    ax2.set_xlabel('Theta 0')
    ax2.set_ylabel('Theta 1')
    ax2.set_title('Contour Plot')
    ax2.plot(theta[0], theta[1], 'ro')

    plt.tight_layout()
    plt.show()


def calculate_gradient(x, y, theta):
    # 该函数的功能可以计算梯度
    # x: 模型预测值
    # y: 实际值
    # theta: 模型参数
    res = [0, 0]
    for i in range(len(x)):
        res[0] += 2 * theta[1] * x[i] ** 2 - 2 * x[i] * y[i] + 2 * theta[0] * x[i]
        res[1] += 2 * theta[0] - 2 * y[i] + 2 * theta[1] * x[i]
    res[0] /= len(x)
    res[1] /= len(x)
    # 返回最后的梯度值
    return res


def initial_weights():
    # 该函数的功能是初始化参数
    # 生成随机数
    return [random.random(), random.random()]


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
        theta[1] -= alpha * grad[0]
        theta[0] -= alpha * grad[1]
        # 计算损失函数
        loss.append(calculate_loss(x, y, theta))
    # 绘制损失函数
    plot_loss(loss)
    return theta


# 主函数
if __name__ == "__main__":
    alpha = 0.01  # 学习率
    epochs = 1000  # 迭代次数
    theta = initial_weights()  # 初始化参数
    x = []
    y = []
    # 从数据集中提取数据
    for i in A:
        x.append(i[0])
        y.append(i[1])
    # 训练模型
    theta = train(x, y, theta, alpha, epochs)
    print("Theta: ", theta)
    plot_data(x, y, theta)
    plot_theta(x, y, theta)
