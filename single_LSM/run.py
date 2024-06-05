"""
Author: caodong0225
Date: 2024-06-04
Description: 该文件的功能是训练模型
"""
import random
import matplotlib.pyplot as plt
import numpy as np
from dataset import A


def calculate_loss(model_predict, actual_values, model_values):
    """
    该函数的功能可以计算损失函数
    :param model_predict: 模型预测值
    :param actual_values: 实际值
    :param model_values: 模型参数
    """
    res = 0
    for index, prediction in enumerate(model_predict):
        res += (model_values[1] * prediction + model_values[0] - actual_values[index]) ** 2
    res /= 2 * len(model_predict)
    # 返回最后的损失值
    return res


def plot_data(model_predict, actual_values, model_values):
    """
    该函数的功能是绘制数据
    :param model_predict: 模型预测值
    :param actual_values: 实际值
    :param model_values: 模型参数
    """
    # 横坐标表示城市人口
    # 纵坐标表示利润
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.plot(model_predict, actual_values, 'ro')
    plt.plot(model_predict, [model_values[1] * index + model_values[0] for index in model_predict])
    plt.show()


def plot_loss(loss):
    """
    该函数的功能是绘制损失函数
    :param loss: 损失值
    :return:
    """
    # 横坐标表示迭代次数，纵坐标表示损失值
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.show()


def plot_theta(train_data, valid_data, model_value):
    """
    该函数的功能是绘制参数空间
    :param train_data: 训练数据
    :param valid_data: 验证数据
    :param model_value: 模型参数
    :return:
    """
    # 生成参数范围
    theta0_range = np.linspace(-11, 11, 100)
    theta1_range = np.linspace(-1.5, 4.5, 100)

    # 生成参数网格
    theta0, theta1 = np.meshgrid(theta0_range, theta1_range)

    # 计算损失值
    loss = np.zeros_like(theta0)
    for index0, theta0_ in enumerate(theta0_range):
        for index1, theta1_ in enumerate(theta1_range):
            loss[index0, index1] = calculate_loss(train_data, valid_data, [theta0_, theta1_])
    # 绘制3D散点图
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(theta0, theta1, loss)
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Loss')
    ax.set_title('3D Scatter Plot')

    # 绘制2D等高线图
    ax = fig.add_subplot(122)
    contour = ax.contourf(theta0, theta1, loss, 20, cmap='RdGy')
    plt.colorbar(contour, ax=ax)
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_title('Contour Plot')
    ax.plot(model_value[0], model_value[1], 'ro')

    plt.tight_layout()
    plt.show()


def calculate_gradient(model_predict, actual_values, model_values):
    """
    该函数的功能可以计算梯度
    :param model_predict: 模型预测值
    :param actual_values: 实际值
    :param model_values: 模型参数
    :return: res: 梯度值
    """
    res = [0, 0]
    for index, prediction in enumerate(model_predict):
        res[0] += (2 * model_values[1] * prediction ** 2 -
                   2 * prediction * actual_values[index]
                   + 2 * model_values[0] * prediction)
        res[1] += 2 * model_values[0] - 2 * actual_values[index] + 2 * model_values[1] * prediction
    res[0] /= len(model_predict)
    res[1] /= len(model_predict)
    # 返回最后的梯度值
    return res


def initial_weights():
    """
    该函数的功能是初始化参数
    :return:
    """
    return [random.random(), random.random()]


def train(model_predict, actual_values, model_values, learning_rate, training_epochs):
    """
    该函数的功能是训练模型
    :param model_predict: 模型预测值
    :param actual_values: 实际值
    :param model_values: 模型参数
    :param learning_rate: 学习率
    :param training_epochs: 迭代次数
    """
    loss = []
    for _ in range(training_epochs):
        # 计算梯度
        grad = calculate_gradient(model_predict, actual_values, model_values)
        # 更新参数
        model_values[1] -= learning_rate * grad[0]
        model_values[0] -= learning_rate * grad[1]
        # 计算损失函数
        loss.append(calculate_loss(model_predict, actual_values, model_values))
    # 绘制损失函数
    plot_loss(loss)
    return model_values


# 主函数
if __name__ == "__main__":
    ALPHA = 0.01  # 学习率
    EPOCHS = 1000  # 迭代次数
    theta = initial_weights()  # 初始化参数
    x = []
    y = []
    # 从数据集中提取数据
    for i in A:
        x.append(i[0])
        y.append(i[1])
    # 训练模型
    theta = train(x, y, theta, ALPHA, EPOCHS)
    print("Theta: ", theta)
    plot_data(x, y, theta)
    plot_theta(x, y, theta)
