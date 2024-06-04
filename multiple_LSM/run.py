"""
Author: caodong0225
Date: 2024-06-04
Description: 多元线性回归
"""
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from dataset import A


def initial_value(data_dimension):
    """
    :param data_dimension: 参数维度
    :return:
    """
    # 该函数的功能是初始化参数
    # 生成随机数
    return [random.random() for _ in range(data_dimension)]


def calculate_single_predict(training_set, model_theta):
    """
    该函数的功能是计算单个预测值
    :param training_set:  训练集
    :param model_theta:  模型参数
    :return:
    """
    return model_theta[0] + sum(training_set[_] *
                                model_theta[_ + 1]
                                for _ in range(len(training_set)))


def calculate_loss(training_set, actual_values, model_theta):
    """
    该函数的功能是计算损失函数
    :param training_set:  训练集
    :param actual_values:  实际值
    :param model_theta:  模型参数
    :return:
    """
    return sum((calculate_single_predict(training_set[_], model_theta)
                - actual_values[_]) ** 2 for _ in
               range(len(training_set))) / len(training_set)


def calculate_gradient(training_set, actual_values, model_theta):
    """
    该函数的功能是计算梯度
    :param training_set:  训练集
    :param actual_values:  实际值
    :param model_theta:  模型参数
    :return:
    """
    res = [0 for _ in range(len(model_theta))]
    for index, training_set_ in enumerate(training_set):
        # 计算预测值
        res[0] += 2 * (calculate_single_predict(training_set_, model_theta)
                       - actual_values[index])
    for index0 in range(len(model_theta) - 1):
        for index1, training_set_ in enumerate(training_set):
            # 计算预测值
            res[index0 + 1] += 2 * training_set_[index0] * (
                    calculate_single_predict(training_set_, model_theta)
                    - actual_values[index1])
    for index0 in range(len(model_theta)):
        res[index0] /= len(training_set)
    # 返回最后的梯度值
    return res


def plot_loss(loss):
    """
    该函数的功能是绘制损失函数
    :param loss:
    :return:  None
    """
    # 横坐标表示迭代次数，纵坐标表示损失值
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.show()


def train(predict_values, actual_values, model_theta, alpha, epochs):
    """
    该函数的功能是训练模型
    :param predict_values:  模型预测值
    :param actual_values:  实际值
    :param model_theta:  模型参数
    :param alpha:  学习率
    :param epochs:  迭代次数
    :return:
    """
    loss = []
    for _ in range(epochs):
        # 计算梯度
        grad = calculate_gradient(predict_values, actual_values, model_theta)
        # 更新参数
        for index, _ in enumerate(model_theta):
            model_theta[index] -= alpha * grad[index]
        loss.append(calculate_loss(predict_values, actual_values, model_theta))
    plot_loss(loss)
    return model_theta


def plot_model(model_theta, a_):
    """
    该函数的功能是绘制模型
    :param model_theta:
    :param a_:
    :return:
    """
    points = np.array(a_)
    # 创建数据
    x_ = np.linspace(1000, 5000, 100)
    y_ = np.linspace(0, 5, 100)
    x_, y_ = np.meshgrid(x_, y_)
    z = (model_theta[1] * x_
         + model_theta[2] * y_
         + model_theta[0])  # 将平面方程改写为z的形式

    # 创建一个三维图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维平面
    ax.plot_surface(x_, y_, z, color='b', alpha=0.6)

    # 绘制三维空间中的点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', s=50)

    # 设置轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 创建动画
    FuncAnimation(fig, lambda _: ax.view_init(elev=20, azim=_),
                  frames=np.arange(0, 360, 1), interval=50)

    # 显示图形
    plt.show()


if __name__ == "__main__":
    ALPHA = 0.00000001  # 学习率
    EPOCHS = 100  # 迭代次数
    DIMENSION = 3  # 参数维度
    theta = initial_value(DIMENSION)  # 初始化参数
    x = []
    y = []
    # 从数据集中提取数据
    for i in A:
        x.append(i[:DIMENSION - 1])
        y.append(i[-1])
    # 训练模型
    theta = train(x, y, theta, ALPHA, EPOCHS)
    plot_model(theta, A)
