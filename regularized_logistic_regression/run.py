import random
import math
import matplotlib.pyplot as plt
import numpy as np

from dataset import data


def map_feature(x, dimension):
    # 扩充数据集
    x_new = []
    for i in range(0, dimension + 1):
        for j in range(i + 1):
            x_new.append(x[0] ** (i - j) * x[1] ** j)
    return x_new


def initial_values(dimension):
    # 初始化参数
    theta = []
    for i in range((dimension + 2) * (dimension + 1) // 2):
        theta.append(random.random())
    return theta


def sigmoid(theta, x):
    # 该函数的功能是计算sigmoid函数
    z = 0
    for i in range(len(theta)):
        z += theta[i] * x[i]
    return 1 / (1 + math.exp(-z))


def plot_loss(loss):
    # 该函数的功能是绘制损失函数
    # loss: 损失值
    # 横坐标表示迭代次数，纵坐标表示损失值
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.show()


def calculate_loss(x, y, theta, regularization_parameter):
    # 计算交叉熵损失函数
    loss = 0
    regularization_loss = sum([theta[j] ** 2 for j in range(len(theta))]) * regularization_parameter / 2
    for i in range(len(x)):
        h = sigmoid(theta, x[i])
        # 检查参数是否为正数
        if h <= 0 or h >= 1:
            # 如果参数不是正数，手动处理
            loss += 0  # 或者设置为其他默认值
        else:
            loss += -y[i] * math.log(h) - (1 - y[i]) * math.log(1 - h)
    return (loss + regularization_loss) / len(x)


def calculate_decision_boundary(x, y, theta, dimension):
    # 该函数的功能是计算决策边界
    # x: 模型预测值
    # y: 实际值
    # theta: 模型参数
    res = np.zeros((len(x), len(y)))
    index = 0
    for i in range(0, dimension + 1):
        for j in range(i + 1):
            res_tem = np.multiply(np.power(x, i - j), np.power(y, j))
            res_tem = np.multiply(res_tem, theta[index])
            res += res_tem
            index += 1
    return res


def calculate_gradient(x, y, theta, regularization_parameter):
    # 该函数的功能可以计算梯度
    # x: 模型预测值
    # y: 实际值
    # theta: 模型参数
    res = [0] * len(theta)
    for i in range(len(x)):
        h = sigmoid(theta, x[i])
        for j in range(len(theta)):
            res[j] += (h - y[i]) * x[i][j]
    for j in range(len(theta)):
        res[j] = (res[j] + regularization_parameter * theta[j]) / len(x)
    return res


def train(x, y, theta, alpha, epochs, regularization_parameter):
    # 该函数的功能是训练模型
    loss = []
    for _ in range(epochs):
        # 计算梯度
        grad = calculate_gradient(x, y, theta, regularization_parameter)
        for _ in range(len(theta)):
            theta[_] -= alpha * grad[_]
        loss.append(calculate_loss(x, y, theta, regularization_parameter))
    plot_loss(loss)
    return theta


def plot_result(data, theta):
    admitted_data = [(_[0], _[1]) for _ in data if _[2] == 1]
    not_admitted_data = [(_[0], _[1]) for _ in data if _[2] == 0]

    # 绘制数据
    plt.scatter(*zip(*admitted_data), c="blue", marker="o", label="y = 1")
    plt.scatter(*zip(*not_admitted_data), c="red", marker="x", label="y = 0")

    # 绘制决策边界
    x = np.arange(-1, 1.5, 0.01)
    y = np.arange(-1, 1.5, 0.01)
    x, y = np.meshgrid(x, y)
    z = calculate_decision_boundary(x, y, theta, dimension)
    # 转化为网格
    plt.contour(x, y, z, 0)

    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    # 显示图例
    plt.legend()
    plt.show()


if __name__ == "__main__":
    alpha = 0.04  # 学习率
    epochs = 5000  # 迭代次数
    dimension = 6  # 数据维度
    regularization_parameter = 0  # 正则惩罚系数
    theta = initial_values(dimension)  # 初始化参数
    x = []
    y = []
    for _ in data:
        x.append(map_feature(_[:2], dimension))
        y.append(_[-1])
    theta = train(x, y, theta, alpha, epochs, regularization_parameter)
    plot_result(data, theta)
