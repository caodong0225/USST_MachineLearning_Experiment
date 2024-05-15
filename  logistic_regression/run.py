# 实现逻辑回归
import random
import math
from dataset import data
import matplotlib.pyplot as plt


def initial_weights():
    # 该函数的功能是初始化参数
    # 生成随机数
    # return [random.random(), random.random(), random.random()]
    return [-6.5, 0, 0]


def sigmoid(theta, x):
    # 该函数的功能是计算sigmoid函数
    return 1 / (1 + math.exp(-(theta[0] + theta[1] * x[0] + theta[2] * x[1])))


def calculate_loss(x, y, theta):
    # 计算交叉熵损失函数
    loss = 0
    for i in range(len(x)):
        h = sigmoid(theta, x[i])
        # 检查参数是否为正数
        if h <= 0 or h >= 1:
            # 如果参数不是正数，手动处理
            loss += 0  # 或者设置为其他默认值
        else:
            loss += -y[i] * math.log(h) - (1 - y[i]) * math.log(1 - h)
    return loss / len(x)


def calculate_gradient(x, y, theta):
    # 该函数的功能可以计算梯度
    # x: 模型预测值
    # y: 实际值
    # theta: 模型参数
    res = [0, 0, 0]
    for i in range(len(x)):
        res[0] += sigmoid(theta, x[i]) - y[i]
        res[1] += (sigmoid(theta, x[i]) - y[i]) * x[i][0]
        res[2] += (sigmoid(theta, x[i]) - y[i]) * x[i][1]
    res[0] /= len(x)
    res[1] /= len(x)
    res[2] /= len(x)
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
    loss = []
    for _ in range(epochs):
        # 计算梯度
        grad = calculate_gradient(x, y, theta)
        for _ in range(len(theta)):
            theta[_] -= alpha * grad[_]
        loss.append(calculate_loss(x, y, theta))
    plot_loss(loss)
    return theta


def plot_result(data, theta):
    admitted_data = [(_[0], _[1]) for _ in data if _[2] == 1]
    not_admitted_data = [(_[0], _[1]) for _ in data if _[2] == 0]

    # 绘制数据
    plt.scatter(*zip(*admitted_data), c="blue", marker="o", label="Admitted")
    plt.scatter(*zip(*not_admitted_data), c="red", marker="x", label="Not admitted")

    # 绘制决策边界
    x = [_[0] for _ in data]
    y = [(-theta[0] - theta[1] * i) / theta[2] for i in x]
    plt.plot(x, y, label="Decision Boundary")

    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    # 显示图例
    plt.legend()
    plt.show()


if __name__ == "__main__":
    alpha = 0.00001  # 学习率
    epochs = 400  # 迭代次数
    theta = initial_weights()  # 初始化参数
    x = []
    y = []
    for _ in data:
        x.append(_[:2])
        y.append(_[-1])
    # 训练模型
    theta = train(x, y, theta, alpha, epochs)
    print("Theta: ", theta)
    plot_result(data, theta)
