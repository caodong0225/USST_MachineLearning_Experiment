"""
Author: caodong0225
Date: 2024-06-04
Description: 实现逻辑回归
"""
# 实现逻辑回归
import math
import matplotlib.pyplot as plt
from dataset import data



def initial_weights():
    """
    初始化参数
    :return:
    """
    # 生成随机数
    # return [random.random(), random.random(), random.random()]
    return [-6.5, 0, 0]


def sigmoid(data_theta, x_):
    """
    计算sigmoid函数
    :param data_theta:
    :param x_:
    :return:
    """
    # 该函数的功能是计算sigmoid函数
    return 1 / (1 + math.exp(-(data_theta[0] + data_theta[1] * x_[0] + data_theta[2] * x_[1])))


def calculate_loss(x_, y_, model_theta):
    """
    计算损失函数
    :param x_:  模型预测值
    :param y_:  实际值
    :param model_theta:  模型参数
    :return:
    """
    # 计算交叉熵损失函数
    loss = 0
    for i, x_i in enumerate(x_):
        h = sigmoid(model_theta, x_i)
        # 检查参数是否为正数
        if h <= 0 or h >= 1:
            # 如果参数不是正数，手动处理
            loss += 0  # 或者设置为其他默认值
        else:
            loss += -y_[i] * math.log(h) - (1 - y_[i]) * math.log(1 - h)
    return loss / len(x_)


def calculate_gradient(x_, y_, model_theta):
    """
    计算梯度
    :param x_:  模型预测值
    :param y_:  实际值
    :param model_theta:  模型参数
    :return:
    """
    res = [0, 0, 0]
    for i, x_i in enumerate(x_):
        res[0] += sigmoid(model_theta, x_i) - y_[i]
        res[1] += (sigmoid(model_theta, x_i) - y_[i]) * x_i[0]
        res[2] += (sigmoid(model_theta, x_i) - y_[i]) * x_i[1]
    res[0] /= len(x_)
    res[1] /= len(x_)
    res[2] /= len(x_)
    # 返回最后的梯度值
    return res


def plot_loss(loss):
    """
    绘制损失函数
    :param loss:
    :return:
    """
    # 该函数的功能是绘制损失函数
    # loss: 损失值
    # 横坐标表示迭代次数，纵坐标表示损失值
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.show()


def train(x_, y_, model_theta, learning_rate, training_epochs):
    """
    训练模型
    :param x_:  模型预测值
    :param y_:  实际值
    :param model_theta:  模型参数
    :param learning_rate:  学习率
    :param training_epochs:  迭代次数
    :return:
    """
    # 该函数的功能是训练模型
    loss = []
    for _ in range(training_epochs):
        # 计算梯度
        grad = calculate_gradient(x_, y_, model_theta)
        for _, __ in enumerate(model_theta):
            model_theta[_] -= learning_rate * grad[_]
        loss.append(calculate_loss(x_, y_, model_theta))
    plot_loss(loss)
    return model_theta


def plot_result(dataset, model_theta):
    """
    绘制结果
    :param dataset:
    :param model_theta:
    :return:
    """
    admitted_data = [(_[0], _[1]) for _ in dataset if _[2] == 1]
    not_admitted_data = [(_[0], _[1]) for _ in dataset if _[2] == 0]

    # 绘制数据
    plt.scatter(*zip(*admitted_data), c="blue", marker="o", label="Admitted")
    plt.scatter(*zip(*not_admitted_data), c="red", marker="x", label="Not admitted")

    # 绘制决策边界
    x_ = [_[0] for _ in dataset]
    y_ = [(-model_theta[0] - model_theta[1] * i) / model_theta[2] for i in x_]
    plt.plot(x_, y_, label="Decision Boundary")

    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    # 显示图例
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ALPHA = 0.00001  # 学习率
    EPOCHS = 400  # 迭代次数
    theta = initial_weights()  # 初始化参数
    x = []
    y = []
    for _ in data:
        x.append(_[:2])
        y.append(_[-1])
    # 训练模型
    theta = train(x, y, theta, ALPHA, EPOCHS)
    print("Theta: ", theta)
    plot_result(data, theta)
