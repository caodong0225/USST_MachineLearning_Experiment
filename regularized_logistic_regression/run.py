"""
Author: caodong0225
Date: 2024-06-04
Description: 正则化逻辑回归，机器学习算法
"""
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from dataset import data


def map_feature(dataset, data_dimension):
    """
    该函数的功能是将数据集扩充到多项式
    :param dataset: 数据集
    :param data_dimension: 数据维度
    :return: 扩充后的数据集
    """
    # 扩充数据集
    x_new = []
    for i in range(0, data_dimension + 1):
        for j in range(i + 1):
            x_new.append(dataset[0] ** (i - j) * dataset[1] ** j)
    return x_new


def initial_values(data_dimension):
    """
    该函数的功能是初始化参数
    :param data_dimension: 数据维度
    :return:
    """
    # 初始化参数
    model_theta = []
    for _ in range((data_dimension + 2) * (data_dimension + 1) // 2):
        model_theta.append(random.random())
    return model_theta


def sigmoid(model_theta, inp):
    """
    该函数的功能是计算sigmoid函数
    :param model_theta: 模型参数
    :param inp: 模型输入值
    :return:
    """
    z = 0
    for index, model_theta_index in enumerate(model_theta):
        z += model_theta_index * inp[index]
    return 1 / (1 + math.exp(-z))


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


def calculate_loss(inp, out, model_theta, regularization_weight):
    """
    该函数的功能是计算损失函数
    :param inp: 模型输入值
    :param out: 模型预测值
    :param model_theta: 模型参数
    :param regularization_weight: 正则化权重
    :return:
    """
    # 计算交叉熵损失函数
    loss = 0
    regularization_loss = sum(theta_ ** 2 for theta_ in model_theta) * regularization_weight / 2
    for i, input_value in enumerate(inp):
        h = sigmoid(model_theta, input_value)
        # 检查参数是否为正数
        if h <= 0 or h >= 1:
            # 如果参数不是正数，手动处理
            loss += 0  # 或者设置为其他默认值
        else:
            loss += -out[i] * math.log(h) - (1 - out[i]) * math.log(1 - h)
    return (loss + regularization_loss) / len(inp)


def calculate_decision_boundary(inp, out, model_theta, data_dimension):
    """
    该函数的功能是计算决策边界
    :param inp: 模型输入值
    :param out: 模型预测值
    :param model_theta: 模型参数
    :param data_dimension: 数据维度
    :return:
    """
    res = np.zeros((len(inp), len(out)))
    index = 0
    for i in range(0, data_dimension + 1):
        for j in range(i + 1):
            res_tem = np.multiply(np.power(inp, i - j), np.power(out, j))
            res_tem = np.multiply(res_tem, model_theta[index])
            res += res_tem
            index += 1
    return res


def calculate_gradient(inp, out, model_theta, regularization_weight):
    """
    该函数的功能是计算梯度
    :param inp: 模型输入值
    :param out: 模型预测值
    :param model_theta: 模型参数
    :param regularization_weight: 正则化权重
    :return:
    """
    res = [0] * len(model_theta)
    for index, inp_index in enumerate(inp):
        h = sigmoid(model_theta, inp_index)
        for j in range(len(model_theta)):
            res[j] += (h - out[index]) * inp_index[j]
    for j, model_theta_ in enumerate(model_theta):
        res[j] = (res[j] + regularization_weight * model_theta_) / len(inp)
    return res


def train(inp, out, model_theta, learning_rate, training_epochs, regularization_weight):
    """
    该函数的功能是训练模型
    :param inp: 模型输入值
    :param out: 模型预测值
    :param model_theta: 模型参数
    :param learning_rate: 学习率
    :param training_epochs: 训练次数
    :param regularization_weight: 正则化权重
    :return: 训练后的模型参数
    """
    # 该函数的功能是训练模型
    loss = []
    for _ in range(training_epochs):
        # 计算梯度
        grad = calculate_gradient(inp, out, model_theta, regularization_weight)
        for _, __ in enumerate(model_theta):
            model_theta[_] -= learning_rate * grad[_]
        loss.append(calculate_loss(inp, out, model_theta, regularization_weight))
    plot_loss(loss)
    return model_theta


def plot_result(dataset, model_theta, data_dimension):
    """
    该函数的功能是绘制结果
    :param dataset: 数据集
    :param model_theta: 模型参数
    :param data_dimension: 数据维度
    :return:
    """
    admitted_data = [(_[0], _[1]) for _ in dataset if _[2] == 1]
    not_admitted_data = [(_[0], _[1]) for _ in dataset if _[2] == 0]

    # 绘制数据
    plt.scatter(*zip(*admitted_data), c="blue", marker="o", label="y = 1")
    plt.scatter(*zip(*not_admitted_data), c="red", marker="x", label="y = 0")

    # 绘制决策边界
    x_ = np.arange(-1, 1.5, 0.01)
    y_ = np.arange(-1, 1.5, 0.01)
    x_, y_ = np.meshgrid(x_, y_)
    z = calculate_decision_boundary(x_, y_, model_theta, data_dimension)
    # 转化为网格
    plt.contour(x_, y_, z, 0)

    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    # 显示图例
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ALPHA = 0.04  # 学习率
    EPOCHS = 5000  # 迭代次数
    DIMENSION = 6  # 数据维度
    REGULARIZATION_PARAMETER = 1  # 正则惩罚系数
    theta = initial_values(DIMENSION)  # 初始化参数
    x = []
    y = []
    for _ in data:
        x.append(map_feature(_[:2], DIMENSION))
        y.append(_[-1])
    theta = train(x, y, theta, ALPHA, EPOCHS, REGULARIZATION_PARAMETER)
    plot_result(data, theta, DIMENSION)
