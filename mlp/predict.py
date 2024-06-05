"""
Author: caodong0225
Date: 2024-06-05
Description: 这是模型的预测代码
"""
import os
import matplotlib.pyplot as plt
import numpy as np

from train import Ann
from dataset import dataset


def plot_data(inp):
    """
    绘制图片
    :param inp: 输入图片
    :return:
    """
    pic = np.reshape(inp, (20, 20))
    # 将矩阵转置
    pic = pic.T
    plt.matshow(pic, cmap='gray')
    plt.show()


def plot_heatmap(predict_value):
    """
    绘制热力图
    :param predict_value: 预测值
    :return:
    """
    # 标签顺序为1，2，3，4，5，6，7，8，9，0
    label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    plt.bar(label, predict_value)
    plt.xlabel('Label')
    plt.ylabel('Probability')
    plt.title('Predict Probability')
    plt.show()


if __name__ == '__main__':
    model = Ann()
    model_structure = [400, 25, 10]  # 输入层400个神经元，隐藏层25个神经元，输出层10个神经元
    active_mod = ["leakyrelu", "sigmoid"]  # 第一层使用leakyrelu激活函数，第二层使用sigmoid激活函数
    loss_mod = "mse"  # 使用平方差损失函数
    # 判断是否有模型参数，如果有则加载模型参数，如果没有则初始化模型参数
    if "grad.txt" in os.listdir() and "bias.txt" in os.listdir():
        # 加载模型参数
        with open("grad.txt", "r", encoding="utf-8") as f:
            model.grad = eval(f.read())
        with open("bias.txt", "r", encoding="utf-8") as f:
            model.bias = eval(f.read())
    else:
        model.initial(model_structure)  # 初始化模型
    # 从dataset随机抽取一个样本
    idx = np.random.randint(len(dataset))
    # 预测
    pred = model.forward(dataset[idx], active_mod)
    # 绘制图片
    plot_data(dataset[idx])
    print("预测概率为：", pred)
    # 绘制热力图
    plot_heatmap(pred)
