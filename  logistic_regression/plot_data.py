"""
Author: caodong0225
Date: 2024-06-04
Description: 绘制数据集
"""
import matplotlib.pyplot as plt
from dataset import data


def plot_data(dataset):
    """
    绘制数据集
    :param dataset:
    :return:
    """
    # 分组数据
    admitted_data = [(_[0], _[1]) for _ in dataset if _[2] == 1]
    not_admitted_data = [(_[0], _[1]) for _ in dataset if _[2] == 0]

    # 绘制数据
    plt.scatter(*zip(*admitted_data), c="blue", marker="o", label="Admitted")
    plt.scatter(*zip(*not_admitted_data), c="red", marker="x", label="Not admitted")

    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    # 显示图例
    plt.legend()
    plt.show()

plot_data(data)
