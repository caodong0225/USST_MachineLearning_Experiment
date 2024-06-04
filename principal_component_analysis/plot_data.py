"""
Author: caodong0225
Date: 2024-06-04
Description: 该文件的功能是显示数据
"""
import matplotlib.pyplot as plt
from dataset import data_pca, data_faces


def show_data(data):
    """
    显示数据
    :param data:
    :return:
    """
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    plt.scatter(x, y)
    plt.show()


# 显示部分人脸数据
def show_faces(data):
    """
    显示人脸数据
    :param data:
    :return:
    """
    _, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(data[i * 10 + j], cmap='gray')
            axes[i, j].axis('off')
    plt.show()


if __name__ == "__main__":
    show_data(data_pca)
    show_faces(data_faces)
